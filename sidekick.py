"""The Sidekick: a create_agent worker wrapped in a homemade evaluator loop.

The worker is a single create_agent (Layer 3). Around it we run our own loop that checks
the worker's answer against the user's success criteria, and either accepts it, sends it
back for another attempt, or returns to the user with a question. Middleware gives the
worker a plan it shares with the UI, guardrails for PII and runaway costs, and a pause
for human approval before sensitive actions.
"""

import os
import re
import uuid
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    PIIMiddleware,
    TodoListMiddleware,
)
from langchain_core.messages import RemoveMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Command

from sidekick_tools import get_all_tools

load_dotenv(override=True)

HERE = os.path.dirname(os.path.abspath(__file__))
SANDBOX = os.path.join(HERE, "sandbox")
MAX_ATTEMPTS = 3


class EvaluatorOutput(BaseModel):
    feedback: str = Field(description="Feedback on the assistant's response")
    success_criteria_met: bool = Field(description="Whether the success criteria have been met")
    user_input_needed: bool = Field(
        description="True if the assistant has a question, needs clarification, or is stuck and needs the user"
    )


WORKER_PROMPT = """You are Sidekick, a capable personal assistant who completes tasks for the user.
You have a real web browser, a sandbox filesystem, web search, Wikipedia, and the ability to send push notifications.
When you use the browser, navigate to a page and read it with a snapshot rather than clicking around unnecessarily.
Dismiss cookie banners and popups yourself by clicking in the browser. If you reach something only a human can do,
like logging in, a captcha, or two-factor authentication, use the request_human_help tool to tell the user exactly
what to do in your browser window, then carry on once they have done it.
For flight searches, use Google Flights in your browser: go straight to https://www.google.com/travel/flights?q=...
with a natural language query like "flights from New York to London leaving 14 July returning 21 July".
Keep working on the task until the success criteria are met, or until you genuinely need to ask the user a question.
If you have a question, ask it plainly. When you are finished, give your final answer clearly,
saying what you did, what you produced, and what you found."""


_MISSING_TOOL_CALL_IDS = re.compile(
    r"tool_call_ids did not have response messages:\s*(.+?)(?:['\"]|$)"
)


def _call_id(call) -> str | None:
    if isinstance(call, dict):
        return call.get("id")
    return getattr(call, "id", None)


def _tool_call_ids(message) -> list[str]:
    """IDs OpenAI will treat as tool_calls on this assistant message.

    Chat Completions also serializes invalid_tool_calls and additional_kwargs
    tool_calls, so those need responses too.
    """
    ids = []
    extra = {}
    if isinstance(message, dict):
        calls = list(message.get("tool_calls") or [])
        extra = message.get("additional_kwargs") or {}
    else:
        calls = list(getattr(message, "tool_calls", None) or [])
        calls += list(getattr(message, "invalid_tool_calls", None) or [])
        extra = getattr(message, "additional_kwargs", None) or {}
    calls += list(extra.get("tool_calls") or [])
    for call in calls:
        if call_id := _call_id(call):
            ids.append(call_id)
    return list(dict.fromkeys(ids))


def _response_tool_call_id(message) -> str | None:
    call_id = getattr(message, "tool_call_id", None)
    if call_id:
        return str(call_id)
    if isinstance(message, dict):
        call_id = message.get("tool_call_id")
        return str(call_id) if call_id else None
    return None


def _placeholder_tool_message(call_id: str) -> ToolMessage:
    return ToolMessage(
        content="This tool call did not return a result. Continue with the information you have.",
        tool_call_id=call_id,
    )


def _repair_tool_pairing(messages: list, extra_ids: list[str] | None = None) -> list:
    """Insert placeholder ToolMessages so every tool_call is answered before the next
    non-tool message. OpenAI rejects the request otherwise, and parallel tools that
    return Command can drop a sibling result."""
    extra_ids = extra_ids or []
    last_tool_index = max(
        (index for index, message in enumerate(messages) if _tool_call_ids(message)),
        default=None,
    )
    repaired = []
    changed = False
    i = 0
    while i < len(messages):
        message = messages[i]
        needed = _tool_call_ids(message)
        if extra_ids and i == last_tool_index:
            needed = list(dict.fromkeys([*needed, *extra_ids]))
        repaired.append(message)
        i += 1
        if not needed:
            continue
        seen = set()
        while i < len(messages):
            tool_call_id = _response_tool_call_id(messages[i])
            if tool_call_id not in needed:
                break
            repaired.append(messages[i])
            seen.add(tool_call_id)
            i += 1
        for call_id in needed:
            if call_id not in seen:
                repaired.append(_placeholder_tool_message(call_id))
                changed = True
    return repaired if changed else messages


def _missing_ids_from_error(error: BaseException) -> list[str]:
    match = _MISSING_TOOL_CALL_IDS.search(str(error))
    if not match:
        return []
    return [part.strip() for part in match.group(1).split(",") if part.strip()]


class TolerateToolErrors(AgentMiddleware):
    """Hand tool failures back to the model as a message so it can recover, rather than
    crashing the run. Tools that touch the outside world, like a browser, fail now and then."""

    async def awrap_tool_call(self, request, handler):
        try:
            return await handler(request)
        except Exception as error:
            return ToolMessage(
                content=f"That tool call failed: {error}. Try another approach.",
                tool_call_id=request.tool_call["id"],
            )


class RepairToolPairing(AgentMiddleware):
    """Keep the OpenAI message list valid when a tool_call never got a ToolMessage.

    That happens if a parallel tool returns Command and drops a sibling result, or if
    the evaluator loop appends a user message after a run that stopped mid-tools."""

    def _state_update(self, state):
        messages = state.get("messages") or []
        repaired = _repair_tool_pairing(messages)
        if repaired is messages:
            return None
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *repaired]}

    def before_model(self, state, runtime):
        return self._state_update(state)

    async def abefore_model(self, state, runtime):
        return self._state_update(state)

    def _with_repaired_messages(self, request, extra_ids=None):
        repaired = _repair_tool_pairing(request.messages, extra_ids)
        if repaired is request.messages:
            return request
        return request.override(messages=repaired)

    def wrap_model_call(self, request, handler):
        request = self._with_repaired_messages(request)
        try:
            return handler(request)
        except Exception as error:
            missing = _missing_ids_from_error(error)
            retried = self._with_repaired_messages(request, missing) if missing else request
            if retried is request:
                raise
            return handler(retried)

    async def awrap_model_call(self, request, handler):
        request = self._with_repaired_messages(request)
        try:
            return await handler(request)
        except Exception as error:
            missing = _missing_ids_from_error(error)
            retried = self._with_repaired_messages(request, missing) if missing else request
            if retried is request:
                raise
            return await handler(retried)


class Sidekick:
    def __init__(self):
        self.sidekick_id = str(uuid.uuid4())
        self.memory = InMemorySaver()
        self.tools = None
        self.sessions = None
        self.worker: Any = None
        self.evaluator: Any = None
        self.task = ""
        self.success_criteria = ""
        self.attempts = 0
        self.paused = False
        self.pending_actions = 0
        self.todos = []

    async def setup(self):
        os.makedirs(SANDBOX, exist_ok=True)
        self.tools, self.sessions = await get_all_tools(SANDBOX)
        self.worker = create_agent(
            model="openai:gpt-5.4-mini",
            tools=self.tools,
            system_prompt=f"{WORKER_PROMPT}\nToday is {datetime.now():%A %d %B %Y}.",
            middleware=[  # type: ignore[arg-type]
                RepairToolPairing(),
                TolerateToolErrors(),
                TodoListMiddleware(),
                PIIMiddleware("email"),
                PIIMiddleware("credit_card", apply_to_tool_results=True),
                ModelCallLimitMiddleware(run_limit=30),
                HumanInTheLoopMiddleware(
                    interrupt_on={"send_push_notification": True, "request_human_help": True}
                ),
            ],
            checkpointer=self.memory,
        )
        self.evaluator = ChatOpenAI(model="gpt-5.4-mini").with_structured_output(EvaluatorOutput)

    async def evaluate(
        self, message: str, success_criteria: str, last_reply: str, tools_used: list[str]
    ) -> EvaluatorOutput:
        prompt = f"""You decide whether an assistant has met the success criteria for a task.

The user's request was:
{message}

The success criteria are:
{success_criteria}

The tools the assistant called while working, in order:
{", ".join(tools_used) or "none"}

The assistant's most recent reply was:
{last_reply}

Decide whether the success criteria are met, using the tool calls as evidence of what was actually done.
Also decide whether the assistant needs more input from the user, either because it asked a question,
needs clarification, or seems stuck. Give brief, concrete feedback."""
        return await self.evaluator.ainvoke(prompt)

    async def run_turn(self, message: str, success_criteria: str, history: list) -> list:
        """One turn of conversation: the worker attempts the task and the evaluator checks it,
        retrying with feedback up to MAX_ATTEMPTS. If the worker pauses for approval, this
        returns straight away with paused set, and resume() continues the same turn."""
        self.task = message
        self.success_criteria = success_criteria or "The answer should be clear, correct and complete"
        self.attempts = 0
        self.todos = []
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": f"{message}\n\nThe success criteria for this task are: {self.success_criteria}",
                }
            ]
        }
        return await self._advance(payload, history + [{"role": "user", "content": message}])

    async def resume(self, history: list) -> list:
        """Approve the actions the worker paused on, and continue the turn."""
        payload = Command(resume={"decisions": [{"type": "approve"}] * self.pending_actions})
        return await self._advance(payload, history)

    async def _interrupt_actions(self, result, config) -> list | None:
        interrupts = result.get("__interrupt__") if isinstance(result, dict) else None
        if not interrupts:
            snapshot = await self.worker.aget_state(config)
            interrupts = snapshot.interrupts
        if not interrupts:
            return None
        first = interrupts[0]
        value = first.value if hasattr(first, "value") else first.get("value")
        return list((value or {}).get("action_requests") or [])

    async def _advance(self, payload, history: list) -> list:
        config = {"configurable": {"thread_id": self.sidekick_id}}
        pairing_retries = 0
        while True:
            result: Any = None
            try:
                async for result in self.worker.astream(payload, config=config, stream_mode="values"):
                    self.todos = (result or {}).get("todos", self.todos)
            except Exception as error:
                missing = _missing_ids_from_error(error)
                if not missing or pairing_retries >= 1:
                    raise
                pairing_retries += 1
                snapshot = await self.worker.aget_state(config)
                repaired = _repair_tool_pairing(snapshot.values.get("messages") or [], missing)
                if repaired is not snapshot.values.get("messages"):
                    await self.worker.aupdate_state(
                        config, {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *repaired]}
                    )
                payload = None
                continue
            pairing_retries = 0

            actions = await self._interrupt_actions(result, config)
            if actions is not None:
                self.paused = True
                self.pending_actions = len(actions)
                described = "\n".join(action["description"] for action in actions)
                return history + [{"role": "assistant", "content": f"Waiting for your approval:\n{described}"}]

            self.paused = False
            reply = result["messages"][-1].content
            tools_used = [
                call["name"] for m in result["messages"] for call in (getattr(m, "tool_calls", None) or [])
            ]
            self.attempts += 1
            verdict = await self.evaluate(self.task, self.success_criteria, reply, tools_used)
            if verdict.success_criteria_met or verdict.user_input_needed or self.attempts >= MAX_ATTEMPTS:
                return history + [
                    {"role": "assistant", "content": reply},
                    {"role": "assistant", "content": f"Evaluator: {verdict.feedback}"},
                ]
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Your last response did not meet the success criteria. "
                        f"Here is the feedback: {verdict.feedback}. Please keep working and address it.",
                    }
                ]
            }

    def cleanup(self):
        """Shut down the MCP servers; the browser window closes."""
        if self.sessions:
            self.sessions.stop()
