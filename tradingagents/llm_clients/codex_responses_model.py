"""LangChain chat model for the ChatGPT Codex Responses backend."""

from __future__ import annotations

import json
import re
from types import SimpleNamespace
from typing import Any, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict, Field, PrivateAttr

from .codex_auth import DEFAULT_CODEX_BASE_URL, build_httpx_client_for_url


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content or "")


def _to_responses_content(content: Any, *, assistant: bool = False) -> list[dict[str, Any]]:
    text_type = "output_text" if assistant else "input_text"
    text = _message_text(content)
    return [{"type": text_type, "text": text}] if text else []


def _message_role(message: BaseMessage) -> str:
    if isinstance(message, AIMessage):
        return "assistant"
    return "user"


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    value = getattr(obj, key, None)
    if value is None and isinstance(obj, dict):
        value = obj.get(key, default)
    return value if value is not None else default


def _tool_args_to_json(args: Any) -> str:
    if isinstance(args, str):
        return args.strip() or "{}"
    if isinstance(args, dict):
        return json.dumps(args, ensure_ascii=False)
    return json.dumps({} if args is None else args, ensure_ascii=False)


def _convert_messages(messages: list[BaseMessage]) -> tuple[str, list[dict[str, Any]]]:
    instructions: list[str] = []
    input_items: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            text = _message_text(message.content).strip()
            if text:
                instructions.append(text)
            continue
        if isinstance(message, ToolMessage):
            tool_call_id = str(message.tool_call_id or "").strip()
            if tool_call_id:
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_call_id,
                        "output": _message_text(message.content),
                    }
                )
            continue
        role = _message_role(message)
        content = _to_responses_content(message.content, assistant=role == "assistant")
        if content:
            input_items.append({"role": role, "content": content})
        if isinstance(message, AIMessage):
            for tool_call in message.tool_calls or []:
                name = str(tool_call.get("name") or "").strip()
                call_id = str(tool_call.get("id") or "").strip()
                if not name or not call_id:
                    continue
                input_items.append(
                    {
                        "type": "function_call",
                        "call_id": call_id,
                        "name": name,
                        "arguments": _tool_args_to_json(tool_call.get("args")),
                    }
                )
    return "\n\n".join(instructions), input_items


def _convert_tools(tools: Sequence[Any] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    converted: list[dict[str, Any]] = []
    for item in tools:
        try:
            item = convert_to_openai_tool(item)
        except Exception:
            pass
        if not isinstance(item, dict):
            continue
        fn = item.get("function") if item.get("type") == "function" else item
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        converted.append(
            {
                "type": "function",
                "name": name,
                "description": fn.get("description", ""),
                "strict": False,
                "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
            }
        )
    return converted or None


def _extract_response_text(response: Any) -> str:
    output = _get_value(response, "output")
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            content = _get_value(item, "content")
            if isinstance(content, list):
                for block in content:
                    if _get_value(block, "type") == "output_text":
                        text = _get_value(block, "text", "")
                        if text:
                            parts.append(text)
                    elif isinstance(block, dict) and block.get("type") == "output_text":
                        text = block.get("text")
                        if isinstance(text, str):
                            parts.append(text)
        if parts:
            return "\n".join(parts)
    output_text = _get_value(response, "output_text")
    if isinstance(output_text, str):
        return output_text
    return ""


def _extract_tool_calls(response: Any) -> list[dict[str, Any]]:
    tool_calls: list[dict[str, Any]] = []
    output = _get_value(response, "output")
    if not isinstance(output, list):
        return tool_calls
    for item in output:
        if _get_value(item, "type") != "function_call":
            continue
        name = str(_get_value(item, "name", "") or "").strip()
        call_id = str(_get_value(item, "call_id", "") or _get_value(item, "id", "") or "").strip()
        raw_args = _get_value(item, "arguments", "{}")
        if not name or not call_id:
            continue
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args or "{}")
            except json.JSONDecodeError:
                args = {"__raw_arguments": raw_args}
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            args = {}
        tool_calls.append({"id": call_id, "name": name, "args": args})
    return tool_calls


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if match:
        return match.group(0)
    raise ValueError("No JSON object found in Codex structured output")


class _CodexStructuredRunnable(Runnable):
    def __init__(self, llm: "CodexResponsesChatModel", schema: Any):
        self.llm = llm
        self.schema = schema

    def _schema_instruction(self) -> str:
        if hasattr(self.schema, "model_json_schema"):
            schema_json = json.dumps(self.schema.model_json_schema(), ensure_ascii=False)
        elif hasattr(self.schema, "schema"):
            schema_json = json.dumps(self.schema.schema(), ensure_ascii=False)
        else:
            schema_json = str(self.schema)
        return (
            "Return only one valid JSON object that conforms to this JSON Schema. "
            "Do not include Markdown fences, prose, commentary, or extra keys.\n"
            f"{schema_json}"
        )

    def _with_schema_instruction(self, input: Any) -> Any:
        instruction = SystemMessage(content=self._schema_instruction())
        if isinstance(input, str):
            return [instruction, HumanMessage(content=input)]
        if isinstance(input, BaseMessage):
            return [instruction, input]
        if isinstance(input, list) and all(isinstance(item, BaseMessage) for item in input):
            return [instruction, *input]
        return input

    def _tool_schema(self) -> dict[str, Any]:
        if hasattr(self.schema, "model_json_schema"):
            parameters = self.schema.model_json_schema()
        elif hasattr(self.schema, "schema"):
            parameters = self.schema.schema()
        else:
            parameters = {"type": "object", "properties": {}}
        return {
            "type": "function",
            "function": {
                "name": "structured_output",
                "description": "Return the requested structured response.",
                "parameters": parameters,
            },
        }

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        structured_llm = self.llm.bind_tools([self._tool_schema()])
        structured_llm._tool_choice = {"type": "function", "name": "structured_output"}
        structured_llm._parallel_tool_calls = False
        response = structured_llm.invoke(self._with_schema_instruction(input), config=config, **kwargs)
        if response.tool_calls:
            raw_args = response.tool_calls[0].get("args", {})
            raw_json = json.dumps(raw_args, ensure_ascii=False)
            if hasattr(self.schema, "model_validate_json"):
                return self.schema.model_validate_json(raw_json)
            return self.schema.parse_raw(raw_json)
        raw_json = _extract_json_object(str(response.content or ""))
        if hasattr(self.schema, "model_validate_json"):
            return self.schema.model_validate_json(raw_json)
        return self.schema.parse_raw(raw_json)


class CodexResponsesChatModel(BaseChatModel):
    """Minimal LangChain chat model for Codex OAuth Responses calls."""

    model_name: str = Field(alias="model")
    api_key: str
    base_url: str = DEFAULT_CODEX_BASE_URL
    timeout: float | None = None
    max_retries: int | None = None
    reasoning_effort: str = "medium"
    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    _client: Any = PrivateAttr(default=None)
    _bound_tools: list[dict[str, Any]] | None = PrivateAttr(default=None)
    _tool_choice: Any = PrivateAttr(default="auto")
    _parallel_tool_calls: bool = PrivateAttr(default=True)

    def __init__(self, **data: Any):
        client = data.pop("client", None)
        super().__init__(**data)
        self._client = client

    @property
    def _llm_type(self) -> str:
        return "codex-oauth-responses"

    @property
    def model(self) -> str:
        return self.model_name

    def _default_client(self) -> Any:
        from openai import OpenAI

        kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "base_url": self.base_url,
        }
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        if self.max_retries is not None:
            kwargs["max_retries"] = self.max_retries
        kwargs["http_client"] = build_httpx_client_for_url(self.base_url, timeout=self.timeout)
        return OpenAI(**kwargs)

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = self._default_client()
        return self._client

    def _build_kwargs(self, messages: list[BaseMessage]) -> dict[str, Any]:
        instructions, input_items = _convert_messages(messages)
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "instructions": instructions,
            "input": input_items,
            "store": False,
            "reasoning": {"effort": self.reasoning_effort, "summary": "auto"},
            "include": ["reasoning.encrypted_content"],
        }
        tools = _convert_tools(self._bound_tools)
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = self._tool_choice
            kwargs["parallel_tool_calls"] = self._parallel_tool_calls
        return kwargs

    def _stream_response(self, request: dict[str, Any]) -> Any:
        responses = self._get_client().responses
        create_method = getattr(responses, "create", None)
        if callable(create_method):
            stream_or_response = create_method(**request, stream=True)
            return self._collect_stream_or_response(stream_or_response)

        stream_method = getattr(responses, "stream", None)
        if not callable(stream_method):
            raise RuntimeError("Codex Responses client exposes neither create() nor stream().")

        return self._collect_stream_or_response(stream_method(**request))

    def _collect_stream_or_response(self, stream_or_response: Any) -> Any:
        if hasattr(stream_or_response, "output"):
            return stream_or_response
        collected_output_items: list[Any] = []
        collected_text_deltas: list[str] = []
        saw_function_call = False
        terminal_response = None
        stream = stream_or_response
        close_after = None
        if hasattr(stream_or_response, "__enter__"):
            stream = stream_or_response.__enter__()
            close_after = stream_or_response
        try:
            for event in stream:
                event_type = str(_get_value(event, "type", "") or "")
                if event_type == "response.output_item.done":
                    item = _get_value(event, "item")
                    if item is not None:
                        collected_output_items.append(item)
                elif "output_text.delta" in event_type:
                    delta = _get_value(event, "delta", "")
                    if isinstance(delta, str) and delta:
                        collected_text_deltas.append(delta)
                elif "function_call" in event_type:
                    saw_function_call = True
                elif event_type in {"response.completed", "response.incomplete", "response.failed"}:
                    terminal_response = _get_value(event, "response")

            if terminal_response is None and hasattr(stream, "get_final_response"):
                terminal_response = stream.get_final_response()
        finally:
            if close_after is not None:
                close_after.__exit__(None, None, None)
            else:
                close_fn = getattr(stream_or_response, "close", None)
                if callable(close_fn):
                    close_fn()

        final = terminal_response or SimpleNamespace(output=[])

        output = _get_value(final, "output")
        if not isinstance(output, list):
            try:
                final.output = []
                output = final.output
            except Exception:
                final = SimpleNamespace(output=[])
                output = final.output
        if not output:
            if collected_output_items:
                final.output = list(collected_output_items)
            elif collected_text_deltas and not saw_function_call:
                final.output = [
                    SimpleNamespace(
                        type="message",
                        role="assistant",
                        status="completed",
                        content=[
                            SimpleNamespace(
                                type="output_text",
                                text="".join(collected_text_deltas),
                            )
                        ],
                    )
                ]
        return final

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        response = self._stream_response(self._build_kwargs(messages))
        message = AIMessage(
            content=_extract_response_text(response),
            tool_calls=_extract_tool_calls(response),
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "CodexResponsesChatModel":
        del kwargs
        bound = self.model_copy()
        bound._client = self._client
        bound._bound_tools = list(tools)
        bound._tool_choice = "auto"
        bound._parallel_tool_calls = True
        return bound

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable:
        del kwargs
        return _CodexStructuredRunnable(self, schema)
