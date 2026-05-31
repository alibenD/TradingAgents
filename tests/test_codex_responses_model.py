from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel

from tradingagents.llm_clients.codex_responses_model import CodexResponsesChatModel


class FakeStream:
    def __init__(self, final, events=None):
        self.final = final
        self.events = events or []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __iter__(self):
        return iter(self.events)

    def get_final_response(self):
        return self.final


class FakeResponses:
    def __init__(self, text='final answer'):
        self.kwargs = None
        self.text = text

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(type="output_text", text=self.text)],
                )
            ]
        )

    def stream(self, **kwargs):
        self.kwargs = kwargs
        return FakeStream(self.create(**kwargs))


class FakeClient:
    def __init__(self, text='final answer'):
        self.responses = FakeResponses(text)


def test_invoke_builds_codex_responses_payload():
    client = FakeClient()
    llm = CodexResponsesChatModel(
        model="gpt-5.4",
        api_key="token",
        base_url="https://chatgpt.com/backend-api/codex",
        client=client,
    )

    result = llm.invoke([SystemMessage(content="system"), HumanMessage(content="hello")])

    assert isinstance(result, AIMessage)
    assert result.content == "final answer"
    assert client.responses.kwargs["model"] == "gpt-5.4"
    assert client.responses.kwargs["instructions"] == "system"
    assert client.responses.kwargs["input"][0]["role"] == "user"
    assert client.responses.kwargs["input"][0]["content"][0]["text"] == "hello"
    assert client.responses.kwargs["store"] is False


def test_bind_tools_adds_function_tools():
    client = FakeClient()
    llm = CodexResponsesChatModel(
        model="gpt-5.4",
        api_key="token",
        base_url="https://chatgpt.com/backend-api/codex",
        client=client,
    )

    bound = llm.bind_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "d",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
    )
    bound.invoke("hello")

    assert client.responses.kwargs["tools"][0]["name"] == "lookup"
    assert client.responses.kwargs["tool_choice"] == "auto"


def test_bind_tools_accepts_langchain_tools():
    @tool
    def lookup(symbol: str) -> str:
        """Look up a symbol."""
        return symbol

    client = FakeClient()
    llm = CodexResponsesChatModel(
        model="gpt-5.4",
        api_key="token",
        base_url="https://chatgpt.com/backend-api/codex",
        client=client,
    )

    llm.bind_tools([lookup]).invoke("hello")

    assert client.responses.kwargs["tools"][0]["name"] == "lookup"
    assert client.responses.kwargs["tools"][0]["parameters"]["properties"]["symbol"]["type"] == "string"


def test_extracts_streamed_tool_calls():
    final = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="function_call",
                call_id="call_123",
                name="lookup",
                arguments='{"symbol":"AAPL"}',
            )
        ]
    )
    client = SimpleNamespace(responses=SimpleNamespace(stream=lambda **kwargs: FakeStream(final)))
    llm = CodexResponsesChatModel(
        model="gpt-5.4",
        api_key="token",
        base_url="https://chatgpt.com/backend-api/codex",
        client=client,
    )

    result = llm.invoke("hello")

    assert result.content == ""
    assert result.tool_calls == [
        {"name": "lookup", "args": {"symbol": "AAPL"}, "id": "call_123", "type": "tool_call"}
    ]


def test_converts_tool_roundtrip_messages():
    client = FakeClient()
    llm = CodexResponsesChatModel(
        model="gpt-5.4",
        api_key="token",
        base_url="https://chatgpt.com/backend-api/codex",
        client=client,
    )
    ai = AIMessage(
        content="",
        tool_calls=[{"name": "lookup", "args": {"symbol": "AAPL"}, "id": "call_123"}],
    )

    llm.invoke([ai, ToolMessage(content="price data", tool_call_id="call_123")])

    assert client.responses.kwargs["input"] == [
        {
            "type": "function_call",
            "call_id": "call_123",
            "name": "lookup",
            "arguments": '{"symbol": "AAPL"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_123",
            "output": "price data",
        },
    ]


class Pick(BaseModel):
    action: str


def test_with_structured_output_parses_json_response():
    client = FakeClient('{"action":"Buy"}')
    llm = CodexResponsesChatModel(
        model="gpt-5.4",
        api_key="token",
        base_url="https://chatgpt.com/backend-api/codex",
        client=client,
    )

    structured = llm.with_structured_output(Pick)
    result = structured.invoke("return JSON")

    assert result.action == "Buy"
    assert "Return only one valid JSON object" in client.responses.kwargs["instructions"]
    assert '"action"' in client.responses.kwargs["instructions"]
    assert client.responses.kwargs["tools"][0]["name"] == "structured_output"
    assert client.responses.kwargs["tool_choice"] == {"type": "function", "name": "structured_output"}
    assert client.responses.kwargs["parallel_tool_calls"] is False


def test_with_structured_output_parses_tool_call_args():
    final = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="function_call",
                call_id="call_structured",
                name="structured_output",
                arguments='{"action":"Sell"}',
            )
        ]
    )
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: FakeStream(final)))
    llm = CodexResponsesChatModel(
        model="gpt-5.4",
        api_key="token",
        base_url="https://chatgpt.com/backend-api/codex",
        client=client,
    )

    structured = llm.with_structured_output(Pick)
    result = structured.invoke("return structured")

    assert result.action == "Sell"
