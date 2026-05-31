import pytest

from tradingagents.graph.trading_graph import TradingAgentsGraph


@pytest.mark.unit
def test_market_tool_node_exposes_verified_market_snapshot():
    graph = TradingAgentsGraph.__new__(TradingAgentsGraph)
    market_node = TradingAgentsGraph._create_tool_nodes(graph)["market"]

    assert "get_verified_market_snapshot" in market_node.tools_by_name
