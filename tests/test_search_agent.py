import pytest
from unittest.mock import patch, MagicMock
from owlai.examples.search_agent import create_search_agent, process_query


@pytest.fixture
def mock_tavily_search():
    with patch("owlai.tools.tavily_search.TavilySearchResults") as mock:
        # Configure the mock to return a sample search result
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = {
            "results": [
                {
                    "title": "AC Milan Latest Results",
                    "content": "AC Milan won their last match 2-0 against Inter Milan",
                    "url": "https://example.com/milan-results",
                }
            ]
        }
        mock.return_value = mock_instance
        yield mock


def test_search_agent_uses_tavily_tool(mock_tavily_search):
    """Test that the search agent actually uses the Tavily search tool when appropriate"""
    # Create the search agent
    agent = create_search_agent()

    # Process a query that should trigger the search tool
    query = "What was the last result of the AC Milan soccer team?"
    response = process_query(agent, query)

    # Verify the Tavily search tool was called
    mock_tavily_search.assert_called_once()

    # Verify the response contains information from the search results
    assert "AC Milan won their last match 2-0 against Inter Milan" in response.content


def test_search_agent_handles_future_matches(mock_tavily_search):
    """Test that the search agent uses the search tool for future match queries"""
    # Configure mock for future matches
    mock_tavily_search.return_value.invoke.return_value = {
        "results": [
            {
                "title": "AC Milan Upcoming Matches",
                "content": "AC Milan will play against Juventus on Sunday at 20:45",
                "url": "https://example.com/milan-schedule",
            }
        ]
    }

    # Create the search agent
    agent = create_search_agent()

    # Process a query about future matches
    query = "When is the AC Milan next playing?"
    response = process_query(agent, query)

    # Verify the Tavily search tool was called
    mock_tavily_search.assert_called_once()

    # Verify the response contains information from the search results
    assert "AC Milan will play against Juventus" in response.content
