from operator import add
from typing import Annotated

from langgraph.graph import add_messages
from typing_extensions import TypedDict


class State(TypedDict):
    """
    Shared state for the College AI Assistant.

    Every LangGraph node can read from this state
    and return updates to it.
    """

    # Student's selected programme
    programme: str

    # Complete conversation history
    messages: Annotated[list, add_messages]

    # Cleaned/rephrased version of the user's latest question
    rewritten_query: str

    # Classification result:
    # academic, fee, both, or general
    query_type: str

    # Context retrieved from the RAG systems
    retrieved_context: Annotated[list[str], add]