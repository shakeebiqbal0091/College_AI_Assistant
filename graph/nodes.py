from typing import Literal

from pydantic import BaseModel, Field

from state import State
from utils.llm import classifier_llm, response_llm

from rag.academic import get_academic_retriever
from rag.fees import get_fee_retriever


class QueryClassification(BaseModel):
    """Structured result returned by the query classifier."""

    query_type: Literal[
        "academic",
        "fee",
        "both",
        "general",
    ] = Field(
        description=(
            "The category of the student's question: "
            "academic, fee, both, or general."
        )
    )


def rewrite_node(state: State) -> dict:
    """
    Rewrite the student's latest question into a clear,
    standalone query using the conversation history.
    """

    messages = state["messages"]
    latest_message = messages[-1]

    conversation = "\n".join(
        f"{message.type}: {message.content}"
        for message in messages[-6:]
    )

    prompt = f"""
You are a query rewriting assistant for a college AI assistant.

Your task is to rewrite the student's latest question into
a clear, standalone question.

Rules:
1. Preserve the student's original meaning.
2. Use previous conversation context when necessary.
3. Include the student's programme when relevant.
4. Do not answer the question.
5. Return ONLY the rewritten question.
6. If the question is already clear, return it unchanged.

Conversation:
{conversation}

Latest student question:
{latest_message.content}
"""

    response = response_llm.invoke(prompt)

    rewritten_query = str(response.content).strip()

    return {
        "rewritten_query": rewritten_query
    }


def classifier_node(state: State) -> dict:
    """
    Classify the rewritten student query.

    Possible categories:

    academic
    fee
    both
    general
    """

    rewritten_query = state["rewritten_query"]
    programme = state["programme"]

    prompt = f"""
You are the query classifier for a college AI assistant.

Classify the student's question into EXACTLY ONE category.

Categories:

academic
- courses
- subjects
- syllabus
- exams
- attendance
- assignments
- academic rules
- academic policies
- programmes
- semesters
- study requirements

fee
- tuition fees
- admission fees
- semester fees
- examination fees
- registration charges
- payments
- refunds
- financial charges

both
- Use this ONLY when the question requires BOTH
  academic information AND fee information.

general
- greetings
- casual conversation
- thanks
- unrelated questions

Student programme:
{programme}

Student query:
{rewritten_query}

Return ONLY valid JSON.

The JSON must have exactly this structure:

{{
    "query_type": "academic"
}}

The value of query_type MUST be exactly one of:

academic
fee
both
general
"""

    response = classifier_llm.invoke(
        prompt,
        response_format={"type": "json_object"},
    )

    result = QueryClassification.model_validate_json(
        response.content
    )

    return {
        "query_type": result.query_type
    }


def academic_rag_node(state: State) -> dict:
    """
    Retrieve relevant information from the academic handbook.
    """

    query = state["rewritten_query"]

    retriever = get_academic_retriever()

    documents = retriever.invoke(query)

    context = [
        document.page_content
        for document in documents
    ]

    return {
        "retrieved_context": context
    }


def fee_rag_node(state: State) -> dict:
    """
    Retrieve relevant fee information from the fee document.
    """

    query = state["rewritten_query"]

    retriever = get_fee_retriever()

    documents = retriever.invoke(query)

    context = [
        document.page_content
        for document in documents
    ]

    return {
        "retrieved_context": context
    }


def both_rag_node(state: State) -> dict:
    """
    Retrieve information from both academic and fee sources.
    """

    query = state["rewritten_query"]

    academic_retriever = get_academic_retriever()
    fee_retriever = get_fee_retriever()

    academic_documents = academic_retriever.invoke(query)
    fee_documents = fee_retriever.invoke(query)

    academic_context = [
        document.page_content
        for document in academic_documents
    ]

    fee_context = [
        document.page_content
        for document in fee_documents
    ]

    return {
        "retrieved_context": (
            academic_context + fee_context
        )
    }


def general_node(state: State) -> dict:
    """
    Generate a response for general conversation.
    """

    messages = state["messages"]

    latest_message = messages[-1].content

    prompt = f"""
You are a friendly college AI assistant.

Respond naturally to the student's message.

Do not invent college-specific information.

Student message:
{latest_message}
"""

    response = response_llm.invoke(prompt)

    return {
        "retrieved_context": [
            str(response.content).strip()
        ]
    }