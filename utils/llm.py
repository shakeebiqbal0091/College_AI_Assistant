from langchain_groq import ChatGroq

from config import (
    CLASSIFIER_MODEL,
    GROQ_API_KEY,
    RESPONSE_MODEL,
)


if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is not set. "
        "Add it to your .env file."
    )


# --------------------------------------------------
# Classifier LLM
# --------------------------------------------------

classifier_llm = ChatGroq(
    model=CLASSIFIER_MODEL,
    temperature=0,
    api_key=GROQ_API_KEY,
)


# --------------------------------------------------
# Response LLM
# --------------------------------------------------

response_llm = ChatGroq(
    model=RESPONSE_MODEL,
    temperature=0.2,
    api_key=GROQ_API_KEY,
)