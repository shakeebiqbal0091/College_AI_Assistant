from functools import lru_cache

import pdfplumber
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import (
    EMBEDDING_MODEL,
    FEE_PDF,
    FEE_TOP_K,
)


@lru_cache(maxsize=1)
def get_fee_retriever():
    """
    Load fee information from the fee PDF, convert it into
    searchable documents, and return a FAISS retriever.
    """

    documents = []

    # --------------------------------------------------
    # 1. Open the fee PDF
    # --------------------------------------------------

    with pdfplumber.open(FEE_PDF) as pdf:

        for page_number, page in enumerate(pdf.pages, start=1):

            # --------------------------------------------------
            # 2. Try to extract tables
            # --------------------------------------------------

            tables = page.extract_tables()

            for table in tables:

                if not table:
                    continue

                # Remove completely empty rows
                rows = [
                    row
                    for row in table
                    if row and any(cell for cell in row)
                ]

                if not rows:
                    continue

                # --------------------------------------------------
                # 3. Convert table into searchable text
                # --------------------------------------------------

                table_text = "\n".join(
                    " | ".join(
                        str(cell).strip() if cell is not None else ""
                        for cell in row
                    )
                    for row in rows
                )

                if table_text.strip():

                    documents.append(
                        Document(
                            page_content=table_text,
                            metadata={
                                "source": str(FEE_PDF),
                                "page": page_number,
                                "type": "fee_table",
                            },
                        )
                    )

            # --------------------------------------------------
            # 4. Also extract normal page text
            # --------------------------------------------------

            page_text = page.extract_text()

            if page_text and page_text.strip():

                documents.append(
                    Document(
                        page_content=page_text,
                        metadata={
                            "source": str(FEE_PDF),
                            "page": page_number,
                            "type": "fee_text",
                        },
                    )
                )

    # --------------------------------------------------
    # 5. Make sure we extracted something
    # --------------------------------------------------

    if not documents:
        raise ValueError(
            "No readable fee information was found in the fee PDF."
        )

    # --------------------------------------------------
    # 6. Create embeddings
    # --------------------------------------------------

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    # --------------------------------------------------
    # 7. Create FAISS vector store
    # --------------------------------------------------

    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embeddings,
    )

    # --------------------------------------------------
    # 8. Create retriever
    # --------------------------------------------------

    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": FEE_TOP_K
        }
    )

    return retriever