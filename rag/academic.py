from functools import lru_cache

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from config import (
    ACADEMIC_PDF,
    ACADEMIC_TOP_K,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
)


@lru_cache(maxsize=1)
def get_academic_retriever():
    """
    Create and return the academic PDF retriever.

    The result is cached so that the PDF does not need to be
    loaded and embedded every time the retriever is requested.
    """

    # --------------------------------------------------
    # 1. Load the academic PDF
    # --------------------------------------------------

    loader = PyPDFLoader(str(ACADEMIC_PDF))

    documents = loader.load()

    if not documents:
        raise ValueError("Academic PDF contains no readable content.")

    # --------------------------------------------------
    # 2. Split the PDF into smaller chunks
    # --------------------------------------------------

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = text_splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No text chunks were created from the academic PDF.")

    # --------------------------------------------------
    # 3. Create embeddings
    # --------------------------------------------------

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    # --------------------------------------------------
    # 4. Create FAISS vector store
    # --------------------------------------------------

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    # --------------------------------------------------
    # 5. Create retriever
    # --------------------------------------------------

    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": ACADEMIC_TOP_K
        }
    )

    return retriever