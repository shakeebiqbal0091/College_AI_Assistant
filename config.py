import os
from pathlib import Path

from dotenv import load_dotenv


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

ACADEMIC_PDF = BASE_DIR / "academics_handbook.pdf"
FEE_PDF = BASE_DIR / "fee_structure.pdf"


# --------------------------------------------------
# API configuration
# --------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# --------------------------------------------------
# LLM models
# --------------------------------------------------

CLASSIFIER_MODEL = "openai/gpt-oss-20b"

RESPONSE_MODEL = "openai/gpt-oss-120b"


# --------------------------------------------------
# Embedding model
# --------------------------------------------------

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# --------------------------------------------------
# RAG configuration
# --------------------------------------------------

CHUNK_SIZE = 800

CHUNK_OVERLAP = 100

ACADEMIC_TOP_K = 4

FEE_TOP_K = 6