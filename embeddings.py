"""Shared embedding function factory for ingest.py and agents/specialist.py.

Default is a local, free embedding model (all-MiniLM-L6-v2, runs on-device via
ONNX, no API key needed) - a deviation from the fixed tech stack's OpenAI
text-embedding-3-small, made because OpenAI billing wasn't available. Set
EMBEDDING_PROVIDER=openai in .env to switch back to OpenAI embeddings once
billing is set up (requires OPENAI_API_KEY).
"""
import os

from chromadb.utils import embedding_functions

OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"


def get_embedding_function():
    provider = os.getenv("EMBEDDING_PROVIDER", "local").lower()

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set (needed for OpenAI embeddings)")
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key, model_name=OPENAI_EMBEDDING_MODEL
        )

    return embedding_functions.DefaultEmbeddingFunction()
