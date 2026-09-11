"""Embed procedure docs from data/procedures/ into a local Chroma collection.

Usage:
    python ingest.py
"""
import os
import re

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

PROCEDURES_DIR = os.path.join(os.path.dirname(__file__), "data", "procedures")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = "procedures"
EMBEDDING_MODEL = "text-embedding-3-small"

DOMAIN_BY_FILENAME = {
    "nic_renewal.md": "nic",
    "nic_replacement_lost_damaged.md": "nic",
    "birth_certificate_gn.md": "civil_registration",
    "marriage_certificate_gn.md": "civil_registration",
    "passport_renewal.md": "passport",
    "driving_license_renewal.md": "driving_license",
}


def chunk_by_heading(text: str) -> list[str]:
    """Split a procedure doc into chunks along its '## ' section headings,
    keeping the doc title (the '# ' line) prefixed to every chunk so each
    chunk is self-contained for retrieval."""
    lines = text.strip().splitlines()
    title = lines[0].lstrip("# ").strip() if lines else ""

    chunks = []
    current = []
    for line in lines[1:]:
        if line.startswith("## ") and current:
            chunks.append("\n".join(current).strip())
            current = []
        current.append(line)
    if current:
        chunks.append("\n".join(current).strip())

    return [f"# {title}\n\n{chunk}" for chunk in chunks if chunk.strip()]


def load_documents() -> list[dict]:
    documents = []
    for filename in sorted(os.listdir(PROCEDURES_DIR)):
        if not filename.endswith(".md"):
            continue
        path = os.path.join(PROCEDURES_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        domain = DOMAIN_BY_FILENAME.get(filename, "unknown")
        for i, chunk in enumerate(chunk_by_heading(text)):
            documents.append(
                {
                    "id": f"{filename}::{i}",
                    "text": chunk,
                    "metadata": {"source": filename, "domain": domain},
                }
            )
    return documents


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set (needed for embeddings)")

    documents = load_documents()
    if not documents:
        raise RuntimeError(f"No procedure docs found in {PROCEDURES_DIR}")

    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key, model_name=EMBEDDING_MODEL
    )

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    client.delete_collection(COLLECTION_NAME) if COLLECTION_NAME in [
        c.name for c in client.list_collections()
    ] else None
    collection = client.create_collection(
        name=COLLECTION_NAME, embedding_function=embedding_fn
    )

    collection.add(
        ids=[d["id"] for d in documents],
        documents=[d["text"] for d in documents],
        metadatas=[d["metadata"] for d in documents],
    )

    print(f"Ingested {len(documents)} chunks from {PROCEDURES_DIR} into '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()
