"""Specialist agent: given a domain and query, retrieves the top-k relevant
chunks from the vector store (filtered to that domain) and drafts an English
answer grounded strictly in the retrieved text. Must not add facts that
weren't retrieved."""
import os
from dataclasses import dataclass, field

import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
SPECIALIST_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "text-embedding-3-small"
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = "procedures"
TOP_K = 4

SYSTEM_PROMPT = """You are a specialist assistant answering questions about a \
single Sri Lankan government procedure domain, using ONLY the context passages \
provided below. Each passage is labeled with its source file.

Rules:
- Answer strictly using facts present in the context. Do not add steps, fees, \
documents, or locations that are not stated in the context.
- If the context does not contain enough information to answer, say so plainly \
instead of guessing.
- Write in plain, clear English aimed at a citizen with no bureaucratic \
background, as a numbered list of steps where applicable.
- Do not mention "the context" or "the passages" in your answer - just answer \
as if you know this directly.
"""


@dataclass
class SpecialistResult:
    answer: str
    sources: list[str] = field(default_factory=list)


def _llm_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)


def _collection():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set (needed for embeddings)")
    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai_api_key, model_name=EMBEDDING_MODEL
    )
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return client.get_collection(COLLECTION_NAME, embedding_function=embedding_fn)


def retrieve(domain: str, query: str, top_k: int = TOP_K):
    collection = _collection()
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        where={"domain": domain},
    )
    documents = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    return documents, metadatas


def draft_answer(domain: str, query: str, top_k: int = TOP_K) -> SpecialistResult:
    documents, metadatas = retrieve(domain, query, top_k)

    if not documents:
        return SpecialistResult(
            answer=(
                "I don't have information on this in the knowledge base yet. "
                "Please check with your local government office."
            ),
            sources=[],
        )

    sources = sorted({m["source"] for m in metadatas})
    context = "\n\n---\n\n".join(
        f"[Source: {m['source']}]\n{doc}" for doc, m in zip(documents, metadatas)
    )

    response = _llm_client().chat.completions.create(
        model=SPECIALIST_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ],
        temperature=0,
    )
    answer = response.choices[0].message.content.strip()

    return SpecialistResult(answer=answer, sources=sources)


if __name__ == "__main__":
    import sys

    domain = sys.argv[1] if len(sys.argv) > 1 else "nic"
    query = " ".join(sys.argv[2:]) or "lost my NIC"
    result = draft_answer(domain, query)
    print(result.answer)
    print("\nSources:", result.sources)
