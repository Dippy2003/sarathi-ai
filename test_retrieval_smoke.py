"""Standalone smoke test for the Chroma retrieval index built by ingest.py.

Confirms that a query like "lost my NIC" returns the NIC Replacement doc,
not the plain NIC Renewal doc. Run `python ingest.py` first.

Usage:
    python test_retrieval_smoke.py
"""
import os

import chromadb
from dotenv import load_dotenv

from embeddings import get_embedding_function

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = "procedures"

QUERIES = [
    ("lost my NIC", "nic_replacement_lost_damaged.md"),
    ("my NIC is expiring soon, how do I renew it", "nic_renewal.md"),
    ("how do I get a copy of my birth certificate", "birth_certificate_gn.md"),
    ("my passport is about to expire", "passport_renewal.md"),
    ("need to renew my driving license", "driving_license_renewal.md"),
]


def main():
    embedding_fn = get_embedding_function()
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = client.get_collection(COLLECTION_NAME, embedding_function=embedding_fn)

    failures = 0
    for query, expected_source in QUERIES:
        results = collection.query(query_texts=[query], n_results=1)
        top_source = results["metadatas"][0][0]["source"]
        status = "OK" if top_source == expected_source else "FAIL"
        if status == "FAIL":
            failures += 1
        print(f"[{status}] query={query!r} expected={expected_source} got={top_source}")

    if failures:
        raise SystemExit(f"{failures} of {len(QUERIES)} smoke queries failed")
    print("All retrieval smoke queries passed.")


if __name__ == "__main__":
    main()
