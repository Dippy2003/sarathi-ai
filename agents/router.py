"""Router agent: one LLM call that classifies a query into a service domain
and flags whether a clarifying question is needed before an answer can be
given (e.g. "my NIC" alone doesn't say whether it's lost, damaged, or just
expiring)."""
import json
import os
from dataclasses import dataclass
from typing import Literal, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
ROUTER_MODEL = "openai/gpt-oss-120b"

Domain = Literal["nic", "civil_registration", "passport", "driving_license"]

VALID_DOMAINS = {"nic", "civil_registration", "passport", "driving_license"}

SYSTEM_PROMPT = """You are a routing classifier for a Sri Lankan government \
services assistant. Given a citizen's question, classify it into exactly one \
service domain and decide if a clarifying question is needed before it can be \
answered correctly.

Domains:
- nic: National Identity Card renewal, replacement (lost/damaged), or updates
- civil_registration: birth certificates, marriage certificates via Grama Niladhari
- passport: passport renewal, replacement
- driving_license: driving license renewal, replacement

Ask a clarifying question when the domain has branching procedures and the query \
doesn't say which branch applies. For example "my NIC" alone is ambiguous between \
renewal (card in hand) and replacement (lost/damaged) - ask which case it is. \
Do not ask a clarifying question if the query already makes the branch clear \
(e.g. "I lost my NIC" clearly means replacement).

Respond with ONLY a JSON object, no other text, in this exact shape:
{"domain": "<one of: nic, civil_registration, passport, driving_license>", \
"needs_clarification": <true or false>, \
"clarifying_question": "<question text, or null if needs_clarification is false>"}
"""


@dataclass
class RouterResult:
    domain: Domain
    needs_clarification: bool
    clarifying_question: Optional[str]


def _client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)


def route_query(query: str) -> RouterResult:
    """Classify a citizen's query into a domain and decide if clarification
    is needed before it can be answered."""
    response = _client().chat.completions.create(
        model=ROUTER_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=0,
    )
    content = response.choices[0].message.content.strip()
    parsed = json.loads(content)

    domain = parsed.get("domain")
    if domain not in VALID_DOMAINS:
        raise ValueError(f"Router returned an unrecognized domain: {domain!r}")

    return RouterResult(
        domain=domain,
        needs_clarification=bool(parsed.get("needs_clarification", False)),
        clarifying_question=parsed.get("clarifying_question"),
    )


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "my NIC"
    result = route_query(query)
    print(result)
