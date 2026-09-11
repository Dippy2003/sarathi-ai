"""Chains router -> specialist -> composer into a single call.

Usage (CLI):
    python pipeline.py "I lost my NIC"
"""
from dataclasses import dataclass
from typing import Optional

from agents.composer import compose_sinhala
from agents.router import route_query
from agents.specialist import draft_answer


@dataclass
class PipelineResult:
    domain: str
    needs_clarification: bool
    clarifying_question: Optional[str] = None
    english_answer: Optional[str] = None
    sinhala_answer: Optional[str] = None
    sources: Optional[list[str]] = None


def answer_query(query: str) -> PipelineResult:
    routed = route_query(query)

    if routed.needs_clarification:
        return PipelineResult(
            domain=routed.domain,
            needs_clarification=True,
            clarifying_question=routed.clarifying_question,
        )

    specialist_result = draft_answer(routed.domain, query)
    composer_result = compose_sinhala(specialist_result.answer, specialist_result.sources)

    return PipelineResult(
        domain=routed.domain,
        needs_clarification=False,
        english_answer=specialist_result.answer,
        sinhala_answer=composer_result.sinhala_text,
        sources=composer_result.sources,
    )


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    query = " ".join(sys.argv[1:])
    if not query:
        print("Usage: python pipeline.py \"<your question>\"")
        raise SystemExit(1)

    result = answer_query(query)

    print(f"Domain: {result.domain}")
    if result.needs_clarification:
        print(f"Clarifying question: {result.clarifying_question}")
    else:
        print(f"\nEnglish answer:\n{result.english_answer}")
        print(f"\nSinhala answer:\n{result.sinhala_answer}")
        print(f"\nSources: {result.sources}")
