"""Composer agent: takes the English draft from the specialist and produces
the final plain-spoken Sinhala version (simple, everyday register - not
bureaucratic Sinhala). This is the only step that localizes to Sinhala;
retrieval and reasoning stay in English (translate-late pattern).

COMPOSER_MODEL defaults to google/gemini-2.5-flash rather than the rest of
the pipeline's openai/gpt-oss-120b: in testing, gpt-oss-120b produced
degenerate/repetitive Sinhala output (a single word repeated hundreds of
times), while Gemini produced clean, natural Sinhala. This is the only call
in the pipeline that uses a different model - routing and retrieval/drafting
stay on gpt-oss-120b as specified. Set COMPOSER_MODEL in .env to override."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_COMPOSER_MODEL = "google/gemini-2.5-flash"

SYSTEM_PROMPT = """You are localizing a government-procedure answer into Sinhala \
for an ordinary citizen. Translate and rewrite the English answer below into \
simple, everyday spoken Sinhala - the way you'd explain it to a family member, \
not formal/bureaucratic Sinhala and not word-for-word literal translation.

Rules:
- Keep every fact, step, document, fee, and location from the English answer. \
Do not add or drop information.
- Keep it natural and easy to listen to when read aloud (this will be sent to \
text-to-speech).
- Do not translate proper nouns/institution names that are commonly used in \
English in Sri Lanka (e.g. "Grama Niladhari", "NIC") unless a Sinhala form is \
clearly more natural.
- Output ONLY the Sinhala text, nothing else - no preamble, no English.
"""


@dataclass
class ComposerResult:
    sinhala_text: str
    sources: list[str]


def _client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)


def compose_sinhala(english_answer: str, sources: list[str]) -> ComposerResult:
    model = os.getenv("COMPOSER_MODEL", DEFAULT_COMPOSER_MODEL)

    response = _client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": english_answer},
        ],
        temperature=0.3,
        max_tokens=1000,
    )
    sinhala_text = response.choices[0].message.content.strip()

    return ComposerResult(sinhala_text=sinhala_text, sources=sources)


if __name__ == "__main__":
    import sys

    english_answer = " ".join(sys.argv[1:]) or (
        "1. Report the loss to the police and get a police report. "
        "2. Fill in the NIC replacement form. "
        "3. Get it certified by your Grama Niladhari. "
        "4. Submit it with your police report at the District Secretariat."
    )
    result = compose_sinhala(english_answer, sources=["nic_replacement_lost_damaged.md"])
    print(result.sinhala_text)
    print("\nSources:", result.sources)
