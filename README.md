# Sarathi AI

Voice-first assistant that helps Sri Lankan citizens navigate government procedures
(NIC renewal, birth/marriage/death certificates via Grama Niladhari, passport renewal,
driving license renewal) in Sinhala and English.

## Status

Early scaffold. See `master-prompt.md` for the full build spec and roadmap.

## Setup

1. `python -m venv venv && source venv/bin/activate` (or `venv\Scripts\activate` on Windows)
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and fill in API keys.

## Required API keys

- OpenRouter API key (LLM: `openai/gpt-oss-120b`)
- OpenAI API key (embeddings: `text-embedding-3-small`)
- Google Cloud service account with Text-to-Speech access (`si-LK` voice)

## Known limitations

Nothing implemented yet beyond repo scaffolding.
