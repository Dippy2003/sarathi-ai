# Sarathi AI

Voice-first assistant that helps Sri Lankan citizens navigate government procedures
(NIC renewal, birth/marriage/death certificates via Grama Niladhari, passport renewal,
driving license renewal) in Sinhala and English.

## Status

Router -> specialist -> composer pipeline is working end-to-end, exposed via
`POST /ask`. Voice layer (`POST /ask-voice`, STT via faster-whisper, TTS via
gTTS) is built but not yet fully verified end-to-end - see Known limitations.
See `master-prompt.md` for the full build spec and roadmap.

## Setup

1. `python -m venv venv && source venv/bin/activate` (or `venv\Scripts\activate` on Windows)
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and fill in API keys.
4. `python ingest.py` to build the local vector index.
5. `python pipeline.py "I lost my NIC"` to test the full pipeline from the CLI,
   or `uvicorn main:app --reload` and `POST /ask` with `{"query": "..."}`.

## Required API keys

- OpenRouter API key (LLM: `openai/gpt-oss-120b`)
- OpenAI API key — only needed if `EMBEDDING_PROVIDER=openai`; by default
  embeddings run locally for free (see below)
- No key needed for TTS by default (`TTS_PROVIDER=gtts`, free). Only needed
  if you switch to `TTS_PROVIDER=google_cloud`: a Google Cloud service
  account with Text-to-Speech access (`si-LK` voice)

## Known limitations

- **Embeddings run locally by default, not via OpenAI.** `EMBEDDING_PROVIDER`
  defaults to `local` (Chroma's built-in `all-MiniLM-L6-v2`, free, no API key)
  instead of the originally planned `text-embedding-3-small`. Set
  `EMBEDDING_PROVIDER=openai` in `.env` (with `OPENAI_API_KEY` set and
  billing enabled) to use the original model instead.
- **`gpt-oss-120b`'s Sinhala output is unreliable.** In testing, the composer
  step occasionally produced degenerate output (a single word repeated
  hundreds of times) when using `gpt-oss-120b` for the Sinhala localization
  call. The composer therefore defaults to `google/gemini-2.5-flash` instead
  (set `COMPOSER_MODEL` in `.env` to override) - this only changes the final
  localization call; routing and retrieval/drafting stay on `gpt-oss-120b`
  as specified.
- **TTS runs via gTTS by default, not Google Cloud TTS.** `TTS_PROVIDER`
  defaults to `gtts` (free, no API key or billing account needed - Google
  Cloud requires billing enabled even for its free tier) instead of the
  originally planned Google Cloud `si-LK` voice. Set
  `TTS_PROVIDER=google_cloud` in `.env` (with `GOOGLE_APPLICATION_CREDENTIALS`
  set) to use the original service instead.
- **Voice layer (`/ask-voice`) is code-complete but not fully tested end to
  end.** TTS (gTTS) has been verified directly. STT (faster-whisper
  `large-v3`) has not been run in this environment (multi-GB model
  download) - the `/ask-voice` endpoint's STT -> pipeline -> TTS chain
  should be tested with a real audio file before relying on it.
- The Streamlit frontend is not built yet.
