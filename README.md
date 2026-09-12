# Sarathi AI

Voice-first assistant that helps Sri Lankan citizens navigate government procedures
(NIC renewal, birth/marriage/death certificates via Grama Niladhari, passport renewal,
driving license renewal) in Sinhala and English.

## Status

All 10 build phases from `master-prompt.md` are implemented: procedure data
and retrieval, the router/specialist/composer agents, `pipeline.py`, the
`/ask` and `/ask-voice` FastAPI endpoints, and the Streamlit frontend
(`app.py`) with a disclaimer, scripted example queries, and a text-input
fallback for when voice isn't available. See Known limitations below for
what's substituted from the original spec and what's still worth a final
manual check before a live demo.

## Setup

1. `python -m venv venv && source venv/bin/activate` (or `venv\Scripts\activate` on Windows)
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and fill in API keys.
4. `python ingest.py` to build the local vector index.
5. `python pipeline.py "I lost my NIC"` to test the full pipeline from the CLI,
   `uvicorn main:app --reload` and `POST /ask` with `{"query": "..."}`, or
   `streamlit run app.py` for the full chat UI (mic or text input).

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
- **`/ask-voice`'s full chain (STT -> pipeline -> TTS) hasn't been tested as
  one request yet.** STT (faster-whisper `large-v3`) and TTS (gTTS) have each
  been verified working individually with real audio; the combined endpoint
  should still be exercised with a real audio file before a live demo.
- The Streamlit app's mic input and button clicks haven't been exercised in
  a real browser session (only verified that the app boots without errors) -
  do a quick pass through the UI, including the example-query buttons and a
  live mic recording, before demoing.
