"""FastAPI app exposing the router -> specialist -> composer pipeline.

Usage:
    uvicorn main:app --reload
"""
import base64
import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from pipeline import answer_query
from voice import synthesize_sinhala_speech, transcribe_to_english

app = FastAPI(title="Sarathi AI")


class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    domain: str
    needs_clarification: bool
    clarifying_question: str | None = None
    english_answer: str | None = None
    sinhala_answer: str | None = None
    sources: list[str] | None = None


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query must not be empty")

    result = answer_query(query)

    return AskResponse(
        domain=result.domain,
        needs_clarification=result.needs_clarification,
        clarifying_question=result.clarifying_question,
        english_answer=result.english_answer,
        sinhala_answer=result.sinhala_answer,
        sources=result.sources,
    )


class AskVoiceResponse(AskResponse):
    transcribed_query: str
    sinhala_audio_base64: str | None = None


@app.post("/ask-voice", response_model=AskVoiceResponse)
async def ask_voice(file: UploadFile = File(...)) -> AskVoiceResponse:
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        transcribed_query = transcribe_to_english(tmp_path)
    finally:
        os.remove(tmp_path)

    if not transcribed_query:
        raise HTTPException(status_code=400, detail="could not transcribe any speech from audio")

    result = answer_query(transcribed_query)

    sinhala_audio_base64 = None
    if not result.needs_clarification and result.sinhala_answer:
        audio_bytes = synthesize_sinhala_speech(result.sinhala_answer)
        sinhala_audio_base64 = base64.b64encode(audio_bytes).decode("ascii")

    return AskVoiceResponse(
        domain=result.domain,
        needs_clarification=result.needs_clarification,
        clarifying_question=result.clarifying_question,
        english_answer=result.english_answer,
        sinhala_answer=result.sinhala_answer,
        sources=result.sources,
        transcribed_query=transcribed_query,
        sinhala_audio_base64=sinhala_audio_base64,
    )
