"""STT and TTS helpers for the voice layer.

STT: faster-whisper (large-v3), using the "translate" task so that both
Sinhala and English speech are transcribed directly into English text -
this feeds straight into the existing English-only pipeline without a
separate translation step.

TTS: Google Cloud Text-to-Speech, si-LK voice, for the final Sinhala answer.
"""
import os

from faster_whisper import WhisperModel
from google.cloud import texttospeech

WHISPER_MODEL_SIZE = "large-v3"
TTS_LANGUAGE_CODE = "si-LK"
TTS_VOICE_NAME = "si-LK-Standard-A"

_whisper_model: WhisperModel | None = None


def _get_whisper_model() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    return _whisper_model


def transcribe_to_english(audio_path: str) -> str:
    """Transcribe an audio file to English text, translating from Sinhala
    (or any spoken language Whisper detects) if needed."""
    model = _get_whisper_model()
    segments, _info = model.transcribe(audio_path, task="translate")
    return " ".join(segment.text.strip() for segment in segments).strip()


def synthesize_sinhala_speech(text: str) -> bytes:
    """Synthesize Sinhala text to speech audio (MP3 bytes) via Google Cloud TTS."""
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS is not set")

    client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code=TTS_LANGUAGE_CODE, name=TTS_VOICE_NAME
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    return response.audio_content


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python voice.py <path-to-audio-file>")
        raise SystemExit(1)

    text = transcribe_to_english(sys.argv[1])
    print(f"Transcribed (English): {text}")
