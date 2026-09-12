"""STT and TTS helpers for the voice layer.

STT: faster-whisper (large-v3), using the "translate" task so that both
Sinhala and English speech are transcribed directly into English text -
this feeds straight into the existing English-only pipeline without a
separate translation step.

TTS: defaults to gTTS (Google Translate's TTS - free, no API key or billing
account needed, supports Sinhala) since Google Cloud TTS requires a billing
account even for free-tier usage. Set TTS_PROVIDER=google_cloud in .env
(with GOOGLE_APPLICATION_CREDENTIALS set) to use Google Cloud TTS's si-LK
voice instead, per the original tech stack.
"""
import io
import os

from faster_whisper import WhisperModel

WHISPER_MODEL_SIZE = "large-v3"
GTTS_LANGUAGE_CODE = "si"
GOOGLE_CLOUD_TTS_LANGUAGE_CODE = "si-LK"
GOOGLE_CLOUD_TTS_VOICE_NAME = "si-LK-Standard-A"

_whisper_model: WhisperModel | None = None


def _get_whisper_model() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    return _whisper_model


def transcribe_to_english(audio_path: str, language: str | None = None) -> str:
    """Transcribe an audio file to English text, translating from Sinhala
    (or any spoken language Whisper detects) if needed.

    Whisper's automatic language detection only looks at the first ~30s and
    can misidentify short/noisy Sinhala clips as a different language. Pass
    language="si" to force correct detection when you know the speaker is
    speaking Sinhala."""
    model = _get_whisper_model()
    segments, info = model.transcribe(audio_path, task="translate", language=language)
    text = " ".join(segment.text.strip() for segment in segments).strip()
    print(
        f"[debug] detected language={info.language} "
        f"(probability={info.language_probability:.2f})"
    )
    return text


def synthesize_sinhala_speech(text: str) -> bytes:
    """Synthesize Sinhala text to speech audio (MP3 bytes)."""
    provider = os.getenv("TTS_PROVIDER", "gtts").lower()

    if provider == "google_cloud":
        return _synthesize_with_google_cloud(text)
    return _synthesize_with_gtts(text)


def _synthesize_with_gtts(text: str) -> bytes:
    from gtts import gTTS

    buffer = io.BytesIO()
    gTTS(text=text, lang=GTTS_LANGUAGE_CODE).write_to_fp(buffer)
    return buffer.getvalue()


def _synthesize_with_google_cloud(text: str) -> bytes:
    from google.cloud import texttospeech

    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS is not set")

    client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code=GOOGLE_CLOUD_TTS_LANGUAGE_CODE, name=GOOGLE_CLOUD_TTS_VOICE_NAME
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
        print("Usage: python voice.py <path-to-audio-file> [language-code]")
        print("  e.g. python voice.py recording.m4a si   (force Sinhala)")
        raise SystemExit(1)

    forced_language = sys.argv[2] if len(sys.argv) > 2 else None
    text = transcribe_to_english(sys.argv[1], language=forced_language)
    print(f"Transcribed (English): {text}")
