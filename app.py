"""Streamlit frontend: mic or text input, chat history, source citations,
and audio playback of the Sinhala answer.

Usage:
    streamlit run app.py
"""
import hashlib
import os
import tempfile

import streamlit as st

from pipeline import answer_query
from voice import synthesize_sinhala_speech, transcribe_to_english

st.set_page_config(page_title="Sarathi AI", page_icon="🇱🇰")
st.title("Sarathi AI")
st.caption("Ask about NIC, birth/marriage certificates, passport, or driving license procedures.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_audio_hash" not in st.session_state:
    st.session_state.last_audio_hash = None


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("english"):
            with st.expander("English"):
                st.write(message["english"])
        if message.get("sources"):
            st.caption("Sources: " + ", ".join(message["sources"]))
        if message.get("audio"):
            st.audio(message["audio"], format="audio/mp3")


for message in st.session_state.messages:
    render_message(message)


def handle_query(query: str) -> None:
    user_message = {"role": "user", "content": query}
    st.session_state.messages.append(user_message)
    render_message(user_message)

    with st.spinner("Looking that up..."):
        try:
            result = answer_query(query)
        except Exception as e:
            error_message = {"role": "assistant", "content": f"Something went wrong: {e}"}
            st.session_state.messages.append(error_message)
            render_message(error_message)
            return

        if result.needs_clarification:
            assistant_message = {
                "role": "assistant",
                "content": result.clarifying_question,
            }
        else:
            try:
                audio_bytes = synthesize_sinhala_speech(result.sinhala_answer)
            except Exception:
                audio_bytes = None
            assistant_message = {
                "role": "assistant",
                "content": result.sinhala_answer,
                "english": result.english_answer,
                "sources": result.sources,
                "audio": audio_bytes,
            }

    st.session_state.messages.append(assistant_message)
    render_message(assistant_message)


audio_value = st.audio_input("Ask by voice")
text_value = st.chat_input("Or type your question")

if audio_value is not None:
    audio_bytes_in = audio_value.getvalue()
    audio_hash = hashlib.md5(audio_bytes_in).hexdigest()
    if audio_hash != st.session_state.last_audio_hash:
        st.session_state.last_audio_hash = audio_hash
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes_in)
            tmp_path = tmp.name
        try:
            with st.spinner("Transcribing..."):
                transcribed_query = transcribe_to_english(tmp_path)
        finally:
            os.remove(tmp_path)

        if transcribed_query:
            handle_query(transcribed_query)
        else:
            st.warning("Could not make out any speech in that recording - please try again.")

elif text_value:
    handle_query(text_value)
