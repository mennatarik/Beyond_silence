import streamlit as st
import tempfile
import torch
import pandas as pd
import whisper
from transformers import pipeline

st.set_page_config(page_title="Beyond Silence App", page_icon="🔊")
st.title("🔊 Beyond Silence App")
st.info("Transcribes speech to text and detects emotion.")

whisper_model = whisper.load_model("base")
classifier = pipeline(
    "text-classification",
    model="./my_trained_model",
    return_all_scores=False
)

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.audio(uploaded_file, format='audio/wav')

    st.write("Processing transcription...")
    result = whisper_model.transcribe(tmp_path)
    transcription = result["text"]

    st.subheader("📝 Transcription:")
    st.write(transcription)

    st.write("Detecting emotion...")
    emotion_result = classifier(transcription)[0]
    emotion_label = emotion_result['label']
    emotion_score = emotion_result['score']

    st.subheader("😊 Detected Emotion:")
    st.write(f"{emotion_label} (confidence: {emotion_score:.2f})")

    st.success("Done ✅")
