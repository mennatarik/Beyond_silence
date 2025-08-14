import streamlit as st
import tempfile
import torch

import pandas as pd
st.set_page_config(page_title="Beyond Silence App", page_icon="🔊")
st.title("🔊 Beyond Silence App")
st.info("Transcribes speech to text and detects emotion.")

#Printing the output in the streamlit app
# Upload audio file
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Play audio
    st.audio(uploaded_file, format='audio/wav')

    # Transcription with Whisper
    import whisper
    whisper_model = whisper.load_model("base")  # or "small", "medium", etc.

    st.write("Processing transcription...")
    result = whisper_model.transcribe(tmp_path)
    transcription = result["text"]

    st.subheader("📝 Transcription:")
    st.write(transcription)

    # Emotion detection using your fine-tuned model
    st.write("Detecting emotion...")
    emotion_result = classifier(transcription)[0]  # classifier returns a list of dicts
    emotion_label = emotion_result['label']
    emotion_score = emotion_result['score']

    st.subheader("😊 Detected Emotion:")
    st.write(f"{emotion_label} (confidence: {emotion_score:.2f})")

    st.success("Done ✅")
