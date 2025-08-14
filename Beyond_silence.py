import streamlit as st
import tempfile
import torch
import whisper
from transformers import pipeline
import pandas as pd

# ===== Load Models =====
whisper_model = whisper.load_model("base")
classifier = pipeline("text-classification", model="./my_trained_model", return_all_scores=False)

label_list = ['joy', 'sadness', 'anger', 'fear', 'love', 'surprise']

# ===== Page Config =====
st.set_page_config(page_title="Beyond Silence App", page_icon="🎙️", layout="centered")

# ===== Header =====
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>🎙️ Beyond Silence</h1>
    <p style='text-align: center; font-size: 18px; color: #555;'>
    Transform speech into text & detect emotions instantly
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ===== File Upload =====
uploaded_file = st.file_uploader("📤 Upload an audio file", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Audio player
    st.audio(uploaded_file, format='audio/wav')

    # ===== Transcription =====
    with st.spinner("📝 Transcribing audio..."):
        result = whisper_model.transcribe(tmp_path)
        transcription = result["text"]

    st.markdown(
        f"""
        <div style='background-color: #f9f9f9; padding: 15px; border-radius: 10px; border-left: 5px solid #4CAF50;'>
        <h3>📝 Transcription</h3>
        <p style='font-size: 16px; color: #333;'>{transcription}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ===== Emotion Detection =====
    with st.spinner("🔍 Detecting emotion..."):
        emotion_result = classifier(transcription)[0]
        emotion_id = int(emotion_result['label'].split("_")[-1])
        emotion_label = label_list[emotion_id]
        emotion_score = emotion_result['score']

    # Styled output
    color_map = {
        'joy': '#FFD700',
        'sadness': '#1E90FF',
        'anger': '#FF4500',
        'fear': '#8B0000',
        'love': '#FF69B4',
        'surprise': '#9932CC'
    }
    color = color_map.get(emotion_label, '#4CAF50')

    st.markdown(
        f"""
        <div style='background-color: {color}; padding: 15px; border-radius: 10px; text-align: center;'>
        <h2 style='color: white;'>😊 Detected Emotion: {emotion_label}</h2>
        <p style='color: white; font-size: 18px;'>Confidence: {emotion_score:.2f}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.success("✅ Analysis Complete")
