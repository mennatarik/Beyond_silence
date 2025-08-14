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

# ===== Dark Theme CSS =====
st.markdown("""
    <style>
        body {
            background-color: #121212;
            color: #E0E0E0;
        }
        .stApp {
            background-color: #121212;
            color: #E0E0E0;
        }
        h1, h2, h3, p {
            color: #E0E0E0;
        }
    </style>
""", unsafe_allow_html=True)

# ===== Header =====
st.markdown(
    """
    <h1 style='text-align: center; color: #BB86FC;'>🎙️ Beyond Silence</h1>
    <p style='text-align: center; font-size: 18px; color: #B0B0B0;'>
    Transform speech into text & detect emotions instantly
    </p>
    <hr style='border-color: #333;'>
    """,
    unsafe_allow_html=True
)

# ===== File Upload =====
uploaded_file = st.file_uploader("📤 Upload an audio file", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.audio(uploaded_file, format='audio/wav')

    # ===== Transcription =====
    with st.spinner("📝 Transcribing audio..."):
        result = whisper_model.transcribe(tmp_path)
        transcription = result["text"]

    st.markdown(
        f"""
        <div style='background-color: #1E1E1E; padding: 15px; border-radius: 10px; border-left: 5px solid #BB86FC;'>
        <h3 style='color: #BB86FC;'>📝 Transcription</h3>
        <p style='font-size: 16px; color: #E0E0E0;'>{transcription}</p>
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

    # Dark Color Map
    color_map = {
        'joy': '#FFD369',
        'sadness': '#4A90E2',
        'anger': '#FF6B6B',
        'fear': '#9B59B6',
        'love': '#FF9FF3',
        'surprise': '#2ECC71'
    }
    color = color_map.get(emotion_label, '#B0BEC5')

    st.markdown(
        f"""
        <div style='background-color: #1E1E1E; padding: 15px; border-radius: 10px; text-align: center; border: 2px solid {color};'>
        <h2 style='color: {color};'>😊 Detected Emotion: {emotion_label}</h2>
        <p style='color: #E0E0E0; font-size: 18px;'>Confidence: {emotion_score:.2f}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.success("✅ Analysis Complete")
