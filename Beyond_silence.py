import streamlit as st
import tempfile
import torch
import whisper
from transformers import pipeline
import pandas as pd

# تحميل الموديلات مرة واحدة
whisper_model = whisper.load_model("base")  # أو small/medium حسب ما تحبي
classifier = pipeline(
    "text-classification", 
    model="path/to/your/fine_tuned_model",  # ← عدّلي المسار هنا
    return_all_scores=False
)

st.set_page_config(page_title="Beyond Silence App", page_icon="🔊")
st.title("🔊 Beyond Silence App")
st.info("Transcribes speech to text and detects emotion.")

# رفع ملف الصوت
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    # حفظ الملف مؤقتًا
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # تشغيل الصوت
    st.audio(uploaded_file, format='audio/wav')

    # تحويل الكلام لنص
    st.write("Processing transcription...")
    result = whisper_model.transcribe(tmp_path)
    transcription = result["text"]

    st.subheader("📝 Transcription:")
    st.write(transcription)

    # تحليل المشاعر
    st.write("Detecting emotion...")
    emotion_result = classifier(transcription)[0]
    emotion_label = emotion_result['label']
    emotion_score = emotion_result['score']

    st.subheader("😊 Detected Emotion:")
    st.write(f"{emotion_label} (confidence: {emotion_score:.2f})")

    st.success("Done ✅")
