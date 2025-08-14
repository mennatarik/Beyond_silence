# !pip install --upgrade pip
# !pip install git+https://github.com/openai/whisper.git
# !pip install setuptools-rust

# 3. Write your app

#%%writefile Beyond_Silence.py
import streamlit as st
import tempfile
import torch

st.set_page_config(page_title="Beyond Silence App", page_icon="🔊")
st.title("🔊 Beyond Silence App")
st.info("Transcribes speech to text and detects emotion.")

import pandas as pd
from datasets import Dataset, DatasetDict

train_df = pd.read_csv("train.txt", sep=";", header=None, names=["text", "label"])
val_df = pd.read_csv("val.txt", sep=";", header=None, names=["text", "label"])
test_df = pd.read_csv("test.txt", sep=";", header=None, names=["text", "label"])

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

dataset = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset
})

#Using DistilBert model
import transformers
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

label_list = ['joy', 'sadness', 'anger', 'fear', 'love', 'surprise']
num_labels = len(label_list)

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)

#returning the labels from encoded to their real names "sadness/ joy/...."
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True, max_length=128)

tokenized_datasets = dataset.map(preprocess_function, batched=True)

def encode_labels(example):
    example['label'] = label_list.index(example['label'])
    return example

tokenized_datasets = tokenized_datasets.map(encode_labels)

#fine tuning
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    report_to="none",
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    tokenizer=tokenizer,
)

trainer.train()
eval_results = trainer.evaluate()

#saving the fine-tuned model
trainer.save_model("./my_trained_model")
tokenizer.save_pretrained("./my_trained_model")

trainer.save_model("./my_trained_model")
tokenizer.save_pretrained("./my_trained_model")

#pipelining
from transformers import pipeline

classifier = pipeline("text-classification", model="./my_trained_model", tokenizer="./my_trained_model")

# #whisper installing
# !pip install -q --upgrade pip
# !pip install -q git+https://github.com/openai/whisper.git
# !pip install -q setuptools-rust

import whisper
# Load Whisper model (GPU if available)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("base")


from transformers import pipeline
classifier = pipeline("text-classification", model="./my_trained_model", tokenizer="./my_trained_model")

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

