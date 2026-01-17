import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
import plotly.express as px

from transformers import (
    DistilBertTokenizerFast,
    AutoModelForSequenceClassification,
    pipeline
)

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Sentiment & Emotion Analysis Dashboard",
    layout="wide"
)

st.title("📊 Sentiment & Emotion Analysis Dashboard")
st.write(
    "This dashboard analyzes sentiment polarity "
    "(positive, neutral, negative) and advanced emotions "
    "(anger, joy, sadness, fear, etc.) using Transformer models."
)

# --------------------------------------------------
# LOAD SENTIMENT MODEL
# --------------------------------------------------
@st.cache_resource
def load_sentiment_model():
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=3,
        id2label={0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"},
        label2id={"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return tokenizer, model, device


tokenizer, sentiment_model, device = load_sentiment_model()

# --------------------------------------------------
# LOAD EMOTION MODEL
# --------------------------------------------------
@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True
    )


emotion_classifier = load_emotion_model()

# --------------------------------------------------
# FUNCTIONS
# --------------------------------------------------
def predict_sentiment(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = sentiment_model(**inputs)

    probs = F.softmax(outputs.logits, dim=-1)
    pred_idx = torch.argmax(probs).item()

    return sentiment_model.config.id2label[pred_idx], probs[0][pred_idx].item()


def predict_emotion(text):
    outputs = emotion_classifier(text)[0]
    top_emotion = max(outputs, key=lambda x: x["score"])
    return top_emotion["label"], top_emotion["score"], outputs


def analyze_text(text):
    sentiment, sent_conf = predict_sentiment(text)
    emotion, emo_conf, emo_scores = predict_emotion(text)

    return sentiment, sent_conf, emotion, emo_conf, emo_scores


# --------------------------------------------------
# LIVE TEXT INPUT
# --------------------------------------------------
st.subheader("🔴 Live Text Analysis")

user_text = st.text_area(
    "Enter a tweet, review, or sentence:",
    height=120,
    placeholder="e.g. I am extremely angry with the airline service"
)

if st.button("Analyze"):
    if user_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        sentiment, sent_conf, emotion, emo_conf, emo_scores = analyze_text(user_text)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Sentiment", sentiment, f"{sent_conf:.2%}")

        with col2:
            st.metric("Emotion", emotion, f"{emo_conf:.2%}")

        # Emotion probability bar chart
        emo_df = pd.DataFrame(emo_scores)
        fig_emo = px.bar(
            emo_df,
            x="label",
            y="score",
            title="Emotion Confidence Scores",
            labels={"label": "Emotion", "score": "Confidence"}
        )
        st.plotly_chart(fig_emo, use_container_width=True)

# --------------------------------------------------
# DATASET VISUALIZATION (STATIC)
# --------------------------------------------------
st.subheader("📈 Dataset-Level Visualization")

uploaded_file = st.file_uploader(
    "Upload Airline Tweets CSV (optional)",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = df[['text', 'airline_sentiment']]

    # Sentiment distribution
    sent_counts = df['airline_sentiment'].value_counts().reset_index()
    sent_counts.columns = ['sentiment', 'count']

    fig_sent = px.bar(
        sent_counts,
        x="sentiment",
        y="count",
        title="Sentiment Distribution",
        color="sentiment"
    )
    st.plotly_chart(fig_sent, use_container_width=True)

    # Emotion distribution (sample)
    st.info("Computing emotion distribution on a sample of data...")
    sample_texts = df['text'].sample(min(200, len(df)))

    emotions = [predict_emotion(t)[0] for t in sample_texts]
    emo_counts = pd.Series(emotions).value_counts().reset_index()
    emo_counts.columns = ['emotion', 'count']

    fig_emo_dist = px.bar(
        emo_counts,
        x="emotion",
        y="count",
        title="Emotion Distribution (Sample Data)",
        color="emotion"
    )
    st.plotly_chart(fig_emo_dist, use_container_width=True)
