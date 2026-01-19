import streamlit as st
import pandas as pd
import torch
import plotly.express as px
from transformers import (
    DistilBertTokenizerFast,
    AutoModelForSequenceClassification,
    pipeline
)
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import torch.nn.functional as F

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Sentiment & Emotion Analysis Dashboard",
    page_icon="💬",
    layout="wide"
)

st.title("💬 Sentiment & Emotion Analysis Dashboard")

# =====================================================
# LOAD MODELS (CACHED)
# =====================================================
@st.cache_resource
def load_sentiment_model():
    tokenizer = DistilBertTokenizerFast.from_pretrained("sentiment_model")
    model = AutoModelForSequenceClassification.from_pretrained("sentiment_model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=False
    )

tokenizer, sentiment_model, device = load_sentiment_model()
emotion_model = load_emotion_model()

id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================
page = st.sidebar.radio(
    "Navigation",
    ["Live Text Analysis", "Dataset Visualization"]
)

# =====================================================
# LIVE TEXT ANALYSIS PAGE
# =====================================================
if page == "Live Text Analysis":

    st.subheader("📝 Live Sentiment & Emotion Analysis")

    user_text = st.text_area(
        "Enter text (review / tweet / comment):",
        height=120,
        placeholder="Example: I am extremely disappointed with this airline service."
    )

    if st.button("Analyze") and user_text.strip():

        inputs = tokenizer(
            user_text,
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

        sentiment = id2label[pred_idx]
        sentiment_conf = probs[0][pred_idx].item()

        emo_out = emotion_model(user_text)[0]
        emotion = emo_out["label"]
        emotion_conf = emo_out["score"]

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Sentiment",
                sentiment,
                f"{sentiment_conf*100:.2f}%"
            )

        with col2:
            st.metric(
                "Emotion",
                emotion.capitalize(),
                f"{emotion_conf*100:.2f}%"
            )

        st.divider()

        df_sent = pd.DataFrame({
            "Category": [sentiment],
            "Confidence": [sentiment_conf]
        })

        df_emo = pd.DataFrame({
            "Emotion": [emotion],
            "Confidence": [emotion_conf]
        })

        col3, col4 = st.columns(2)

        with col3:
            fig1 = px.bar(
                df_sent,
                x="Category",
                y="Confidence",
                range_y=[0,1],
                title="Sentiment Confidence"
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col4:
            fig2 = px.bar(
                df_emo,
                x="Emotion",
                y="Confidence",
                range_y=[0,1],
                title="Emotion Confidence"
            )
            st.plotly_chart(fig2, use_container_width=True)

# =====================================================
# DATASET VISUALIZATION PAGE
# =====================================================
else:

    st.subheader("📊 Dataset Visualization")

    uploaded_file = st.file_uploader(
        "Upload CSV file (must contain 'text' & 'airline_sentiment')",
        type=["csv"]
    )

    @st.cache_data
    def load_default_data():
        return pd.read_csv("Tweets.csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("Custom dataset loaded.")
    else:
        df = load_default_data()
        st.info("Using default Tweets.csv dataset")

    # ----------------------------------
    # SENTIMENT DISTRIBUTION
    # ----------------------------------
    st.subheader("Sentiment Distribution")

    sent_counts = df["airline_sentiment"].value_counts().reset_index()
    sent_counts.columns = ["Sentiment", "Count"]

    fig_sent = px.bar(
        sent_counts,
        x="Sentiment",
        y="Count",
        title="Sentiment Distribution"
    )

    st.plotly_chart(fig_sent, use_container_width=True)

    # ----------------------------------
    # EMOTION DISTRIBUTION (SAMPLED)
    # ----------------------------------
    st.subheader("Emotion Distribution (Sampled)")

    sample_df = df.sample(min(300, len(df)), random_state=42)

    sample_df["emotion"] = sample_df["text"].apply(
        lambda x: emotion_model(x)[0]["label"]
    )

    emo_counts = sample_df["emotion"].value_counts().reset_index()
    emo_counts.columns = ["Emotion", "Count"]

    fig_emo = px.bar(
        emo_counts,
        x="Emotion",
        y="Count",
        title="Emotion Distribution"
    )

    st.plotly_chart(fig_emo, use_container_width=True)

    # ----------------------------------
    # WORD CLOUDS
    # ----------------------------------
    st.subheader("☁️ Word Clouds")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Positive Reviews")
        pos_text = " ".join(df[df["airline_sentiment"] == "positive"]["text"])
        wc_pos = WordCloud(
            width=600,
            height=400,
            background_color="white"
        ).generate(pos_text)

        plt.figure(figsize=(6,4))
        plt.imshow(wc_pos)
        plt.axis("off")
        st.pyplot(plt)

    with col2:
        st.write("Negative Reviews")
        neg_text = " ".join(df[df["airline_sentiment"] == "negative"]["text"])
        wc_neg = WordCloud(
            width=600,
            height=400,
            background_color="white"
        ).generate(neg_text)

        plt.figure(figsize=(6,4))
        plt.imshow(wc_neg)
        plt.axis("off")
        st.pyplot(plt)

    st.subheader("☁️ Word Cloud by Emotion")

    selected_emotion = st.selectbox(
        "Select Emotion",
        sorted(sample_df["emotion"].unique())
    )

    emotion_text = " ".join(
        sample_df[sample_df["emotion"] == selected_emotion]["text"]
    )

    if emotion_text.strip():
        wc_emotion = WordCloud(
            width=800,
            height=400,
            background_color="white"
        ).generate(emotion_text)

        plt.figure(figsize=(8,4))
        plt.imshow(wc_emotion)
        plt.axis("off")
        st.pyplot(plt)
