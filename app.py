import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Sentiment & Emotion Analysis Dashboard",
    page_icon="💬",
    layout="wide"
)

st.title("💬 Sentiment & Emotion Analysis Dashboard")
st.write(
    "This dashboard analyzes **sentiment polarity** (Positive, Neutral, Negative) "
    "and **advanced emotions** (anger, joy, sadness, fear, etc.) using "
    "Transformer-based NLP models."
)

# ==============================
# LOAD MODELS (CACHED)
# ==============================
@st.cache_resource
def load_models():
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment"
    )

    emotion_pipeline = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=False
    )

    return sentiment_pipeline, emotion_pipeline


sentiment_pipeline, emotion_pipeline = load_models()

# Label mapping for sentiment model
label_map = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

# ==============================
# TEXT INPUT
# ==============================
st.subheader("📝 Live Text Analysis")

user_text = st.text_area(
    "Enter customer review or social media text:",
    height=120,
    placeholder="Example: I am extremely angry and disappointed with the service."
)

analyze_button = st.button("Analyze")

# ==============================
# ANALYSIS FUNCTION
# ==============================
def analyze_text(text):
    # Sentiment
    sentiment_result = sentiment_pipeline(text)[0]
    sentiment = label_map[sentiment_result["label"]]
    sentiment_score = sentiment_result["score"]

    # Emotion
    emotion_result = emotion_pipeline(text)[0]
    emotion = emotion_result["label"]
    emotion_score = emotion_result["score"]

    return sentiment, sentiment_score, emotion, emotion_score


# ==============================
# RESULT DISPLAY
# ==============================
if analyze_button and user_text.strip() != "":
    sentiment, sentiment_score, emotion, emotion_score = analyze_text(user_text)

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Sentiment",
            value=sentiment,
            delta=f"{sentiment_score:.2f}"
        )

    with col2:
        st.metric(
            label="Emotion",
            value=emotion.capitalize(),
            delta=f"{emotion_score:.2f}"
        )

    st.divider()

    # ==============================
    # VISUALIZATION
    # ==============================
    st.subheader("📊 Prediction Confidence Visualization")

    # Sentiment chart
    sentiment_df = pd.DataFrame({
        "Category": [sentiment],
        "Confidence": [sentiment_score]
    })

    fig_sentiment = px.bar(
        sentiment_df,
        x="Category",
        y="Confidence",
        title="Sentiment Confidence",
        range_y=[0, 1]
    )

    # Emotion chart
    emotion_df = pd.DataFrame({
        "Emotion": [emotion],
        "Confidence": [emotion_score]
    })

    fig_emotion = px.bar(
        emotion_df,
        x="Emotion",
        y="Confidence",
        title="Emotion Confidence",
        range_y=[0, 1]
    )

    col3, col4 = st.columns(2)

    with col3:
        st.plotly_chart(fig_sentiment, use_container_width=True)

    with col4:
        st.plotly_chart(fig_emotion, use_container_width=True)

# ==============================
# FOOTER
# ==============================
st.caption(
    "Models: Twitter-RoBERTa (Sentiment) | DistilRoBERTa (Emotion) • "
    "Built with Streamlit & HuggingFace Transformers"
)
