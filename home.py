import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline

st.title("📝 Live Sentiment & Emotion Analysis")

@st.cache_resource
def load_models():
    sentiment_model = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment"
    )

    emotion_model = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=False
    )

    return sentiment_model, emotion_model

sentiment_model, emotion_model = load_models()

label_map = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

user_text = st.text_area(
    "Enter review or social media text:",
    height=120,
    placeholder="Example: I am very angry and disappointed with the service."
)

if st.button("Analyze") and user_text.strip():

    sent_out = sentiment_model(user_text)[0]
    sentiment = label_map[sent_out["label"]]
    sent_score = sent_out["score"]

    emo_out = emotion_model(user_text)[0]
    emotion = emo_out["label"]
    emo_score = emo_out["score"]

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Sentiment", sentiment, f"{sent_score:.2f}")

    with col2:
        st.metric("Emotion", emotion.capitalize(), f"{emo_score:.2f}")

    st.divider()

    df_sent = pd.DataFrame({
        "Category": [sentiment],
        "Confidence": [sent_score]
    })

    df_emo = pd.DataFrame({
        "Emotion": [emotion],
        "Confidence": [emo_score]
    })

    fig1 = px.bar(df_sent, x="Category", y="Confidence", range_y=[0,1])
    fig2 = px.bar(df_emo, x="Emotion", y="Confidence", range_y=[0,1])

    col3, col4 = st.columns(2)

    with col3:
        st.plotly_chart(fig1, use_container_width=True)

    with col4:
        st.plotly_chart(fig2, use_container_width=True)
