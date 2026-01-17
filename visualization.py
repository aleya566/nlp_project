import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.title("📊 Dataset Visualization Dashboard")

@st.cache_data
def load_data():
    return pd.read_csv("data/Tweets.csv")

df = load_data()

# =========================
# SENTIMENT DISTRIBUTION
# =========================
sent_counts = df["airline_sentiment"].value_counts().reset_index()
sent_counts.columns = ["Sentiment", "Count"]

fig_sent = px.bar(
    sent_counts,
    x="Sentiment",
    y="Count",
    title="Sentiment Distribution (Dataset)"
)

st.plotly_chart(fig_sent, use_container_width=True)

# =========================
# EMOTION DISTRIBUTION
# =========================
@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=False
    )

emotion_model = load_emotion_model()

sample_df = df.sample(300, random_state=42)
sample_df["emotion"] = sample_df["text"].apply(
    lambda x: emotion_model(x)[0]["label"]
)

emo_counts = sample_df["emotion"].value_counts().reset_index()
emo_counts.columns = ["Emotion", "Count"]

fig_emo = px.bar(
    emo_counts,
    x="Emotion",
    y="Count",
    title="Emotion Distribution (Sampled Data)"
)

st.plotly_chart(fig_emo, use_container_width=True)

# =========================
# WORD CLOUDS
# =========================
st.subheader("☁️ Word Cloud")

col1, col2 = st.columns(2)

with col1:
    st.write("Positive Reviews")
    pos_text = " ".join(df[df["airline_sentiment"] == "positive"]["text"])
    wc_pos = WordCloud(
        width=600, height=400, background_color="white"
    ).generate(pos_text)

    plt.figure(figsize=(6,4))
    plt.imshow(wc_pos, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

with col2:
    st.write("Negative Reviews")
    neg_text = " ".join(df[df["airline_sentiment"] == "negative"]["text"])
    wc_neg = WordCloud(
        width=600, height=400, background_color="white"
    ).generate(neg_text)

    plt.figure(figsize=(6,4))
    plt.imshow(wc_neg, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)
