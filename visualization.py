import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px

st.title("📊 Sentiment & Emotion Visualization Dashboard")

# =========================
# Load Dataset
# =========================
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.warning("Please upload a CSV file to view visualizations.")
    st.stop()

# Expected columns
# text, sentiment, emotion

# =========================
# Sentiment Distribution
# =========================
st.subheader("Sentiment Distribution")

sentiment_counts = df["sentiment"].value_counts().reset_index()
sentiment_counts.columns = ["Sentiment", "Count"]

fig_sent = px.bar(
    sentiment_counts,
    x="Sentiment",
    y="Count",
    color="Sentiment",
    title="Sentiment Polarity Distribution"
)
st.plotly_chart(fig_sent, use_container_width=True)

# =========================
# Emotion Distribution
# =========================
st.subheader("Emotion Distribution")

emotion_counts = df["emotion"].value_counts().reset_index()
emotion_counts.columns = ["Emotion", "Count"]

fig_emo = px.bar(
    emotion_counts,
    x="Emotion",
    y="Count",
    color="Emotion",
    title="Emotion Classification Distribution"
)
st.plotly_chart(fig_emo, use_container_width=True)

# =========================
# Sentiment WordClouds
# =========================
st.subheader("Sentiment WordClouds")

pos_text = " ".join(df[df["sentiment"] == "positive"]["text"].astype(str))
neg_text = " ".join(df[df["sentiment"] == "negative"]["text"].astype(str))

col1, col2 = st.columns(2)

with col1:
    wc_pos = WordCloud(
        width=600,
        height=400,
        background_color="white"
    ).generate(pos_text)

    fig_pos, ax_pos = plt.subplots(figsize=(6,4))
    ax_pos.imshow(wc_pos, interpolation="bilinear")
    ax_pos.axis("off")
    ax_pos.set_title("Positive Sentiment")
    st.pyplot(fig_pos)

with col2:
    wc_neg = WordCloud(
        width=600,
        height=400,
        background_color="white"
    ).generate(neg_text)

    fig_neg, ax_neg = plt.subplots(figsize=(6,4))
    ax_neg.imshow(wc_neg, interpolation="bilinear")
    ax_neg.axis("off")
    ax_neg.set_title("Negative Sentiment")
    st.pyplot(fig_neg)

# =========================
# Emotion-Based WordCloud (OPTIONAL / ADVANCED)
# =========================
st.subheader("Emotion-Based WordCloud (Optional)")

selected_emotion = st.selectbox(
    "Select Emotion",
    df["emotion"].unique()
)

emotion_text = " ".join(
    df[df["emotion"] == selected_emotion]["text"].astype(str)
)

wc_emo = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(emotion_text)

fig_emo_wc, ax_emo_wc = plt.subplots(figsize=(8,4))
ax_emo_wc.imshow(wc_emo, interpolation="bilinear")
ax_emo_wc.axis("off")
ax_emo_wc.set_title(f"Emotion WordCloud: {selected_emotion}")
st.pyplot(fig_emo_wc)
