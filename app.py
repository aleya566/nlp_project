import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Sentiment & Emotion Dashboard",
    page_icon="💬",
    layout="wide"
)

st.title("💬 Sentiment & Emotion Analysis Dashboard")

# ==============================
# LOAD DATASET
# ==============================
@st.cache_data
def load_data():
    df = pd.read_csv("Tweets.csv")
    return df

df = load_data()

# ==============================
# LOAD MODELS
# ==============================
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

# ==============================
# SIDEBAR NAVIGATION
# ==============================
page = st.sidebar.selectbox(
    "Select Page",
    ["Live Text Analysis", "Visualization Dashboard"]
)

# ======================================================
# PAGE 1: LIVE TEXT ANALYSIS
# ======================================================
if page == "Live Text Analysis":

    st.subheader("📝 Live Sentiment & Emotion Analysis")

    user_text = st.text_area(
        "Enter review or social media text:",
        height=120,
        placeholder="Example: I am very angry and disappointed with the service."
    )

    if st.button("Analyze") and user_text.strip() != "":

        # Sentiment
        sent_out = sentiment_model(user_text)[0]
        sentiment = label_map[sent_out["label"]]
        sent_score = sent_out["score"]

        # Emotion
        emo_out = emotion_model(user_text)[0]
        emotion = emo_out["label"]
        emo_score = emo_out["score"]

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Sentiment", sentiment, f"{sent_score:.2f}")

        with col2:
            st.metric("Emotion", emotion.capitalize(), f"{emo_score:.2f}")

        st.divider()

        # Charts
        st.subheader("📊 Prediction Confidence")

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

# ======================================================
# PAGE 2: VISUALIZATION DASHBOARD (DATASET)
# ======================================================
if page == "Visualization Dashboard":

    st.subheader("📊 Dataset-Based Visualization")

    # -------------------------------
    # Sentiment Distribution
    # -------------------------------
    sent_counts = df["airline_sentiment"].value_counts().reset_index()
    sent_counts.columns = ["Sentiment", "Count"]

    fig_sent_dist = px.bar(
        sent_counts,
        x="Sentiment",
        y="Count",
        title="Sentiment Distribution (Dataset)"
    )

    st.plotly_chart(fig_sent_dist, use_container_width=True)

    # -------------------------------
    # Emotion Prediction on Dataset
    # -------------------------------
    st.subheader("Emotion Distribution (Sampled Data)")

    sample_df = df.sample(300, random_state=42)
    sample_df["emotion"] = sample_df["text"].apply(
        lambda x: emotion_model(x)[0]["label"]
    )

    emo_counts = sample_df["emotion"].value_counts().reset_index()
    emo_counts.columns = ["Emotion", "Count"]

    fig_emo_dist = px.bar(
        emo_counts,
        x="Emotion",
        y="Count",
        title="Emotion Distribution"
    )

    st.plotly_chart(fig_emo_dist, use_container_width=True)

    # -------------------------------
    # WORD CLOUDS
    # -------------------------------
    st.subheader("☁️ Word Cloud")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Positive Reviews")
        pos_text = " ".join(df[df["airline_sentiment"] == "positive"]["text"])
        wc_pos = WordCloud(width=600, height=400, background_color="white").generate(pos_text)

        plt.figure(figsize=(6,4))
        plt.imshow(wc_pos, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)

    with col2:
        st.write("Negative Reviews")
        neg_text = " ".join(df[df["airline_sentiment"] == "negative"]["text"])
        wc_neg = WordCloud(width=600, height=400, background_color="white").generate(neg_text)

        plt.figure(figsize=(6,4))
        plt.imshow(wc_neg, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)

# ==============================
# FOOTER
# ==============================
st.caption(
    "Built using Streamlit, HuggingFace Transformers, Plotly & WordCloud"
)
