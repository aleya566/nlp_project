import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.title("📊 Dataset Visualization Dashboard")

# ======================================
# DATASET UPLOAD
# ======================================
st.subheader("📤 Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV file (must contain 'text' and 'airline_sentiment' columns)",
    type=["csv"]
)

@st.cache_data
def load_default_data():
    return pd.read_csv("Tweets.csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Custom dataset loaded successfully!")
else:
    df = load_default_data()
    st.info("Using default Tweets.csv dataset")

# ======================================
# SENTIMENT DISTRIBUTION
# ======================================
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

# ======================================
# LOAD EMOTION MODEL
# ======================================
@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=False
    )

emotion_model = load_emotion_model()

# ======================================
# EMOTION DISTRIBUTION (SAMPLED)
# ======================================
st.subheader("Emotion Distribution")

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

# ======================================
# WORD CLOUD - BY SENTIMENT
# ======================================

st.subheader("☁️ WordCloud by Sentiment")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("Positive")
    text = " ".join(df[df["airline_sentiment"] == "positive"]["text"])

with col2:
    st.write("Neutral")
    text = " ".join(df[df["airline_sentiment"] == "neutral"]["text"])

with col3:
    st.write("Negative")
    text = " ".join(df[df["airline_sentiment"] == "negative"]["text"])


# ======================================
# WORD CLOUD - BY EMOTION ⭐ NEW
# ======================================
st.subheader("☁️ WordCloud by Emotion")

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
    plt.imshow(wc_emotion, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"WordCloud for Emotion: {selected_emotion.capitalize()}")
    st.pyplot(plt)
else:
    st.warning("Not enough text for selected emotion.")
