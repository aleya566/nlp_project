import streamlit as st

st.set_page_config(
    page_title="Sentiment & Emotion Dashboard",
    page_icon="💬",
    layout="wide"
)

home = st.Page(
    "home.py",
    title="Live Text Analysis",
    icon=":material/edit:",
    default=True
)

visualize = st.Page(
    "visualization.py",
    title="Visualization Dashboard",
    icon=":material/bar_chart:"
)

pg = st.navigation(
    {
        "Menu": [home, visualize]
    }
)

pg.run()
