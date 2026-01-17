import matplotlib.pyplot as plt
import plotly.express as px
import altair as alt
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
from pathlib import Path

load_dotenv()
client = OpenAI() 

st.title("Hello GenAI")
st.write("Type a prompt, adjust creativity, get an answer.")

user_prompt = st.text_input(
    "Enter your prompt",
    value="Explain generative AI in one sentence."
)

temperature = st.slider(
    "Creativity (temperature)",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.1
)

@st.cache_data(show_spinner=False)
def get_response(prompt: str, temp: float) -> str:
    resp = client.responses.create(
        model="gpt-4o-mini",
        input=[{"role": "user", "content": prompt}],
        temperature=temp
    )
    return resp.output_text

if user_prompt:
    with st.spinner("Thinking..."):
        answer = get_response(user_prompt, temperature)
    st.write(answer)

st.divider()
st.header("Dataset workflow")

col1, col2 = st.columns(2)

with col1:
    load_btn = st.button("1) Load dataset")

with col2:
    clean_btn = st.button("2) Clean reviews")

import pandas as pd
from pathlib import Path

# ---- Dataset paths ----
DATA_PATH = Path(__file__).parent / "data" / "customer_reviews.csv"

# ---- Button: Load dataset ----
if load_btn:
    if not DATA_PATH.exists():
        st.error(f"Can't find CSV at: {DATA_PATH}")
        st.stop()

st.session_state.df = pd.read_csv(DATA_PATH)

with st.expander("Debug: show columns"):
    st.write(st.session_state.df.columns.tolist())

st.success(f"Loaded {len(st.session_state.df):,} rows from customer_reviews.csv")


# Filter dataset by product
if "df" in st.session_state:
    st.subheader("Filter by Product")

    products = ["All Products"] + sorted(
        st.session_state.df["PRODUCT"].unique().tolist()
    )

    selected_product = st.selectbox("Choose a product", products)

    if selected_product == "All Products":
        filtered_df = st.session_state.df
    else:
        filtered_df = st.session_state.df[
            st.session_state.df["PRODUCT"] == selected_product
        ]

    # Display dataset preview
    st.subheader("Dataset Preview")
    st.dataframe(filtered_df.head(30), use_container_width=True)

st.divider()
st.subheader("📊 Mean sentiment by product")

SENTIMENT_COL = "SENTIMENT_SCORE"

if "df" not in st.session_state:
    st.info("Load the dataset to view sentiment charts.")
elif SENTIMENT_COL not in st.session_state.df.columns:
    st.info("Sentiment column not available yet.")
else:
    chart_df = (
        filtered_df
        .groupby("PRODUCT")[SENTIMENT_COL]
        .mean()
        .sort_values()
    )

    st.bar_chart(chart_df)