import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Text to Emoji Predictor", page_icon="🔮")

st.title("🔮 Text to Emoji Predictor")
st.write("Welcome to the Text to Emoji Machine Learning Predictor.")

def load_dataset():
    if os.path.exists("dataset.csv"):
        return pd.read_csv("dataset.csv")
    return None

df = load_dataset()
if df is not None:
    st.write(f"Loaded dataset with {len(df)} samples.")
