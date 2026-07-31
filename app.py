import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
# Set page configuration
st.set_page_config(
    page_title="Text to Emoji Predictor",
    page_icon="🔮",
    layout="centered"
)
# App Header
st.title("🔮 Text to Emoji Predictor")
st.markdown(
    """
    This app trains a machine learning model (TF-IDF + Logistic Regression) 
    on your custom `dataset.csv` and predicts the best emoji matching your text!
    """
)
# Helper function to load dataset
@st.cache_data
def load_data(filepath):
    if not os.path.exists(filepath):
        return None
    return pd.read_csv(filepath)
