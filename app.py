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
# Locate dataset.csv
dataset_file = "dataset.csv"
# Load the dataset
df = load_data(dataset_file)
if df is None:
    st.error(f"❌ '{dataset_file}' not found in the current directory. Please make sure the dataset file is in the same folder as this script.")
    st.info("💡 You can create a file named `dataset.csv` and paste the CSV dataset into it.")
else:
    # Display dataset info
    with st.expander("📊 View Training Dataset"):
        st.dataframe(df, use_container_width=True)
        st.write(f"Total samples: **{len(df)}**")
