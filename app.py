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