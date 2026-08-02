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
    st.error(f"'{dataset_file}' not found in the current directory. Please make sure the dataset file is in the same folder as this script.")
    st.info(" You can create a file named `dataset.csv` and paste the CSV dataset into it.")
else:
    # Display dataset info
    with st.expander(" View Training Dataset"):
        st.dataframe(df, use_container_width=True)
        st.write(f"Total samples: **{len(df)}**")
     # Train Model Pipeline
    @st.cache_resource
    def train_model(data):
        # We need to fill any NaN values just in case
        data = data.dropna(subset=['text', 'emoji'])
        # Train Model Pipeline
    @st.cache_resource
    def train_model(data):
        # We need to fill any NaN values just in case
        data = data.dropna(subset=['text', 'emoji'])
        
        # Build a pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), stop_words='english', min_df=1)),
            ('clf', LogisticRegression(C=1.0, max_iter=200, multi_class='multinomial'))
        ])
        
        pipeline.fit(data['text'], data['emoji'])
        return pipeline
    with st.spinner("Training ML Model..."):
        model = train_model(df)
    
    st.success("✅ Model trained successfully!")
     # User Input Section
    st.markdown("---")
    st.subheader("💡 Try it out!")
    user_input = st.text_input("Enter a sentence to predict its emoji:", placeholder="I want to eat pizza tonight!")
    if user_input:
        # Make prediction
        prediction = model.predict([user_input])[0]
        
        # Get probability distribution to show confidence
        probabilities = model.predict_proba([user_input])[0]
        classes = model.classes_
        pred_index = list(classes).index(prediction)
        confidence = probabilities[pred_index] * 100
        # Display result
        st.markdown("### Result:")
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown(f"<h1 style='text-align: center; font-size: 80px; margin: 0;'>{prediction}</h1>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"**Predicted Emoji:** `{prediction}`")
            st.markdown(f"**Confidence:** `{confidence:.2f}%`")
            
        # Top 3 suggestions
        top_indices = probabilities.argsort()[-3:][::-1]
        st.markdown("#### Top 3 Suggestions:")
        for idx in top_indices:
            st.write(f"- {classes[idx]} ({probabilities[idx]*100:.1f}%)")
            
