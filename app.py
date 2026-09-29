import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from preprocess import clean_text

st.set_page_config(page_title="Text to Emoji Predictor", page_icon="🔮")

st.title("🔮 Text to Emoji Predictor")

def load_data(filepath="dataset.csv"):
    if not os.path.exists(filepath):
        return None
    return pd.read_csv(filepath)

df = load_data()

if df is not None:
    df_clean = df.dropna(subset=['text', 'emoji']).copy()
    df_clean['clean_text'] = df_clean['text'].apply(clean_text)
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), stop_words='english', min_df=1)),
        ('clf', LogisticRegression(C=1.0, max_iter=200, multi_class='multinomial'))
    ])
    pipeline.fit(df_clean['clean_text'], df_clean['emoji'])
    
    user_input = st.text_input("Enter text:")
    if user_input:
        cleaned = clean_text(user_input)
        prediction = pipeline.predict([cleaned])[0]
        
        probabilities = pipeline.predict_proba([cleaned])[0]
        classes = pipeline.classes_
        pred_index = list(classes).index(prediction)
        confidence = probabilities[pred_index] * 100
        
        st.subheader(f"Predicted Emoji: {prediction}")
        st.write(f"Confidence: {confidence:.2f}%")
        
        st.markdown("#### Top 3 Suggestions:")
        top_indices = probabilities.argsort()[-3:][::-1]
        for idx in top_indices:
            st.write(f"- {classes[idx]} ({probabilities[idx]*100:.1f}%)")
