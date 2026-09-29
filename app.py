import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from preprocess import clean_text

st.set_page_config(page_title="Text to Emoji Predictor", page_icon="🔮", layout="centered")

st.title("🔮 Text to Emoji Predictor")
st.markdown(
    """
    This app trains a Machine Learning pipeline (TF-IDF + Logistic Regression) 
    on your dataset and predicts the best matching emoji for any sentence!
    """
)

def load_data(filepath="dataset.csv"):
    if not os.path.exists(filepath):
        return None
    return pd.read_csv(filepath)

df = load_data()

if df is None:
    st.error("❌ `dataset.csv` not found in the current directory.")
else:
    with st.expander("📊 View Training Dataset"):
        st.dataframe(df, use_container_width=True)
        st.write(f"Total samples: **{len(df)}**")

    df_clean = df.dropna(subset=['text', 'emoji']).copy()
    df_clean['clean_text'] = df_clean['text'].apply(clean_text)
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), stop_words='english', min_df=1)),
        ('clf', LogisticRegression(C=1.0, max_iter=200, multi_class='multinomial'))
    ])
    pipeline.fit(df_clean['clean_text'], df_clean['emoji'])
    
    st.markdown("---")
    st.subheader("💡 Try it out!")
    user_input = st.text_input("Enter a sentence to predict its emoji:", placeholder="I want to eat pizza tonight!")

    if user_input:
        cleaned_input = clean_text(user_input)
        prediction = pipeline.predict([cleaned_input])[0]
        
        probabilities = pipeline.predict_proba([cleaned_input])[0]
        classes = pipeline.classes_
        pred_index = list(classes).index(prediction)
        confidence = probabilities[pred_index] * 100

        st.markdown("### Result:")
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown(f"<h1 style='text-align: center; font-size: 80px; margin: 0;'>{prediction}</h1>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"**Original Input:** *\"{user_input}\"*")
            st.markdown(f"**Cleaned Input:** *\"{cleaned_input}\"*")
            st.markdown(f"**Predicted Emoji:** `{prediction}`")
            st.markdown(f"**Confidence:** `{confidence:.2f}%`")
            
        top_indices = probabilities.argsort()[-3:][::-1]
        st.markdown("#### Top 3 Suggestions:")
        for idx in top_indices:
            st.write(f"- {classes[idx]} ({probabilities[idx]*100:.1f}%)")
