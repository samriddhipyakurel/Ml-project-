import pandas as pd
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from preprocess import clean_text

def load_and_preprocess_data(file_path="dataset.csv"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at {file_path}")
    
    df = pd.read_csv(file_path)
    df = df.dropna(subset=["text", "emoji"])
    df["clean_text"] = df["text"].apply(clean_text)
    return df

def build_model_pipeline():
    return Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), stop_words='english', min_df=1)),
        ('clf', LogisticRegression(C=1.0, max_iter=200))
    ])

def train_and_save(model_path="model.pkl"):
    df = load_and_preprocess_data()
    pipeline = build_model_pipeline()
    pipeline.fit(df["clean_text"], df["emoji"])
    
    predictions = pipeline.predict(df["clean_text"])
    acc = accuracy_score(df["emoji"], predictions)
    print(f"Training Accuracy: {acc * 100:.2f}%")
    
    # Save model artifact
    joblib.dump(pipeline, model_path)
    print(f"Saved trained model to {model_path}")
    return pipeline

if __name__ == "__main__":
    train_and_save()