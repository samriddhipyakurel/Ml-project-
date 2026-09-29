import pandas as pd
import os
from preprocess import clean_text

def load_and_preprocess_data(file_path="dataset.csv"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at {file_path}")
    
    df = pd.read_csv(file_path)
    df = df.dropna(subset=["text", "emoji"])
    df["clean_text"] = df["text"].apply(clean_text)
    print(f"Successfully loaded and preprocessed {len(df)} samples.")
    return df

if __name__ == "__main__":
    df = load_and_preprocess_data()