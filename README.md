# Text to Emoji ML Predictor 🔮

An end-to-end Machine Learning application built in Python and Streamlit that maps English text sentences to appropriate emojis using Natural Language Processing (NLP).

## 🚀 Features
- **Custom NLP Preprocessing**: Text normalization, lowercasing, URL removal, and punctuation stripping.
- **TF-IDF + Logistic Regression**: Scikit-Learn machine learning pipeline.
- **Interactive Web UI**: Streamlit web interface for real-time predictions and dataset browsing.
- **Probability & Confidence Scores**: Displays prediction confidence and top 3 emoji suggestions.

## 📁 Repository Structure
```
├── dataset.csv        # 90+ text-to-emoji training samples
├── preprocess.py     # Text cleaning & normalization utilities
├── train_model.py    # Offline model training & evaluation script
├── app.py            # Interactive Streamlit Web application
├── requirements.txt  # Python package dependencies
└── README.md         # Project documentation
```

## 🛠️ Installation & Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/samriddhipyakurel/Ml-project-.git
   cd Ml-project-
   ```
2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Running the Application
To start the interactive Streamlit dashboard:
```bash
streamlit run app.py
```

To run model training offline:
```bash
python train_model.py
```
