# Sentiment Analysis Project

This repository contains a sentiment analysis pipeline for analyzing and predicting the sentiment of ChatGPT review data. The project covers data cleaning, exploratory data analysis (EDA), model training, and deployment-ready prediction scripts.

---

## Project Structure

sentiment_analysis/
│
├── cleaned_data/
│ └── chatgpt_reviews_cleaned.csv # Cleaned review dataset
│
├── eda/
│ └── sentiment_analysis_eda.ipynb # EDA notebook for sentiment analysis
│
├── model/
│ ├── categorical_label_encoders.pkl # Encoders for categorical features
│ ├── model_lr.pkl # Trained Logistic Regression model
│ ├── model_nb.pkl # Trained Naive Bayes model
│ ├── model_rf.pkl # Trained Random Forest model
│ └── tfidf_vectorizer.pkl # TF-IDF vectorizer
│
├── notebooks/
│ ├── sentiment_eda_app.ipynb # Streamlit EDA app notebook
│ └── sentiment_model.ipynb # Notebook for training and evaluating models
│
└── sent_pred.py # Script for loading models and making predictions




---

## Contents

- **cleaned_data/**  
  Contains the preprocessed and cleaned dataset used for analysis and model training.

- **eda/**  
  Contains Jupyter notebooks for exploratory data analysis, including visualizations and insights about the sentiment data.

- **model/**  
  Stores trained machine learning models (Logistic Regression, Naive Bayes, Random Forest), label encoders, and the TF-IDF vectorizer used for feature extraction.

- **notebooks/**  
  - **sentiment_eda_app.ipynb:** Interactive Streamlit notebook for EDA.
  - **sentiment_model.ipynb:** Notebook for model training, evaluation, and comparison.

- **sent_pred.py**  
  Python script for loading trained models and making sentiment predictions on new data.

---

## Getting Started

1. **Clone the repository:**

2. **Set up the environment:**
- Install required packages (example using `requirements.txt`):
  ```
  pip install -r requirements.txt
  ```

3. **Data Preparation:**
- The cleaned dataset is available in `cleaned_data/chatgpt_reviews_cleaned.csv`.

4. **Exploratory Data Analysis:**
- Open and run `eda/sentiment_analysis_eda.ipynb` or `notebooks/sentiment_eda_app.ipynb` for interactive EDA.

5. **Model Training & Evaluation:**
- Use `notebooks/sentiment_model.ipynb` to train, evaluate, and compare models.
- Trained models and encoders are saved in the `model/` directory.

6. **Making Predictions:**
- Run `sent_pred.py` to load models and predict sentiment for new text inputs.

---


