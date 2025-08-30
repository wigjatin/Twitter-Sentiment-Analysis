# NLP Twitter Sentiment Analysis with Logistic Regression

A lightweight NLP-powered sentiment analysis system that classifies tweets into positive or negative categories. Built using Logistic Regression and text vectorization techniques for simplicity, speed, and interpretability.

---

## Problem Statement

Social media platforms like Twitter are flooded with opinions, reactions, and emotions. Understanding sentiment at scale is valuable for:

- Businesses tracking customer feedback
- Researchers analyzing public opinion
- Individuals filtering through online chaos

Manual classification is inefficient and inconsistent. This project automates sentiment detection with a robust NLP pipeline.

---

## Key Features

- Binary Sentiment Classification → Only "Positive" or "Negative" 
- TF-IDF Vectorization for transforming tweets into numerical features
- Preprocessing pipeline with stopword removal, tokenization, and normalization
- Interpretable Logistic Regression model for fast, explainable results
- Streamlit Web App for real-time tweet analysis


---

## Model Architecture

```python
LogisticRegression(
        C=1.0,
    solver="liblinear",
    max_iter=1000,
    random_state=42
)
Vectorizer: TfidfVectorizer (word-level, unigram + bigram support)
Model: Logistic Regression (linear decision boundary for sentiment separation)
```
---

## Demo

You can access the live demo of the application by visiting the following link:  
[View Demo](https://twitter-sentiment-analysis-by-jatinwig.streamlit.app/)
