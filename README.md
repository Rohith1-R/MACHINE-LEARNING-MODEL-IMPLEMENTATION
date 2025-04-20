# MACHINE-LEARNING-MODEL-IMPLEMENTATION
COMPANY:CODTECH IT SOLUTIONS NAME:ROHITH R INTERN ID:CT04WP74 DOMAIN:PYTHON BATCH DURATION:March 21st/2025 to April 21st/2025

DESCRIPTION

Spam Email Detection Using Machine Learning

Overview

This project builds a spam email classifier using Scikit-Learn and Natural Language Processing (NLP) techniques. The goal is to develop a predictive model that can classify emails as spam (1) or ham (0) (not spam).

The model follows a structured machine learning pipeline: 1. Load and preprocess the dataset 2. Convert text data into numerical features using TF-IDF 3. Train a Naïve Bayes classifier 4. Evaluate the model’s accuracy

This approach helps automate spam detection, reducing the time and effort required to filter out unwanted emails.

Key Features of the Code

Importing Required Libraries
The script begins by importing the necessary Python libraries:

import pandas as pd import numpy as np import nltk import string from sklearn.feature_extraction.text import TfidfVectorizer from sklearn.model_selection import train_test_split from sklearn.naive_bayes import MultinomialNB from sklearn.metrics import accuracy_score, classification_report

•	Pandas & NumPy: For data manipulation and analysis.
•	NLTK (Natural Language Toolkit): For text preprocessing, including stopword removal.
•	Scikit-Learn: Provides tools for vectorization, model training, and evaluation.
Downloading Stopwords for Text Preprocessing
The script downloads a predefined list of common words (stopwords) in English using NLTK.

nltk.download("stopwords") from nltk.corpus import stopwords

•	Stopwords like “the”, “is”, “in”, etc., are removed as they do not contribute to spam detection.
Loading and Preprocessing the Dataset
The dataset (CSV file) is loaded using Pandas:

df = pd.read_csv("spam.csv", encoding="latin-1") df = df.iloc[:, [0, 1]] # Selecting relevant columns df.columns = ["label", "message"] # Renaming columns

•	The dataset contains spam and ham messages.
•	Only the label (spam/ham) and message columns are selected.
•	Labels are converted into binary values:
•	Spam = 1
•	Ham = 0
df["label"] = df["label"].map({"ham": 0, "spam": 1})

Text Preprocessing
Each email message is cleaned using a preprocessing function:

def preprocess_text(text): text = text.lower() # Convert to lowercase text = "".join([char for char in text if char not in string.punctuation]) # Remove punctuation words = text.split() # Tokenization words = [word for word in words if word not in stopwords.words("english")] # Remove stopwords return " ".join(words)

This function: ✅ Converts text to lowercase (ensures uniformity). ✅ Removes punctuation (e.g., !@#$%^&*). ✅ Splits text into words (tokenization). ✅ Removes stopwords (e.g., “the”, “is”, “and”).

df["message"] = df["message"].apply(preprocess_text)

•	The function is applied to all email messages to prepare them for analysis.
Feature Extraction using TF-IDF
After preprocessing, we convert text into numerical data using TF-IDF (Term Frequency-Inverse Document Frequency):

vectorizer = TfidfVectorizer(max_features=5000) # Limit to top 5000 words X = vectorizer.fit_transform(df["message"]).toarray() y = df["label"]

•	TF-IDF assigns weights to words based on their frequency across all messages.
•	The top 5000 most important words are selected.
Splitting Data into Training and Testing Sets
The dataset is split into: • 80% for training • 20% for testing

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

This ensures that the model is trained and evaluated on separate data.

Training the Naïve Bayes Classifier
The Multinomial Naïve Bayes model is trained on the processed data:

model = MultinomialNB() model.fit(X_train, y_train)

•	Naïve Bayes is a popular algorithm for text classification.
•	It assumes that words are conditionally independent, making it computationally efficient.
Making Predictions & Evaluating Model Performance
After training, the model is tested on unseen data:

y_pred = model.predict(X_test)

•	The model predicts spam or ham labels for test messages.
The accuracy and classification report are calculated:

accuracy = accuracy_score(y_test, y_pred) report = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}") print("Classification Report:\n", report)

•	Accuracy Score: Measures the percentage of correct predictions.
•	Classification Report: Provides detailed metrics like precision, recall, and F1-score.
Expected Output

After running the script, you should see an output similar to:

Model Accuracy: 0.98 Classification Report: precision recall f1-score support

       0       0.99      0.99      0.99      965
       1       0.93      0.94      0.94      150

accuracy                           0.98     1115
macro avg 0.96 0.97 0.96 1115 weighted avg 0.98 0.98 0.98 1115

•	98% accuracy means the model is highly effective at detecting spam.
•	High precision & recall for spam (1) ensures low false positives.
Key Benefits of This Model

✅ Automates spam detection with minimal human intervention. ✅ Uses NLP techniques to extract meaningful features from text. ✅ Computationally efficient with Naïve Bayes classification. ✅ High accuracy & recall ensures effective email filtering.

Potential Improvements

🚀 Deep Learning: Use LSTMs or Transformers (BERT) for better performance. 🚀 Larger Dataset: Train on a bigger and diverse dataset. 🚀 Feature Engineering: Add n-grams, word embeddings, or sentiment analysis. 🚀 Deployment: Convert the model into an API for real-time spam filtering.

Conclusion

This project demonstrates a practical application of machine learning in spam detection. By leveraging Scikit-Learn, NLP, and TF-IDF, we built a highly accurate model for classifying emails as spam or ham.