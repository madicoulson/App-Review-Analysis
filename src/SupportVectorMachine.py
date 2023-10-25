# Import necessary libraries
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Sample list of app store reviews and their corresponding sentiments
reviews = [
    "This app is amazing. I love it!",
    "The app crashes frequently and is very frustrating.",
    "Decent app, but needs improvement.",
    "The worst app I've ever used. Avoid at all costs!",
]

sentiments = [2, 0, 1, 0]  # 0: Negative, 1: Neutral, 2: Positive

# Convert text data to TF-IDF features
tfidf_vectorizer = TfidfVectorizer(max_features=10000)  # You can adjust the number of features
X = tfidf_vectorizer.fit_transform(reviews)

# Create and train the SVM model
svm_classifier = SVC(kernel='linear', C=1, random_state=42)
svm_classifier.fit(X, sentiments)

# New reviews for prediction
new_reviews = [
    "Great app! It's very useful and user-friendly.",
    "I don't like this app. It's confusing and not helpful at all.",
]

# Convert new review text to TF-IDF features
new_reviews_tfidf = tfidf_vectorizer.transform(new_reviews)

# Predict sentiments for new reviews
predicted_sentiments = svm_classifier.predict(new_reviews_tfidf)

# Interpret the predictions
sentiment_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
predicted_sentiments = [sentiment_labels[pred] for pred in predicted_sentiments]

# Display the predictions
for review, sentiment in zip(new_reviews, predicted_sentiments):
    print(f"Review: {review}\nPredicted Sentiment: {sentiment}\n")

