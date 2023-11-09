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
    "This app is great. I would recommend!",
    "I love this app.",
    "This app is terrible. The UI is very poor.",
    "I love the security of this app.",
    "I wish this app had more feautures.",
    "This is my favorite app on the app store.",  
    
    "I like this app, but there are better substitutes.",
    "This app is horrible.", 
    "I like the design of the app.",
    "The features on this app are cool.",
    "I would not purchase this app. It is a waste of money.", 
    "There are a lot of bugs in this app.",
    "The new update for this app was great.",
    "I have heard great things about this app.",
    "There are security issues with this app.",
    "I use this app daily and it works very well.",
    
    "I do not like this app.",
    "This app is very bad.",
    "This app is decent.",
    "The speed of this app is awesome!",
    "The speed of this app is way too slow.",
    "This is a great app.",
    "I don't know why anyone would like this app.",
    "This app deserves a zero star rating.",
    "I use this app from time to time.",
    "This app is extremely unsecure and bad."    
]

sentiments = [2, 0, 1, 0, 2, 2, 0, 2, 1, 2, 
              1, 0, 2, 2, 0, 0, 2, 2, 0, 2,
              0, 0, 1, 2, 0, 2, 0, 0, 1, 0]  # Adjust the sentiment labels accordingly

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
4
# Interpret the predictions
sentiment_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
predicted_sentiments = [sentiment_labels[pred] for pred in predicted_sentiments]

# Display the predictions
for review, sentiment in zip(new_reviews, predicted_sentiments):
    print(f"Review: {review}\nPredicted Sentiment: {sentiment}\n")

