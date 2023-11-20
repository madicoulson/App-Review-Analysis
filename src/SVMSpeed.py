# Import necessary libraries
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Sample list of app store reviews and their corresponding labels (0 for not related to speed, 1 for related to speed)
reviews = [
    "The app crashes frequently and is very frustrating.",
    "The speed of this app is awesome!",
    "The speed of this app is way too slow.",
    "The app frequently loses data. Implement a reliable auto-save feature to prevent data loss.",
    "The app takes too long to load. Optimize the speed for better user experience.",
    "The latest update improved the app's performance significantly. Great job!",
    "The app frequently crashes, making it frustrating to use. Please fix this issue.",
    "The latest update made the app slower on my device. Improve the performance.",
    "The app is reliable, and I've had no issues with it so far.",
    "The app's design is outdated. It needs a modern and fresh look.",
    
    "This app is amazing. I love it!",
    "Decent app, but needs improvement.",
    "I love this app.",
    "I would not purchase this app. It is a waste of money.",
    "This app is terrible. The UI is very poor.",
    "I use this app daily and it works very well.",
    "I like the simple design of this app.",
    "I regret purchasing this app. It doesn't live up to its promises.",
    "The app is user-friendly, and the customer support is excellent.",
    "This app is perfect for my needs. It has everything I was looking for."
]

# Labels (0 for not related to speed, 1 for related to speed)
labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# Convert text data to TF-IDF features
tfidf_vectorizer = TfidfVectorizer(max_features=10000)
X = tfidf_vectorizer.fit_transform(reviews)

# Create and train the SVM model
svm_classifier = SVC(kernel='linear', C=1, random_state=42)
svm_classifier.fit(X, labels)

# New reviews for prediction
new_reviews = [
    "This app is super fast and efficient!",
    "The speed of this app is disappointing.",
    "I love the sleek design of the app. It's visually appealing.",
    "The app crashes every time I open it. It's frustrating and needs fixing.",
    "Great app overall, but it lacks a dark mode. Consider adding this feature."
]

# Convert new review text to TF-IDF features
new_reviews_tfidf = tfidf_vectorizer.transform(new_reviews)

# Predict labels for new reviews
predicted_labels = svm_classifier.predict(new_reviews_tfidf)

# Interpret the predictions
speed_labels = {0: 'Not Related to Speed', 1: 'Related to Speed'}
predicted_labels = [speed_labels[pred] for pred in predicted_labels]

# Display the predictions
for review, label in zip(new_reviews, predicted_labels):
    print(f"Review: {review}\nPredicted Label: {label}\n")
