# Import necessary libraries
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Sample list of app store reviews and their corresponding labels (0 for not actionable, 1 for actionable)
reviews = [
    "This app is amazing. I love it!",
    "The app crashes frequently and is very frustrating.",
    "Decent app, but needs improvement.",
    "The worst app I've ever used. Avoid at all costs!",
    "This app is great. I would recommend!",
    "I love this app.",
    "This app is terrible. The UI is very poor.",
    "I love the security of this app.",
    "I wish this app had more features.",
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
    "This app deserves a zero-star rating.",
    "I use this app from time to time.",
    "This app is extremely unsecure and bad.",
    
    "The login process is too complicated. Streamlining it would improve the overall user experience.",
    "The app crashes every time I try to open it.",
    "This app is fantastic!",
    "The UI is very confusing and needs redesign.",
    "The app always freezes on my device. The performance needs resolved.",
    "I hate this app so much.",
    "This is the best app I have every downloaded!",
    "This app is not secure! My data has been leaked many times.",
    "The speed of the app is too slow. It needs fixing.",
    "This is my favorite app.",
    
    "In the next update, there should be more features added.",
    "I like the simple design of this app.",
    "The app frequently loses data. Implement a reliable auto-save feature to prevent data loss.",
    "There are several bugs that you should add fixes for involving security.",
    "I love how easy the app is to navigate.",
    "This app costs too much money.",
    "There is a security vulnerability in the app. I suggest implementing stronger encryption measures.",
    "I highly recommend this app!",
    "The speed needs to be updated in the next update. The app runs much slower than competitors.",
    "This app stinks.",

    "The app crashes occasionally. Please investigate and fix this issue.",
    "This app is a lifesaver. It has made my daily tasks so much easier.",
    "Not bad, but the user interface could be more intuitive.",
    "I encountered a security issue while using this app. Please address it urgently.",
    "The latest update caused my phone to freeze. Fix this bug in the next release.",
    "The app is simple and effective. I use it for work every day.",
    "This app is worth the price. It has all the features I need.",
    "The customer support for this app is responsive and helpful.",
    "I enjoy using this app. It has improved my productivity.",
    "I uninstalled the app because of constant crashes.",
    
    "The app's design is outdated. Consider a modern redesign for a fresh look.",
    "The app is too expensive for the limited features it offers.",
    "Great app overall, but the login process needs improvement.",
    "I encountered a bug that caused data loss. Implement better error handling.",
    "The app is user-friendly and straightforward.",
    "I regret purchasing this app. It doesn't live up to its promises.",
    "This app is a must-have. It has everything I need for my daily tasks.",
    "The app takes too long to load. Optimize the speed for better user experience.",
    "I appreciate the regular updates. Keep up the good work!",
    "The app crashes on startup. This issue needs immediate attention.",
    
    "This app is a game-changer. It revolutionized how I organize my tasks.",
    "Average app. Could use some improvements in functionality.",
    "The app is reliable, and I've had no issues with it so far.",
    "This app exceeded my expectations. It's feature-packed and easy to use.",
    "The app is a bit pricey, but the quality justifies the cost.",
    "I encountered a bug in the latest version. Please release a fix soon.",
    "This app is a gem. It simplifies complex tasks effortlessly.",
    "Not satisfied with the app's performance. It lags frequently.",
    "I recommend this app to everyone. It's a must-try!",
    "The app is user-friendly, and the customer support is excellent.",
    
    "This app is perfect for my needs. It has everything I was looking for.",
    "The app constantly crashes, making it unusable. Fix the stability issues.",
    "The user interface is clean and intuitive. I appreciate the design.",
    "I encountered a security vulnerability while using this app. Please address it promptly.",
    "The latest update improved the app's performance on my device.",
    "I find the app's features lacking. Consider adding more functionalities.",
    "This app is a waste of money. I regret purchasing it.",
    "The app's design is outdated. It needs a modern and fresh look.",
    "The app is too expensive for the limited features it offers.",
    "I encountered a bug that caused data loss. Implement better error handling.",
    
    "The app crashes every time I open it. It's frustrating and needs fixing.",
    "This app is a lifesaver! It has streamlined my daily tasks.",
    "The user interface is confusing. Please redesign it for better usability.",
    "Great app overall, but it lacks a dark mode. Consider adding this feature.",
    "The app is reliable, and I've had no issues with it so far.",
    "The latest update made the app slower on my device. Improve the performance.",
    "This app is worth the price. It has all the features I need.",
    "The customer support for this app is responsive and helpful.",
    "The app frequently freezes on my phone. Resolve the performance issues.",
    "I regret purchasing this app. It doesn't live up to its promises.",
    
    "The app frequently crashes, making it frustrating to use. Please fix this issue.",
    "This app is amazing! It has all the features I need for my daily tasks.",
    "The user interface is confusing and needs a redesign for better usability.",
    "Security is a concern. There's a vulnerability that needs immediate attention.",
    "The latest update improved the app's performance significantly. Great job!",
    "The app lacks some essential features. Consider adding them in the next update.",
    "I love the sleek design of the app. It's visually appealing.",
    "There's a bug causing data loss. Please implement better error handling.",
    "Customer support is unresponsive. Improve the service for better user experience.",
    "The app is worth the price. It has all the features I need for my tasks."
]

labels = [0, 1, 1, 0, 0, 0, 1, 1, 1, 0,  
          0, 0, 0, 1, 0, 0, 1, 0, 1, 0,
          0, 1, 0, 1, 1, 0, 0, 0, 1, 1,
          1, 1, 0, 1, 1, 0, 0, 1, 1, 0,
          1, 0, 1, 1, 0, 0, 1, 0, 1, 0,
          1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 
          1, 0, 1, 1, 0, 0, 1, 0, 1, 0,
          1, 1, 0, 0, 0, 1, 0, 1, 0, 0,
          0, 1, 0, 1, 0, 1, 1, 1, 0, 1,
          1, 0, 1, 1, 0, 1, 0, 0, 1, 1,
          1, 0, 1, 1, 0, 1, 0, 1, 0, 0]  

# Convert text data to TF-IDF features
tfidf_vectorizer = TfidfVectorizer(max_features=10000)  # You can adjust the number of features
X = tfidf_vectorizer.fit_transform(reviews)

# Create and train the SVM model
svm_classifier = SVC(kernel='linear', C=1, random_state=42)
svm_classifier.fit(X, labels)

# New reviews for prediction
new_reviews = [
    "Great app! It's very useful and user-friendly.",
    "I don't like this app. It's confusing and not helpful at all.",
    "It would be great to have a dark mode option. Consider adding this feature in the next update.",
    "The app crashes frequently and needs stability improvements.",
    "This app is amazing! It makes my life so much easier.",
    "The user interface is clean and intuitive, but it lacks some key features.",
    "I encountered a security vulnerability in the latest version of the app.",
    "The app's design is outdated and needs a modern refresh.",
    "Great app overall, but it could benefit from a dark mode.",
    "I use this app every day and haven't faced any issues so far.",
    
    "The customer support for this app is responsive and helpful.",
    "The latest update improved the app's performance significantly.",
    "This app is a bit expensive for the features it offers.",
    "The app is user-friendly, but it lacks certain functionalities.",
    "I regret purchasing this app. It doesn't meet my expectations.",
    "The app frequently freezes on my phone, making it frustrating to use.",
    "I love the simplicity of the app. It doesn't overwhelm with unnecessary features.",
    "There's a bug causing data loss in the app. Needs urgent attention.",
    "The app's speed is impressive, providing a smooth user experience.",
    "Overall, a decent app, but it could use some improvements in functionality.",
    
    "The app crashes every time I try to open it. Please fix this issue!",
    "I love the features of this app, but it needs a better user interface.",
    "This app is a lifesaver. It helps me stay organized and efficient.",
    "Security is a major concern. There is a vulnerability that needs immediate attention.",
    "The latest update made the app much slower. Improve the performance!",
    "Great app overall, but it lacks a search functionality. Consider adding it.",
    "The app's design is sleek and modern. I enjoy using it daily.",
    "I encountered a bug that deleted my data. Implement better error handling.",
    "The customer support for this app is unresponsive and unhelpful.",
    "The app is worth the price. It provides all the necessary features and more.",
    
    "The app crashes frequently, making it unusable. Please fix this issue!",
    "This app is fantastic! It meets all my needs and more.",
    "The user interface is confusing, and it needs a redesign for better usability.",
    "Security is a concern. There's a vulnerability that needs immediate attention.",
    "The latest update improved the app's performance significantly. Great job!",
    "The app is reliable, but it lacks some essential features.",
    "I encountered a bug that caused data loss. Implement better error handling.",
    "This app is worth the price. It has all the features I need for my daily tasks.",
    "The customer support for this app is horrible.",
    "The app is user-friendly, and I use it daily for work.",
    
    "The app crashes frequently. Please address this issue promptly.",
    "This app is a game-changer! It's incredibly useful and efficient.",
    "The user interface needs improvement. It's not very intuitive.",
    "I encountered a security vulnerability in the latest version. Urgent fix required.",
    "The latest update significantly improved the app's performance. Great job!",
    "The app lacks some essential features. Consider adding them in the next update.",
    "I love the sleek design of the app. It's visually appealing.",
    "There's a bug causing data loss. Please implement better error handling.",
    "Customer support is unresponsive. Improve the service for better user experience.",
    "The app is worth the price. It has all the features I need for my tasks."
]


# Convert new review text to TF-IDF features
new_reviews_tfidf = tfidf_vectorizer.transform(new_reviews)

# Predict labels for new reviews
predicted_labels = svm_classifier.predict(new_reviews_tfidf)

# Interpret the predictions
actionable_labels = {0: 'Not Actionable', 1: 'Actionable'}
predicted_labels = [actionable_labels[pred] for pred in predicted_labels]

# Display the predictions
for review, label in zip(new_reviews, predicted_labels):
    print(f"Review: {review}\nPredicted Label: {label}\n")
    
# Different labels for new reviews (adjust as needed)
new_reviews_labels = [0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 
                      0, 0, 0, 1, 0, 1, 0, 1, 0, 1,
                      1, 1, 0, 1, 1, 1, 0, 1, 0, 0,
                      1, 0, 1, 1, 0, 1, 1, 0, 0, 0,
                      1, 0, 1, 1, 0, 1, 0, 1, 0, 0]

# Predict labels for new reviews
predicted_labels = svm_classifier.predict(new_reviews_tfidf)

# Calculate accuracy
accuracy = accuracy_score(new_reviews_labels, predicted_labels)

# Interpret the predictions
actionable_labels = {0: 'Not Actionable', 1: 'Actionable'}
predicted_labels = [actionable_labels[pred] for pred in predicted_labels]

# Display the predictions and accuracy
for review, true_label, pred_label in zip(new_reviews, new_reviews_labels, predicted_labels):
    print(f"Review: {review}\nTrue Label: {actionable_labels[true_label]}\nPredicted Label: {pred_label}\n")

print(f"Accuracy: {accuracy * 100:.2f}%")
