# Import necessary libraries
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Sample list of app store reviews and their corresponding labels (0 for not related to speed, 1 for related to speed)
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
    
    "The app crashes frequently and is very frustrating.",
    "No crashes so far. The app works smoothly.",
    "The latest update introduced crashes on my device. Please fix!",
    "Smooth experience without any crashes. Great job!",
    "I love this app, but it crashes occasionally. Needs improvement.",
    "No crashes since the last update. Keep up the good work!",
    "The app frequently loses data due to unexpected crashes.",
    "Haven't experienced any crashes. The app is reliable.",
    "The latest version is a disaster. Constant crashes and freezes.",
    "The app is stable, and I haven't encountered any crashes.",
    
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
    
    "Crashes ruin the user experience. Please address this issue.",
    "I've had to restart the app multiple times due to crashes.",
    "No crashes on my end. The app is running smoothly.",
    "Crashes occur when switching between features. Annoying!",
    "The app is crash-free on my device. Works like a charm.",
    "The recent crashes are frustrating. Fix the stability, please!",
    "I can't rely on this app anymore. Too many unexpected crashes.",
    "Smooth performance without any crashes or glitches.",
    "The constant crashes make this app unusable. Disappointed.",
    "So far, no crashes. The app is performing well.",
    
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
    
    "Crashes every time I try to open a specific feature. Annoying!",
    "The app crashes randomly, impacting my workflow.",
    "No crashes in the latest version. Happy with the stability.",
    "Crashes have become more frequent with each update. Not good.",
    "The app crashed during a crucial moment. Very frustrating.",
    "Stable performance with no crashes. I'm satisfied.",
    "Crashes on startup. Can't even use the app anymore.",
    "The latest update fixed the crashes. Back to smooth operation.",
    "Frequent crashes are making me consider switching to alternatives.",
    "I haven't experienced any crashes. The app is reliable for me.",
    
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
    
    "The app's user interface is outdated. Consider a modern redesign for better usability.",
    "I encountered a bug causing data loss. Please implement better error handling.",
    "This app is a lifesaver! It has streamlined my daily tasks.",
    "The latest update made the app slower on my device. Improve the performance.",
    "The app's dark mode is a game-changer. It's easy on the eyes during nighttime use.",
    "I wish there were more customization options for the app's appearance.",
    "The app occasionally lags, especially when dealing with large files. Needs optimization.",
    "There is a security vulnerability that the developers need to address promptly.",
    "I appreciate the affordable pricing. This app offers great value for the money.",
    "The app's performance on my device improved significantly after the recent update.",
    
    "Crashes persist despite multiple updates. Disappointing.",
    "The app crashed and wiped my data. Extremely frustrating!",
    "No crashes in the recent updates. The app is running well.",
    "Crashes occur when opening the camera. Annoying bug.",
    "The app is crash-prone. Hope the developers address this soon.",
    "Stable performance without any crashes. I'm satisfied.",
    "The app crashed during a video call. Not impressed.",
    "No crashes on my end. The app is performing well.",
    "Frequent crashes make it impossible to enjoy the app.",
    "Haven't experienced any crashes. The app is reliable.",   
]

# Labels (0 for not related to crashes, 1 for related to crashes)
labels = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

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
    
    "The app crashes frequently. Please address this issue promptly.",
    "This app is a game-changer! It's incredibly useful and efficient.",
    "The user interface needs improvement. It's not very intuitive.",
    "I encountered a security vulnerability in the latest version. Urgent fix required.",
    "The latest update significantly improved the app's performance. Great job!",
    "The app lacks some essential features. Consider adding them in the next update.",
    "I love the sleek design of the app. It's visually appealing.",
    "There's a bug causing data loss. Please implement better error handling.",
    "Customer support is unresponsive. Improve the service for better user experience.",
    "The app is worth the price. It has all the features I need for my tasks.",
    
    "Crashes have been resolved in the latest update. Great job!",
    "The app is crash-free on my device. Works like a charm.",
    "Crashes have become more frequent with each update. Not good.",
    "I love the features, but the crashes are unbearable. Please fix.",
    "No crashes since the last update. Keep up the good work!",
    "The app crashes frequently and is very frustrating.",
    "The latest version is a disaster. Constant crashes and freezes.",
    "No crashes in the recent updates. The app is running well.",
    "Crashes persist despite multiple updates. Disappointing.",
    "The app crashed during a crucial moment. Very frustrating.",
    
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
    
    "The app's recent updates have introduced more bugs. It needs thorough testing before releases.",
    "I've encountered a few glitches, but overall, the app is reliable for my daily tasks.",
    "The app's design is sleek, but it lacks a dark mode option. Consider adding this feature.",
    "I've had issues with data synchronization across devices. Improve data syncing functionality.",
    "This app is a waste of time. It lacks basic features and doesn't fulfill its promises.",
    "The app's customer support is outstanding. They respond promptly and resolve issues efficiently.",
    "The subscription cost is justified by the app's premium features. It's worth the investment.",
    "I use this app for work, and it has significantly boosted my productivity. Highly recommended.",
    "The app's security features are top-notch. I feel confident storing sensitive information.",
    "The app frequently crashes, especially when multitasking. Address this issue for better stability."
]

# Different labels for new reviews (adjust as needed)
new_reviews_labels = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                      1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

# Convert new review text to TF-IDF features
new_reviews_tfidf = tfidf_vectorizer.transform(new_reviews)

# Predict labels for new reviews
predicted_labels = svm_classifier.predict(new_reviews_tfidf)

# Interpret the predictions
crash_labels = {0: 'Not Related to Crashes', 1: 'Related to Crashes'}
predicted_labels = [crash_labels[pred] for pred in predicted_labels]

# Display the predictions
for review, label in zip(new_reviews, predicted_labels):
    print(f"Review: {review}\nPredicted Label: {label}\n")
    
# Predict labels for new reviews
predicted_labels = svm_classifier.predict(new_reviews_tfidf)

# Calculate accuracy
accuracy = accuracy_score(new_reviews_labels, predicted_labels)

# Display the predictions and count the number of actionable reviews
num_crash_reviews = 0
for review, label in zip(new_reviews, predicted_labels):
    print(f"Review: {review}\nPredicted Label: {label}\n")
    
    # This is not working - not sure why need to investigate.
    if label == 'Related to Crashes':
        num_crash_reviews += 1

# Output the number of actionable reviews
print(f"Number of Actionable Reviews: {num_crash_reviews} out of {len(new_reviews)}")

print(f"Accuracy: {accuracy * 100:.2f}%")