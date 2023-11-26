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
    "The app is worth the price. It has all the features I need for my tasks.",
    
    "This app is incredibly useful. It helps me stay organized and productive.",
    "The app interface is intuitive and user-friendly. I love the design!",
    "I encountered a minor bug, but the customer support was quick to address and resolve it.",
    "The latest update brought some exciting features. I'm impressed with the improvements.",
    "The app crashed once, but overall, it's been a reliable tool for my daily tasks.",
    "I appreciate the affordable pricing. This app offers great value for the money.",
    "The app's dark mode is a game-changer. It's easy on the eyes during nighttime use.",
    "I recommend this app to anyone looking for a reliable and feature-packed solution.",
    "The app's security measures make me feel confident about using it for sensitive tasks.",
    "I've been using this app for months, and it has become an essential part of my routine.",
    
    "The app's user interface could use some improvement. It feels a bit outdated.",
    "I wish there were more customization options for the app's appearance.",
    "The app occasionally lags, especially when dealing with large files. Needs optimization.",
    "There is a security vulnerability that the developers need to address promptly.",
    "I encountered a bug that caused some data loss. Implementing better error handling is crucial.",
    "The app's performance on my device improved significantly after the recent update.",
    "The app's cost is justified by the quality and variety of features it offers.",
    "I like the simplicity of the app, but it could benefit from additional functionalities.",
    "The app's design is clean and modern. It stands out among other similar applications.",
    "The app crashes on startup, making it frustrating to use. Immediate attention is needed.",
    
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
    
    "This app has a sleek design, but it lacks essential features. Consider adding more functionalities.",
    "I'm impressed with the app's intuitive user interface. It makes navigation a breeze.",
    "The app crashes frequently, making it frustrating to use. Please fix this issue promptly.",
    "I love the simplicity of this app. It does what it's supposed to without unnecessary complications.",
    "There's a security flaw in the app. Strengthen the encryption to ensure user data is protected.",
    "The app is a bit pricey, but the premium features justify the cost.",
    "I encountered a bug that caused data duplication. Implement a fix to prevent this issue.",
    "The app's customer support is unresponsive. Improve communication for a better user experience.",
    "I use this app occasionally, and it meets my needs effectively.",
    "The app's performance is inconsistent. It needs optimization for smoother usage.",
    
    "The app crashes on startup. This issue needs immediate attention.",
    "This app is a must-have for productivity. It simplifies my workflow seamlessly.",
    "Decent app overall, but the customer support is lacking in responsiveness.",
    "I encountered a security vulnerability. Implement stronger measures for user data protection.",
    "The latest update introduced new features, enhancing the overall functionality.",
    "The app's interface is confusing. Consider a redesign for better user experience.",
    "I regret purchasing this app. It doesn't live up to its advertised capabilities.",
    "The app's speed is impressive. It responds quickly to user commands.",
    "The subscription cost is too high for the limited features provided.",
    "I recommend this app to everyone. It's user-friendly and feature-packed.",
    
    "The app frequently freezes, causing a frustrating user experience. Optimize for better performance.",
    "I encountered a bug that caused app crashes. Urgent attention is needed to resolve this issue.",
    "This app is a game-changer. It offers unique features that set it apart from other applications.",
    "The UI design is visually appealing, but the lack of customization options is a drawback.",
    "The app's cost is justified by its robust security features. I feel confident using it for sensitive tasks.",
    "The app is user-friendly, but occasional lags hinder the overall experience. Optimize for smoother usage.",
    "I've been a loyal user for years. The app consistently meets my expectations and needs.",
    "This app is not worth the price. It lacks essential features and functionality.",
    "The latest update brought some exciting changes, but it also introduced new bugs. Please fix promptly.",
    "The app's customer support is excellent. Quick responses and effective solutions to user concerns.",
    
    "The app's recent update significantly improved its functionality. Kudos to the development team!",
    "This app has become an indispensable part of my daily routine. Highly recommended.",
    "The app's interface is confusing, and the lack of a tutorial makes it challenging for new users.",
    "I encountered a security flaw while using this app. Strengthen encryption to protect user data.",
    "The app crashes randomly, disrupting the user experience. Investigate and fix this issue urgently.",
    "I appreciate the regular updates. It shows the developers are committed to enhancing the app.",
    "This app is overpriced for the features it offers. Consider adjusting the pricing structure.",
    "The user interface is outdated and could use a modern redesign for a more polished look.",
    "The app's speed is impressive, providing a seamless and responsive user experience.",
    "I've had no issues with this app. It's reliable, and the simplicity is perfect for my needs.",
    
    "The app's recent updates have introduced more bugs. It needs thorough testing before releases.",
    "I've encountered a few glitches, but overall, the app is reliable for my daily tasks.",
    "The app's design is sleek, but it lacks a dark mode option. Consider adding this feature.",
    "I've had issues with data synchronization across devices. Improve data syncing functionality.",
    "This app is a waste of time. It lacks basic features and doesn't fulfill its promises.",
    "The app's customer support is outstanding. They respond promptly and resolve issues efficiently.",
    "The subscription cost is justified by the app's premium features. It's worth the investment.",
    "I use this app for work, and it has significantly boosted my productivity. Highly recommended.",
    "The app's security features are top-notch. I feel confident storing sensitive information.",
    "The app frequently crashes, especially when multitasking. Address this issue for better stability.",
    
    "The app crashes frequently and is very frustrating.",
    "The speed of this app is awesome!",
    "The app frequently loses data. Implement a reliable auto-save feature to prevent data loss.",
    "The app takes too long to load. Optimize the speed for better user experience.",
    "The latest update improved the app's performance significantly. Great job!",
    "The app frequently crashes, making it frustrating to use. Please fix this issue.",
    "The latest update made the app slower on my device. Improve the performance.",
    "The app is reliable, and I've had no issues with it so far.",
    "The app's design is outdated. It needs a modern and fresh look.",
    "This app is amazing. I love it!",
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
          1, 0, 1, 1, 0, 1, 0, 1, 0, 0,
          0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
          1, 1, 0, 1, 1, 0, 0, 1, 0, 1,
          1, 1, 0, 1, 0, 1, 1, 1, 0, 0,
          1, 1, 1, 0, 1, 0, 1, 0, 0, 1,
          1, 0, 0, 1, 1, 1, 0, 1, 0, 0,
          1, 1, 0, 1, 1, 1, 0, 1, 1, 0,
          1, 0, 1, 1, 1, 0, 0, 1, 1, 0,
          1, 0, 1, 1, 0, 0, 0, 0, 1, 1,
          1, 0, 1, 1, 0, 1, 1, 0, 1, 0]  

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
    "The app is worth the price. It has all the features I need for my tasks.",
    
    "The app constantly crashes. Fix this issue immediately!",
    "I love the new design, but the speed could be improved.",
    "Great app, but it would be better with a dark mode option.",
    "The app's performance is excellent. No issues so far.",
    "The UI is confusing. Consider simplifying the design.",
    "This app is perfect for my needs. No complaints!",
    "I wish there were more filters available. Consider adding new ones.",
    "The speed is impressive, but the UI needs a refresh.",
    "The app frequently loses data. Implement an auto-save feature.",
    "I regret purchasing this app. It doesn't meet my expectations.",
    
    "The latest update made the app slower. Please optimize the performance.",
    "Decent app, but it could use some additional features.",
    "The design is outdated. Consider a more modern look.",
    "The app's speed is a bit slow. Could use some optimization.",
    "The UI is user-friendly. Great job on the design!",
    "I like the simple design, but the app crashes too often.",
    "The latest update improved performance significantly. Good work!",
    "The app's design is sleek. I appreciate the modern look.",
    "I use this app daily, and it works very well for me.",
    "This app is amazing. I love the speed and design!",
    
    "The app crashes occasionally. Please address this issue.",
    "The UI is clunky. It needs a more intuitive interface.",
    "I regret purchasing this app. It's not worth the money.",
    "The speed is impressive, but the design needs improvement.",
    "The app's design is fantastic. Very modern and fresh.",
    "The app takes too long to load. Optimize the speed.",
    "The latest update made the app more reliable. Thank you!",
    "This app is perfect for my needs. No complaints at all.",
    "The UI is outdated. Consider a more modern design.",
    "The app crashes frequently. This needs urgent attention.",
    
    "I love the simple design. No issues with speed.",
    "The app's performance is inconsistent. Needs improvement.",
    "Decent app, but it lacks some essential features.",
    "The design is sleek and modern. Great user experience!",
    "The app crashes occasionally. Please fix this issue.",
    "I wish the app had a dark mode. Consider adding it.",
    "The speed is impressive, but the UI needs refinement.",
    "The latest update improved the app's reliability. Good job!",
    "The app takes too long to load. Optimize the speed.",
    "The design is outdated. Consider a more modern look.",
    
    "The speed is impressive, but the app crashes too often.",
    "I regret purchasing this app. It's not worth the money.",
    "The app's design is fantastic. Very modern and fresh.",
    "The app crashes occasionally. Please address this issue.",
    "The UI is clunky. It needs a more intuitive interface.",
    "This app is amazing. I love the speed and design!",
    "The latest update made the app more reliable. Thank you!",
    "The app's design is sleek. I appreciate the modern look.",
    "I use this app daily, and it works very well for me.",
    "The UI is user-friendly. Great job on the design!",   
]

# Different labels for new reviews
new_reviews_labels = [0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 
                      0, 0, 0, 1, 0, 1, 0, 1, 0, 1,
                      1, 1, 0, 1, 1, 1, 0, 1, 0, 0,
                      1, 0, 1, 1, 0, 1, 1, 0, 0, 0,
                      1, 0, 1, 1, 0, 1, 0, 1, 0, 0,
                      1, 1, 1, 0, 1, 0, 1, 1, 1, 0,
                      1, 1, 1, 1, 1, 1, 0, 1, 0, 0,
                      1, 1, 0, 1, 0, 1, 0, 0, 1, 1,
                      0, 1, 1, 0, 1, 1, 1, 0, 1, 1,
                      1, 0, 0, 1, 1, 0, 0, 0, 0, 0]


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

# Predict labels for new reviews
predicted_labels = svm_classifier.predict(new_reviews_tfidf)

# Calculate accuracy
accuracy = accuracy_score(new_reviews_labels, predicted_labels)

# Interpret the predictions
actionable_labels = {0: 'Not Actionable', 1: 'Actionable'}
predicted_labels = [actionable_labels[pred] for pred in predicted_labels]

# Display the predictions and count the number of actionable reviews
num_actionable_reviews = 0
for review, label in zip(new_reviews, predicted_labels):
    print(f"Review: {review}\nPredicted Label: {label}\n")
    if label == 'Actionable':
        num_actionable_reviews += 1

# Output the number of actionable reviews
print(f"Number of Actionable Reviews: {num_actionable_reviews} out of {len(new_reviews)}")

print(f"Accuracy: {accuracy * 100:.2f}%")
