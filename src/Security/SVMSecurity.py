# Import necessary libraries
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Sample list of app store reviews and their corresponding labels (0 for not related to security, 1 for related to security)
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
    
    "The app's security features are robust, ensuring my data is safe.",
    "I'm concerned about the app's privacy policy. It collects too much personal information.",
    "The latest update introduced enhanced security measures. Great job!",
    "I appreciate the app's commitment to user privacy. No unnecessary data collection.",
    "Security is a top priority for me, and this app meets my expectations.",
    "I'm worried about the app's data integrity. Some files seem corrupted.",
    "The app's privacy settings are confusing. It's not clear what information is being shared.",
    "Data encryption is a must for me, and this app delivers on that front.",
    "The app respects user privacy, and I feel confident using it.",
    "I've noticed some security loopholes in the app. Developers, please address this.",
    
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
    
    "Privacy controls are excellent. I have full control over my personal information.",
    "The app's security protocols are outdated. Needs an urgent update.",
    "I'm impressed by the app's commitment to user data privacy.",
    "The app's security is questionable. I've encountered suspicious activities.",
    "Privacy settings are easy to navigate, giving users control over their data.",
    "Data integrity is crucial, and this app does a great job in ensuring it.",
    "I'm skeptical about the app's security. It seems vulnerable to breaches.",
    "The latest update prioritized user privacy. Kudos to the developers!",
    "I've experienced data leaks with this app. It's a serious concern.",
    "Security is lacking in this app. I don't feel comfortable using it anymore.",
    
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
    
    "The app's privacy policy is clear and transparent. I appreciate that.",
    "I'm concerned about the app's data retention policies. Too much information stored.",
    "Security measures have improved with the recent update. Much appreciated.",
    "The app's privacy controls need improvement. Too many default settings share data.",
    "My data has been compromised while using this app. Not secure at all.",
    "I appreciate the app's commitment to protecting user data. Feel safe using it.",
    "The app's security features are top-notch. No issues with data breaches.",
    "Privacy controls are lacking. I'm unsure about the safety of my data.",
    "The app's security settings are confusing. Developers, simplify them.",
    "I've noticed suspicious activity related to my data. Security is a major concern.",
    
    "The app respects user privacy, but it could use more security features.",
    "Data integrity is compromised in this app. Files often get corrupted.",
    "I appreciate the app's efforts to enhance security. Keep up the good work!",
    "Privacy settings are comprehensive, allowing users to customize data sharing.",
    "The app's security measures are outdated. It's time for an upgrade.",
    "I've encountered security issues with this app. It needs immediate attention.",
    "The app's commitment to user privacy is commendable. Thumbs up!",
    "I'm concerned about data leaks. The app needs tighter security measures.",
    "Security is a major concern. I've experienced unauthorized access to my account.",
    "The app's privacy settings are too restrictive. Allow more customization options.",
    
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
    
    "I've had a positive experience with the app's security features. No complaints.",
    "Data privacy is a priority for me, and this app meets my expectations.",
    "I'm worried about the app's data encryption. It seems inadequate.",
    "The app's commitment to user privacy is evident in its stringent security measures.",
    "Security vulnerabilities have been addressed in the latest update. Good job!",
    "Privacy controls are lacking, making it difficult to manage data sharing preferences.",
    "Data integrity issues persist. The app needs better error-checking mechanisms.",
    "I appreciate the app's dedication to user security. I trust it with my data.",
    "Security features are lacking. The app needs a comprehensive security audit.",
    "The app's privacy settings are user-friendly, ensuring a secure experience.",
    
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
    
    "The app's security is solid, providing peace of mind while using it.",
    "I've experienced unauthorized access to my data. The app's security needs improvement.",
    "Privacy controls are robust, allowing users to manage data sharing effectively.",
    "I'm concerned about the app's data encryption. It seems vulnerable.",
    "Security measures have been enhanced in the latest update. Great job!",
    "Data integrity issues persist. Frequent crashes result in lost data.",
    "The app's privacy features are excellent, ensuring user data is kept confidential.",
    "I appreciate the app's commitment to securing user data. No compromises.",
    "Security vulnerabilities have been addressed in the recent update. Much needed.",
    "The app's security settings are user-friendly, making it easy to customize preferences.",
    
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
    
    "Data integrity is compromised with frequent glitches. Files often get corrupted.",
    "The latest update has introduced comprehensive security measures. Well done!",
    "I'm skeptical about the app's data privacy. Some settings are unclear.",
    "Security controls need improvement. I've encountered issues with unauthorized access.",
    "Privacy settings are too restrictive. Users should have more control over data sharing.",
    "The app's security is questionable. Users are reporting unauthorized account access.",
    "I'm impressed by the app's commitment to user privacy. Keep up the good work!",
    "Data integrity is a major concern. Frequent crashes lead to lost information.",
    "The app prioritizes user privacy, but data encryption could be stronger.",
    "I've noticed security loopholes in the app. Developers, please address these concerns.",
    
    "The app's security features are outdated. It's time for a major upgrade.",
    "I've encountered unauthorized access to my data. The app's security needs attention.",
    "The latest update has improved data encryption, enhancing the overall security.",
    "Privacy settings are confusing. Users struggle to manage their data sharing preferences.",
    "I appreciate the app's commitment to user data privacy. Security measures are top-notch.",
    "Security concerns have been addressed in the latest update. Kudos to the developers!",
    "Data integrity is compromised with frequent crashes. Developers, please fix this issue.",
    "The app respects user privacy, but there are issues with data encryption. Needs improvement.",
    "I'm concerned about the app's data security. Some users have reported breaches.",
    "The latest update has introduced enhanced security features. It's a step in the right direction.",
    
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
]

labels = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
          0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
          0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

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
    "Great app overall, but it lacks a dark mode. Consider adding this feature.",
    "The app crashes frequently, and the speed is unbearable. Fix it!",
    "Fast and responsive. This app is a lifesaver.",
    "I don't notice any significant speed issues with this app.",
    "The timing of this app is perfect. It responds quickly to every command.",
    "The speed of this app needs improvement. It's too slow for my liking.",
    
    "Privacy controls are lacking. Users should have more say in data sharing preferences.",
    "I've noticed improvements in the app's security. It's becoming more reliable.",
    "Data integrity issues persist. Frequent crashes result in lost data and frustration.",
    "The app's commitment to user privacy is evident. Security features are well-implemented.",
    "I'm skeptical about the app's data security. There have been reports of unauthorized access.",
    "Security vulnerabilities have been addressed in the latest update. Users are more confident now.",
    "Privacy controls are too restrictive. Users need more options for managing data sharing.",
    "I've experienced unauthorized access to my data. Security measures need immediate attention.",
    "The app's commitment to user data privacy is commendable. Security features are robust.",
    "Data integrity is compromised with frequent glitches. Developers, please prioritize stability.",
    
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
    
    "Security measures have been enhanced in the latest update. Users appreciate the improvements.",
    "Privacy settings are user-friendly, allowing users to customize their data sharing preferences.",
    "The app's security features are outdated. Developers, it's time for a major overhaul.",
    "I'm concerned about the app's data encryption. It seems inadequate for protecting user information.",
    "The latest update has introduced comprehensive security measures. It's a positive step forward.",
    "Privacy controls are confusing. Users struggle to navigate the settings for data sharing preferences.",
    "I appreciate the app's commitment to user privacy. Security measures are a top priority.",
    "Data integrity issues persist. Frequent crashes result in lost data and inconvenience.",
    "The app's security features are top-notch. I feel confident storing sensitive information.",
    "The app frequently crashes, especially when multitasking. Address this issue for better stability."   
]

# Different labels for new reviews
new_reviews_labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 
                      0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                      1, 1, 1, 1, 1, 1, 1, 1, 1, 0]

# Convert new review text to TF-IDF features
new_reviews_tfidf = tfidf_vectorizer.transform(new_reviews)

# Predict labels for new reviews
predicted_labels = svm_classifier.predict(new_reviews_tfidf)

# Display the predictions
for review, label in zip(new_reviews, predicted_labels):
    print(f"Review: {review}\nPredicted Label: {label}\n")
    
# Predict labels for new reviews
predicted_labels = svm_classifier.predict(new_reviews_tfidf)

# Calculate accuracy
accuracy = accuracy_score(new_reviews_labels, predicted_labels)

# Interpret the predictions
security_labels = {0: 'Not Related to Security', 1: 'Related to Security'}
predicted_labels = [security_labels[pred] for pred in predicted_labels]

# Display the predictions and count the number of security related reviews
num_security_reviews = 0
for review, label in zip(new_reviews, predicted_labels):
    print(f"Review: {review}\nPredicted Label: {label}\n")
    if label == 'Related to Security':
        num_security_reviews += 1

# Output the number of security reviews
print(f"Number of Security Reviews: {num_security_reviews} out of {len(new_reviews)}")

print(f"Accuracy: {accuracy * 100:.2f}%")