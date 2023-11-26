# Necessary scikit-learn modules for SVM implementation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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
    "This app is perfect for my needs. It has everything I was looking for.",
    
    "The app responds quickly to my commands. Love the speed!",
    "I can't believe how fast this app is. It's a game-changer!",
    "The speed of this app is unparalleled. Impressive work!",
    "This app is lightning-fast. Makes my tasks so much easier.",
    "The app loads in a blink of an eye. Superb performance!",
    "I wish the app would run faster. It's a bit sluggish for my liking.",
    "The speed is the only thing keeping this app from being perfect.",
    "Despite the sleek design, the app's speed is disappointing.",
    "The app freezes sometimes, affecting its overall speed. Please fix.",
    "The speed improvements in the latest update are noticeable. Thank you!",
    
    "Just another average app. Nothing special.",
    "I don't see what the hype is about. This app is mediocre at best.",
    "An okay app, but there are better options out there.",
    "This app is so-so. Nothing remarkable, but it gets the job done.",
    "I expected more from this app. The speed is nothing to write home about.",
    "The UI is confusing, and the app is slow. Not a great combination.",
    "The app is overrated. I don't understand the positive reviews.",
    "I uninstalled this app after a day. It's not worth the download.",
    "Not impressed with the app's speed. It needs a major upgrade.",
    "The app lacks features and is slower than similar options in the market.",
    
    "I've been using this app for years, and it's still my favorite!",
    "This app is a hidden gem. Highly recommended!",
    "The simplicity of this app is its strength. No unnecessary frills.",
    "I adore this app. It simplifies my life in so many ways.",
    "The best app I've ever downloaded. Fast, reliable, and user-friendly.",
    "I wasted money on this app. It's slow and full of bugs.",
    "The app's interface is outdated, and it needs a speed boost.",
    "I don't understand the positive reviews. This app is subpar.",
    "The customer support for this app is a nightmare. Slow and unhelpful.",
    "This app falls short of expectations. The speed is a major letdown.",
    
    "The app's speed is unmatched. I'm impressed!",
    "This app needs optimization. It's too slow for my liking.",
    "Fast and efficient! This app is a time-saver.",
    "The speed improvements in the latest version are noticeable.",
    "I appreciate the app's speed, but the interface could use some work.",
    "The app crashed once, but the speed is excellent otherwise.",
    "The loading time for this app is unacceptable. Improve it, please.",
    "This app is a speed demon! I love how responsive it is.",
    "The speed is the best feature of this app. Everything else is average.",
    "I can't recommend this app enough. The speed sets it apart.",
    
    "I don't understand the hype around this app. It's slow and unimpressive.",
    "This app is a game-changer. The speed makes it a joy to use.",
    "I've never experienced such slow performance in an app before. Disappointing.",
    "The app's speed is decent, but the design is lacking.",
    "Fast and reliable! This app is a must-have.",
    "The app crashes frequently, and the speed is nothing special.",
    "The latest update ruined the app's speed. Very frustrating.",
    "I love the simplicity of this app, but the speed could be better.",
    "This app is a gem. Fast, efficient, and user-friendly.",
    "I uninstalled this app immediately. The speed is unbearable.",
    
    "The app's fast response time is its standout feature. Impressive!",
    "I expected better timing from this app. It's slower than I anticipated.",
    "Fast and efficient! The speed of this app exceeded my expectations.",
    "Timing is crucial, and this app delivers with lightning-fast speed.",
    "I'm frustrated with the slow timing of this app. It needs improvement.",
    "The app's loading speed is phenomenal. Makes my tasks a breeze.",
    "The slow response time is a major drawback of this otherwise good app.",
    "Timing matters, and this app's speed is on point. Well done!",
    "The app's slow performance is a dealbreaker. It needs a speed boost.",
    "Fast, reliable, and user-friendly. This app checks all the boxes for me.",
    
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
    
    "The app's speed is fantastic, but the interface is confusing.",
    "This app is slow, and the design is outdated. Not worth it.",
    "Fast and reliable. The app's speed makes it a pleasure to use.",
    "I love the app's design, but the timing could be improved.",
    "The speed of this app is terrible. It takes forever to load.",
    "The app is slow, and the latest update didn't help much.",
    "Timing matters, and this app gets it right. Fast and efficient.",
    "The app's speed is decent, but it lacks some essential features.",
    "I expected better timing from this app. It's disappointingly slow.",
    "Fast loading times and a sleek design. This app is a winner!",
    
    "I don't understand the positive reviews. This app is slow and clunky.",
    "The app crashes on startup. This issue needs immediate attention.",
    "This app is a must-have for productivity. It simplifies my workflow seamlessly.",
    "Decent app overall, but the customer support is lacking in responsiveness.",
    "I encountered a security vulnerability. Implement stronger measures for user data protection.",
    "The latest update introduced new features, enhancing the overall functionality.",
    "The app's interface is confusing. Consider a redesign for better user experience.",
    "I regret purchasing this app. It doesn't live up to its advertised capabilities.",
    "The app's speed is impressive. It responds quickly to user commands.",
    "The subscription cost is too high for the limited features provided.",
    
    "The app's speed is top-notch, no lag whatsoever!",
    "This app has a noticeable lag, especially during heavy usage.",
    "Fast and efficient, with no signs of lag. A great experience!",
    "I experienced some lag with this app. Please optimize for smoother performance.",
    "The timing of this app is perfect. No lag, no issues.",
    "The app's design is sleek, but the occasional lag is frustrating.",
    "Lag ruins the user experience. The speed needs serious improvement.",
    "Impressed by the app's speed; there's no lag in sight!",
    "The app suffers from severe lag. It's affecting my productivity.",
    "Fast loading times, minimal lag – this app is a winner!",
    
    "The app's speed is inconsistent. Sometimes smooth, sometimes laggy."
    "The app's speed is impressive, providing a seamless experience.",
    "I encountered some delays with this app. It needs optimization.",
    "Fast and efficient, making tasks a breeze with quick response times.",
    "The app's timing is impeccable, offering a smooth user experience.",
    "I appreciate the app's speed, but the design could be more modern.",
    "Experienced a slowdown in performance, especially during heavy use.",
    "The app is responsive and quick, but occasional hiccups need fixing.",
    "The timing of this app is great, with no noticeable delays.",
    "The app's speed is inconsistent. Sometimes fast, sometimes slower.",
    "This app combines speed and functionality, meeting all my needs.",
    
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
    
    "The app crashes frequently and is very frustrating.",
    "The speed of this app is awesome!",
    "The speed of this app is way too slow.",
    "The app frequently loses data. Implement a reliable auto-save feature to prevent data loss.",
    "The app takes too long to load. Optimize the speed for better user experience.",
    "The latest update improved the app's performance significantly. Great job!",
    "The app frequently crashes, making it frustrating to use. Please fix this issue.",
    "The latest update made the app slower on my device. Improve the performance.",
    "The app is reliable, and I've had no issues with it so far.",
    "The app's design is outdated. It needs a modern and fresh look."   
]

# Labels (0 for not related to speed, 1 for related to speed)
labels = [0, 1, 1, 0, 1, 0, 0, 1, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          0, 0, 0, 0, 1, 1, 0, 0, 1, 1,
          0, 0, 0, 0, 0, 1, 1, 0, 0, 1,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
          1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 0, 1, 1, 0, 0, 1, 1, 0, 1,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
          0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
          0, 1, 1, 0, 1, 0, 0, 1, 0, 0]

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
    
    "I appreciate the app's speed, but the design is lackluster.",
    "This app is mediocre. The speed and features are just average.",
    "The app's speed is its strongest suit. Well done!",
    "Timing matters, and this app gets it right. No complaints here.",
    "The app's performance is inconsistent. Sometimes fast, sometimes slow.",
    "The app's speed is impressive, but it lacks essential features.",
    "I love the sleek design of this app, but the timing could be better.",
    "The speed of this app is unacceptable. It needs urgent improvement.",
    "Fast loading times make this app a winner in my book.",
    "The app's speed is decent, but the frequent crashes are a major issue.",
    
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
    "The UI is user-friendly. Great job on the design!"
]

# Different labels for new reviews
new_reviews_labels = [1, 1, 0, 0, 0, 1, 1, 1, 1, 1,
                      1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                      0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
                      0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 1, 0, 0, 0, 0, 0, 1, 0, 0,
                      1, 0, 0, 1, 0, 0, 0, 0, 0, 1,
                      0, 0, 0, 1, 0, 1, 0, 0, 0, 0,
                      1, 0, 0, 0, 0, 0, 1, 0, 1, 0,
                      1, 0, 0, 0, 0, 1, 0, 0, 0, 0]

# Convert new review text to TF-IDF features
new_reviews_tfidf = tfidf_vectorizer.transform(new_reviews)

# Predict labels for new reviews
predicted_labels = svm_classifier.predict(new_reviews_tfidf)

# Calculate accuracy
accuracy = accuracy_score(new_reviews_labels, predicted_labels)

# Interpret the predictions
speed_labels = {0: 'Not Related to Speed', 1: 'Related to Speed'}
predicted_labels = [speed_labels[pred] for pred in predicted_labels]

# Display the predictions and count the number of speed related reviews
num_speed_reviews = 0
for review, label in zip(new_reviews, predicted_labels):
    print(f"Review: {review}\nPredicted Label: {label}\n")
    if label == 'Related to Speed':
        num_speed_reviews += 1

# Output the number of speed reviews
print(f"Number of Speed Reviews: {num_speed_reviews} out of {len(new_reviews)}")

print(f"Accuracy: {accuracy * 100:.2f}%")