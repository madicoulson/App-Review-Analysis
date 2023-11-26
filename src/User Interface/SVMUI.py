# Necessary scikit-learn modules for SVM implementation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Sample list of app store reviews and their corresponding labels (0 for not related to UI, 1 for related to UI)
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
    
    "The app has a confusing user interface. I can't find basic features easily.",
    "Beautiful design, but the navigation is clunky. Could use some improvements.",
    "The latest update ruined the app's design. It's too cluttered now.",
    "Intuitive UI with a sleek design. Love using this app!",
    "Design issues make it hard to use. Buttons are not where you expect them.",
    "The app looks great, but the user experience is lacking. It's not user-friendly.",
    "The design is outdated. It needs a modern and fresh look.",
    "UI is smooth, but the color scheme is too harsh on the eyes.",
    "I love the simplicity of the design, but some features are hard to find.",
    "The UI is confusing, and the design choices are questionable.",
    
    "Smooth user experience with a clean and modern design. No complaints.",
    "The app's UI is frustrating. It's not clear how to perform basic actions.",
    "Visually appealing design, but the app crashes frequently. Needs improvement.",
    "The recent design changes are excellent. The app feels more user-friendly.",
    "The UI is cluttered, and the overall design is not intuitive.",
    "The app is beautiful, but there are usability issues. Improve the navigation.",
    "The design is simple and elegant. One of the best-looking apps I've used.",
    "The UI is too busy. There's too much going on, and it's overwhelming.",
    "I like the sleek design, but the app is slow to load.",
    "User-friendly design with easy navigation. No complaints so far.",
    
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
    
    "The latest update improved the app's performance significantly. Great job!",
    "The design is user-friendly, but the app lacks some essential features.",
    "Confusing UI. It's hard to figure out how to customize settings.",
    "The app's design is top-notch. I enjoy using it every day.",
    "Too many ads ruin the clean design. It's distracting.",
    "The UI is intuitive, but the design is too bland. Needs more creativity.",
    "The design is modern and visually appealing. No complaints from me.",
    "The UI is a bit outdated, but the overall experience is decent.",
    "The design is clunky, and the app often freezes. Frustrating experience.",
    "Sleek and modern design, but the app crashes frequently.",
    
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
    
    "The UI is confusing, and the design needs a refresh. Not impressed.",
    "Intuitive design with smooth navigation. I love the user experience.",
    "The app is visually stunning, but there are some design inconsistencies.",
    "The UI is too complicated. It takes too many steps to perform simple tasks.",
    "I like the minimalist design, but the app is slow to respond.",
    "The latest update introduced design changes that I don't like. Revert, please!",
    "The UI is straightforward, but the design feels outdated.",
    "The design is user-friendly, but the app occasionally crashes.",
    "The UI is confusing, and the design choices are not user-friendly.",
    "Beautiful design, but the app lacks essential features.",
    
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
    
    "The UI is sleek, but there are frequent glitches in the design.",
    "I love the design, but the app crashes too often. Please fix!",
    "The design is cluttered, and it's hard to find what I need.",
    "The UI is clean and simple, but the app is slow to load.",
    "The app's design is confusing. It's not easy to navigate.",
    "The design is visually appealing, but there are functionality issues.",
    "The UI is outdated, and the design needs a modern touch.",
    "The app's design is fantastic, but it's slow to respond at times.",
    "The design is user-friendly, but the app has performance issues.",
    "Intuitive UI, but the design is a bit plain. Could use some enhancements.",
    
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
    
    "The dark mode is excellent, but the light mode is too bright. Provide better customization options.",
    "The app lacks a dark mode, making it uncomfortable to use at night. Please add this feature.",
    "Switching between light and dark modes is seamless. I love the design choices.",
    "The light mode is easy on the eyes, but the dark mode has some visibility issues. Improve contrast.",
    "I appreciate the option for both light and dark modes. It caters to different user preferences.",
    "The dark mode is aesthetically pleasing, but the light mode feels outdated. Consider a redesign.",
    "The light mode is perfect for daytime use, but the dark mode needs improvement. Too harsh.",
    "Both light and dark modes are well-implemented, enhancing the overall user experience.",
    "The app's dark mode is fantastic, but the light mode lacks finesse. Work on consistency.",
    "The app looks great in both light and dark modes. Smooth transitions between them.",
    
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
    
    "The dark mode is a game-changer, but the light mode needs refinement for better readability.",
    "Light mode is user-friendly, but the dark mode is too dim. Adjust brightness levels.",
    "The app's design is modern, and both light and dark modes complement the overall aesthetic.",
    "The dark mode enhances the app's visual appeal. I wish more apps had such seamless transitions.",
    "I love the dark mode, but the light mode is too plain. Add more customization options.",
    "The option to toggle between light and dark modes is appreciated. Great for different settings.",
    "Both light and dark modes are well-executed, contributing to a positive user experience.",
    "The light mode is easy on the eyes, but the dark mode could use a bit more contrast.",
    "The dark mode is a welcome addition, but some elements in the light mode feel outdated.",
    "The app's design is excellent, and the dark mode is my preferred choice for night use.",
    
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
    
    "I love the variety of filters available. It adds so much fun to my photos!",
    "The new filters in the latest update are fantastic. Great job!",
    "Filters are limited. Consider adding more options for users.",
    "The app crashes when applying certain filters. Fix this issue!",
    "The vintage filters are my favorite. Keep adding unique ones!",
    "Filters make my photos look amazing. No complaints so far.",
    "The app freezes when using the augmented reality filters. Annoying bug.",
    "The black and white filters need improvement. They look too dull.",
    "Filters are a bit overwhelming. It would be nice to have a simpler selection.",
    "I wish there were more creative filters. Current options are a bit plain.",
    
]

labels = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
          0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
          0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
          0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
          0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
          0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
          0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 0, 0, 0, 0, 0, 0, 1, 0,
          1, 1, 1, 1, 0, 1, 1, 1, 1, 1]


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
    
    "The app's design is exceptional, and the dark mode complements the overall aesthetic.",
    "I prefer using the app in dark mode, but the light mode is equally well-designed.",
    "Switching between light and dark modes is smooth, contributing to a positive user experience.",
    "The light mode is too plain, but the dark mode has a stylish appeal. Consistency is key.",
    "Both light and dark modes are well-executed, providing users with a visually pleasing experience.",
    "The dark mode enhances the app's visual appeal, making it suitable for nighttime use.",
    "The light mode is user-friendly, but the dark mode lacks some features. Ensure consistency.",
    "Switching between light and dark modes is effortless, offering users flexibility and comfort.",
    "The dark mode is excellent, but the light mode feels outdated. Consider a design overhaul.",
    "Both light and dark modes are visually appealing, contributing to an enjoyable user experience.",
    
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
    
    "The customer support for this app is responsive and helpful.",
    "The latest update improved the app's performance significantly.",
    "This app is a bit expensive for the features it offers.",
    "The app is user-friendly, but it lacks certain functionalities.",
    "I regret purchasing this app. It doesn't meet my expectations.",
    "I appreciate the inclusion of both light and dark modes. It caters to different user preferences.",
    "The dark mode is sleek and modern, while the light mode lacks the same level of refinement.",
    "Switching between light and dark modes is seamless, enhancing the app's usability.",
    "The light mode is user-friendly, but the dark mode needs some adjustments for better readability.",
    "Both light and dark modes contribute to a positive user experience, providing versatility.",
    
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
new_reviews_labels = [0, 0, 1, 0, 0, 1, 0, 1, 1, 0,
                      1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                      0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                      0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
                      0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                      0, 1, 1, 0, 1, 0, 1, 0, 0, 0,
                      0, 0, 1, 0, 1, 1, 0, 1, 0, 1,
                      0, 1, 0, 0, 1, 0, 0, 0, 1, 0,
                      1, 0, 0, 1, 0, 1, 1, 0, 0, 1,
                      0, 0, 1, 0, 1, 1, 0, 1, 0, 1]

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
UI_labels = {0: 'Not Related to UI', 1: 'Related to UI'}
predicted_labels = [UI_labels[pred] for pred in predicted_labels]

# Display the predictions and count the number of UI reviews
num_UI_reviews = 0
for review, label in zip(new_reviews, predicted_labels):
    print(f"Review: {review}\nPredicted Label: {label}\n")
    if label == 'Related to UI':
        num_UI_reviews += 1

# Output the number of UI reviews
print(f"Number of UI Reviews: {num_UI_reviews} out of {len(new_reviews)}")

print(f"Accuracy: {accuracy * 100:.2f}%")