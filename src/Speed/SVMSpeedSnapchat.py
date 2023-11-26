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

snapchat_reviews = [
    "My AI is so annoying. I've clicked on it by accident many times. There should be a free option to get rid of it. I don't think it's fair that iPhone users get dark mode for free, but Android users have to pay for premium to have the option. Go back to the simplicity. I've had Snap since 2013. It's a great way to send photos & videos to friends & family & keep in touch. The Discover Page & TikTok pages are also annoying. If I wanna watch TikToks I'll go to TikTok where the videos are clear.",
    "Whatever recent updates have been made have broken this app. I barely ever get notification sounds any longer, nor does it activate my screen. I haven't changed any settings. I suddenly just don't get notifications for some reason, so I keep missing messages from people. And then some messages are coming in super late. Never had these issues before.",
    "This app is super buggy. Every time I add continuous video it either sends the cut clips in the wrong order or duplicates the clip twice. Alongside this, I find the discover feed thing to be annoying, as well as the quick add feature. I feel like they're always trying to shove random people down my throat whenever I use the app. I just want to communicate with my friends. I wish there was an option to turn it off. The number of times I almost sent personal stuff to random people is crazy.",
    "The advertisement of Snapchat+ is seriously invasive and annoying. I also find it hilarious that at first you had to pay for My AI, but now you have to pay to get rid of it. I also wish there was an option to turn off the discover page since I find it so distracting. The one thing I'll never stop complaining about is the new bitmojis. I don't think anyone was asking for creepy 3D ones, and the 2D ones were far more charming. It feels like you're trying to copy Meta (Facebook), disappointing.",
    "The updates are making this app fade into obscurity and are absolutely full of bugs, instability, loss of useful and popular features, totally bizarre and unusable touch screen controls that are not synced with the actual place where you are supposed to touch. On my S22 Ultra, I have to hover a 1/4 of an inch above the icon for it to possibly work, but some on-screen items will never be able to utilize from the ridiculous discrepancy of syncing the VM layout to the touchscreen.",
    "It's an easy way to send pictures and talk to other people but now it's just getting to be too much, an off-brand TikTok page, the bitmoji updates are starting to look a bit weird, I just wish we're the way they used to be. The AI is getting really annoying too, if there was an option to remove it completely that would be nice because sometimes when I click on a completely different conversation, it opens the AI. The AI part of Snap is also making my phone glitch a lot, but it could just be me.",
    "Horrible. Why can't you fix your app so that when a person makes a continuous video. When you click on their page to get the second video, you can't find it.!!! Can you please make it easier for users to find the continuous video? This really upsets me. I watch a video, and they say they have to continue it in another video and well, you can't find it. It makes me mad!!. Gosh make it easier for your users!! I will change ratings once it's done!!",
    "The app functionality has been alright, not really here to comment on that. The bitmojis... I did not see the need to update them again, and they look worse. The extra detail really isn't needed when the previous ones actually looked good. Plus, the way the new one looks while editing and how it stands is definitely worse as well. If you're going to continue updating them, please keep the stickers how they are with the previous ones. Or give us an option to choose our bitmoji design, please.",
    "The discover is terrible, the spotlight is also bad, the AI is just annoying. Any ads they give or notifications for new features just get in the way and don't go away for a long time. There's also no point in trying to maintain streaks, as Snapchat will just take it away randomly, and asks for you to pay to restore it. And there is Snapchat+, which anything that you think would be accessible to the basic user, it's not. And stories repeatedly say you haven't watched them, even though you have.",
    "I love this app, been using it for a long time. love communicating with friends and groups. The filters are awesome and it's overall a good experience. However, ever since they updated the bitmojis and made them 3D it looks cringy and has kind of ruined it for me. I loved the 2D ones though please change it back to that. Also, you should be able to access Snapchat from the web on the phone, but u can only edit basic things. It's still a good app but they could improve.",
    "This app won't let me get past a pop-up window that comes up as soon as I open the app that states, We believe the camera should be optimized for each Snapchatter that uses it. To do that Snap uses info about your face, hands, and voice to make certain features work learn more, and if you WANT to agree and continue tap below. I don't want to agree to this. But it won't let me click out of it or use the app in any way. Uninstalling...very frustrating. This is ridiculous.",
    "Too many bots/spam/inaccurate friend suggestions. Useless features and too much push to promote paying for features. There should be a way to disable friend suggestions that appear beside stories since they are majority of the time people I have no idea who they are. There should be a way to keep 2 different profiles on the same device separate (including message requests and friend suggestions). AI should be able to have a disable/delete feature. Many flaws in the app make me reconsider using it.",
    "Unless you plan to purchase a Snapchat+ plan, don't bother. Various features available to premium users will be advertised in big banners across the screen, impeding your ability to actually interact with the app. Want to click on a chat stream with some friends? Better do it quick, before the entire list is rearranged because they're trying to sell you some new feature! Don't want your messages screen cluttered up with their mediocre AI? Tough, you gotta pay to hide that.",
    "Inconvenient updates. Unnecessary changes that mean you can't always see content you follow. Way too many ads. Certainly not worth paying for. Content often freezes or gets stuck loading. Pics/videos sent from friends or family, you occasionally have to completely close out the app for them to pop up, or they never load. Developers keep removing helpful features and don't put them back until after nearly a month, and they come back worse than before. Paid version not even worth free trial.",
    "It's a fun & easy way to take selfies & group pics to share with friends & family and to capture memorable moments because you are able to download/save photos to your device. I love the filters! There must be 1,000s of filters & the app allows you to save your favs & share. There's a lot more you can do within the app if you take the time to explore.",
    "Ever since the last update, the camera is in landscape mode as soon as I open the app, and there's no way to change it within the app itself. I have to either restart my phone or uninstall and re-install the app, then it will open properly again, but only for a few uses before it opens in landscape mode again and I have to repeat the process.",
    "DON'T SAVE VIDEOS OR PICTURES ON SNAPCHAT, After some time, the videos/photos you save on Snap, can't open anymore. So you lose them. This is not just with old videos you save but also with new videos. This is stupid. I use Snapchat for 7 years, and I am disappointed with it. Thank me later.",
    "It helps you to connect with thousands of people across the world. And most importantly AI (artificial intelligence) it helps an individual to get answers about any rising questions within a second. It is very really helpful for students, employees, etc. Enjoying with many more interesting lenses. Capturing every moment. I am having a great time using Snapchat, thank you.",
    "Now you can't share, can't export to your phone this is really really bad app. This app got the worst music and never have they changed them. On top of takes a lot of data just for opening it. I don't know why anything you do there goes data if not s little it's like 10 gigs that's why I stop using it. Cause it makes no sense then there's the update Why update if there's nothing new.",
    "Something changed recently to where when you post a memory to your story anyone viewing can't see what date/time it's from. It only says how long ago it was added to the story. Ex: If I post an OLD memory it says 1 hour ago instead of August 14th, 2021, in the top left under the user's name. I find it frustrating because when I want to post a memory I want to also include when it's from.",
    "The app was great before the new updates, since the new 3D bitmojis the app doesn't look great nor unique it's basic and annoying and the streaks are getting lost even though we send before the 24-hours max and you have to be subscribed to save them or restore them which is annoying, please bring back the old bitmojis back",
    "Has way too many ads. Each time I'm watching Snapchat I get at least 1 minute each video and get another ad. It's so frustrating. BUT NOW. They added a new feature that makes your avatar look realistic it's so ugly please change it back to normal I hate it and I wanna delete this app.",
    "Snapchat on Android is a disappointment. The app is buggy, crashes frequently, and the camera quality is terrible. It lags during snaps, making the experience frustrating. Updates seem to make it worse, not better. Overall, it is a subpar and frustrating social media platform. Update... App continues to get worse with bugs and even more things that many users dislike, such as the new bitmojis.",
    "I love Snapchat Always been a big fan of it. You can take pictures & videos & save those memories into your phone & into the app. You control who adds you & who sees your posts. It's a great way to connect with old friends & family & make silly memories especially with the funny filters. I just wish snap would add back the games because those games were fun..I don't really like the Snapchat+",
    "Recent photos (from every other application) now in the way when trying to take a snap; could just upload from Memories under camera roll? Moving the emojis to the left under friend icons creates too much clutter on the left side and is hard to look at. Please stop changing the UI you keep making it worse.",
    "Snapchat is good for straightforward chatting, but it has gotten worse lately. Features that you usually would need isn't there, like replay YOUR OWN snap... and it keeps getting changed! I would delete it if I could, but so many people are using it. Not email or something.",
    "The app would be much improved if it had a dark mode NOT BEHIND A PAY WALL. As well as more options for length of time messages are kept and not allowing them to be saved in chat. The filters are ok and the stories are ok. DARK MODE SHOULD NOT BE A PAY WALL",
    "Hey I was using nothing Phone 2 and it seems the app is not optimised for the device as the snaps are having too much processing. Moreover there was a glitch with the night mode it was there when I first installed the app after that the night mode option is not showing at all. To bring it again I have to reinstall the app then it will show but it will only be for the 1st time I open the app.",
    "Banned for trying to log in, my wifi is slow, especially when everyone is on it. I was trying to get I to Snapchat to send my streaks, and due to slow wifi, j was unable to get in, it would just load. I exited and reentered the app a couple times, when I tried logging in the next day, it said my access was temporarily blocked due to multiple failed attempts to log in. I left it sit for over 24 hours, tried logging in and got the same message. It has now been over 2 days, and I am still banned.",
    "One of the most grippling application as there are multifarious filters along with emojis.However,we can not only stay in the touch of our friends but also our celebrities whose have a account on that application and the unique thing about this app it give a lot of functions in one platform such as stories,function of: streaks and spotlight ,AI. Having a chat with AI such a excellent thing as we can ask anything whether it could be about our profession or knowledge . So users should go with it.",
    "I really love this app, I can message my friends, there are so many different filters, I can watch as well and I have maps. This app is really good. The only thing that is really annoying is the new update, it has made the avatars so bad. Especially the facial features such as eyes, facial shape and more. It's really annoying to look at the avatars of my friends in chat. This app would be 5 stars if the update would be removed. I think that there should be an option between 2D and 3D avatars.",
    "Camera won't orient itself upright. Takes pictures as if I'm holding the phone the opposite way that I am holding it. (portrait vs landscape orientation) the filters have also stopped working. What is the issue?"
]

# Convert new review text to TF-IDF features
new_reviews_tfidf = tfidf_vectorizer.transform(snapchat_reviews)

# Predict labels for new reviews
predicted_labels = svm_classifier.predict(new_reviews_tfidf)

# Interpret the predictions
speed_labels = {0: 'Not Related to Speed', 1: 'Related to Speed'}
predicted_labels = [speed_labels[pred] for pred in predicted_labels]

# Initialize a list to store speed reviews
speed_reviews_list = []

# Display the predictions and count the number of speed reviews
num_speed_reviews = 0
for review, label in zip(snapchat_reviews, predicted_labels):
    print(f"Review: {review}\nPredicted Label: {label}\n")
    if label == 'Related to Speed':
        num_speed_reviews += 1
        speed_reviews_list.append(review)
        
# Print the list of speed reviews
print("\nList of Speed Reviews:")
for idx, review in enumerate(speed_reviews_list, 1):
    print(f"{idx}. {review},\n")

# Output the number of speed reviews
print(f"Number of Speed Reviews: {num_speed_reviews} out of {len(snapchat_reviews)}")