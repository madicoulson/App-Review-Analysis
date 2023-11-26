# Import necessary libraries
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

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

# Convert snapchat review text to TF-IDF features
new_reviews_tfidf = tfidf_vectorizer.transform(snapchat_reviews)

# Predict labels for snapchat reviews
predicted_labels = svm_classifier.predict(new_reviews_tfidf)

# Interpret the predictions
UI_labels = {0: 'Not Related to UI', 1: 'Related to UI'}
predicted_labels = [UI_labels[pred] for pred in predicted_labels]

# Initialize a list to store UI reviews
UI_reviews_list = []

# Display the predictions and count the number of UI reviews
num_UI_reviews = 0
for review, label in zip(snapchat_reviews, predicted_labels):
    print(f"Review: {review}\nPredicted Label: {label}\n")
    if label == 'Related to UI':
        num_UI_reviews += 1
        UI_reviews_list.append(review)
        
# Print the list of UI reviews
print("\nList of UI Reviews:")
for idx, review in enumerate(UI_reviews_list, 1):
    print(f"{idx}. {review},\n")

# Output the number of UI reviews
print(f"Number of UI Reviews: {num_UI_reviews} out of {len(snapchat_reviews)}")