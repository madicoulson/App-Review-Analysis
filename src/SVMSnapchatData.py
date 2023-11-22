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
    "The app frequently crashes, especially when multitasking. Address this issue for better stability."
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
          1, 0, 1, 1, 0, 0, 0, 0, 1, 1]    

# Convert text data to TF-IDF features
tfidf_vectorizer = TfidfVectorizer(max_features=10000)  # You can adjust the number of features
X = tfidf_vectorizer.fit_transform(reviews)

# Create and train the SVM model
svm_classifier = SVC(kernel='linear', C=1, random_state=42)
svm_classifier.fit(X, labels)

# New reviews for prediction
new_reviews = [
    "I loved this app, but ever since the last update the filters don't work. Any filter that alters someone's face, I can take a picture but I can't take video anymore. The audio will come up normally and yet the video is like it's doing a slideshow, each frame holds for 5 seconds before switching to the next. Most of the app functions normally but filters that alter people's faces just will not work and it's driving me insane. They worked a couple weeks ago so what did you guys do????",
    "For the last few months, it is giving me problems with viewing my pictures. Super laggy and slow. And it no longer gives me the option to delete or edit my photos. No option menu at all. My internet connection is just fine, so it's not that. I tried uninstalling and reinstalling, but the app is still not letting me edit or delete photos anymore. No option menu is coming up on photos and it's taking super long just to be able to view a photo.",
    "Please for the love of God fix the notification bug! It's been over a year and I STILL can't get notifications from people trying to message me. I may as well unfriend them so they don't think I'm ignoring them when it's been two weeks before I even notice it's there.. I have tried EVERYTHING from reinstalling, clearing the cache, turning off do not disturb, making sure notifications are on and at max volume. Nothing. It's utterly impossible to stay in touch with anyone on this platform.",
    "This app is really annoying. And it takes all of your data even if you set the permissions to not allow it into your contacts. It still takes it. And the notifications don't stop even if you set them to only one a day, the only option to not be annoyed with their notifications all day long is to turn them completely off, so then you don't know when your friends have sent you a message.",
    "My AI is so annoying. I've clicked on it by accident many times. There should be a free option to get rid of it. I don't think it's fair that iPhone users get dark mode for free, but android users have to pay for premium to have the option. Go back to the simplicity. I've had snap since 2013. It's a great way to send photos & videos to friends & family & keep in touch. The Discover Page & tiktok pages are also annoying. If i wanna watch tiktoks I'll go to tiktok where the videos are clear.",
    "Whatever recent updates have been made have broken this app. I barely ever get notification sounds any longer, nor does it activate my screen. I haven't changed any settings. I suddenly just don't get notifications for some reason, so I keep missing messages from people. And then some messages are coming in super late. Never had these issues before.",
    "Works good enough for what I use it for. Great for quickly sending pictures and videos to friends. As well as sending text messages back and forth. Easily create group chats and do the same in them. I dislike the addition of My AI. It is consistently at the top of your screen. Always clicking it by accident when I'm trying to reply to the most recent person that snapped me. Extremely annoying little addition from Snapchat. The only way to remove it is to pay for premium Snapchat.",
    "This app is super buggy. Every time I add continuous video it either sends the cut clips in the wrong order, or duplicates the clip twice. Along side this I find the discover feed thing to be annoying, as well as the quick add feature. I feel like they're always trying to shove random people down my throat whenever I use the app. I just want you communicate with my friends. I wish there was an option to turn it off. The amount of times I almost sent personal stuff to random people is crazy.",
    "The advertisement of Snapchat+ is seriously invasive and annoying. I also find it hilarious that at first you had to pay for My AI, but now you have to pay to get rid of it. I also wish there was an option to turn off the discover page since I find it so distracting. The one thing I'll never stop complaining about is the new bitmojis. I don't think anyone was asking for creepy 3D ones, and the 2D ones were far more charming. It feels like you're trying to copy Meta (Facebook), disappointing.",
    "Absolutely terrible experience on Android. I have been a long-time user of this app on Android and have been waiting for the experience to improve. For one singular app version, it supported my 3x and 10x zoom cameras. I updated it a couple of days later, and it was just gone. They have proven that they have the ability for good camera integration but refuse to roll it out promptly. The fact that I use one of the most popular Android phones and they still can't get it right is astonishing to me.",
    
    "I use this app every day, basically since the app came out. Recently, it has been a lot more of a hassle to use. As others have pointed out, the My AI is inconvenient, and lately, I haven't been able to send anything. Every time I want to send a snap (chat, picture, video), I can actively press save/send, yet nothing will happen. The snap will just disappear as if it never existed in the first place.",
    "I use this app for a lot of things. It's over all a pretty okay app. However over the last few weeks, it randomly stops sending any snaps that I send. Like I take a photo and put text and send it to the person I'm talking to, but it does not actually send. Not even a sending loading message. And then it refuses to let me send already saved photos from my camera roll. I've tried uninstalling, deleting cache and getting off WiFi and it still isn't working.",
    "The updates are making this app fade into obscurity and is absolutely full of bugs, instability, loss of useful and popular features, totally bizarre and unusable touch screen controls that are not synced with the actual place where you are supposed to touch. On my s22ultra I have to hover a 1/4 of an inch above the icon for it to possibly work, but some on screen items will never be able to utilized from the ridiculous discrepancy of syncingthevm layout to the touchscreen.",
    "Don't know what they did, but my messages almost never load now, have to close and relaunch the app multiple times for them to show up. Just started a week or so ago. Editing to one star because it's somehow gotten worse. Now I can't get the messages to load at all, it takes at least 5 or 6 tries of closing and reopening the app just to load the messages. It's ridiculous. I've force closed, I've Uninstaller and reinstalled, nothing seems to fix it.",
    "This once was and remains to be one of my favorite methods of communication. However...The software is so unbelievably sluggish, often to the point that I miss the opportunity to capture the experience I'm trying to Snap. Volume changes are also delayed to the extent that I constantly miss dialogue or have to replay a snap just to hear what they said. The price charged for premium Snapchat is downright absurd as well given what little it offers to change the experience.",
    "There is a bug where after you type a message onto a snap, it just disappears when you close the keyboard. I have reported this bug multiple times and it hasn't been fixed in months. I use the app a lot and it happens several times a day, for at least half of all snaps I send. It's extremely annoying, and it seems like there is no attempt to fix these bugs.",
    "It does what it's supposed to, but I do have a few problems with the app. Like when you're on call with someone and when you open a different app, the call doesn't continue, and there's not pop up window anymore even when I have the setting turned on for that to happen. Also, it took too long for android users to get dark mode, which I don't find completely fair. Edit: I've removed a star because of the hideous bitmojis",
    "I like Snapchat, I think it is a great communications tool and I love keeping streaks with friends, however, the Discover tab and the AI both need to have options to remove them. It is very frustrating that there is so much clutter and distractions with no way to customize or remove unwanted items. I really don't want to delete Snapchat, but if there is no way to resolve this issue, I may have to result back to old fashioned texting for communication again.",
    "it's a easy way to send pictures and talk to other people but now it's just getting to be too much, an off-brand tiktok page, the bitmoji updates are starting to look a bit weird, I just wish we're the way they used to be. The AI is getting really annoying too, of there was a option to remove it completely that would be nice because sometimes when I click on a completely different conversation, it open the AI. The AI part of snap is all making me phone glitch a lot, but it could just be me.",
    "Horeible. Why can't you fix your app so that when a person makes a continuous video. When you click on there page to get the second video you can't find it.!!! Can you please make it easier for user to find the continuous video. This really upsets me. I watch a video and they say they have to continue it in another video and we'll you can't find it. It makes me mad!!. Gosh make it easier for your users!! I will change ratings once it's done!!",
    
    "Recent changes to this app have NOT been for the better. The new 3D bitmoji models are an affront to nature. In addition, there's absolutely no reason I need an AI that sits at the top of my chat list no matter what. Finally, the news/advertisements section is horrendous. There is NOTHING of value from that part of the app, and I wish I could make it disappear. The slow phase out of this app in my friends and my own life is warranted.",
    "The app functionality has been alright, not really here to comment on that. The bitmojis..I did not see the need to update them again, and they look worse. The extra detail really isn't needed when the previous ones actually looked good. Plus, the way the new one looks while editing and how it stands is definitely worse as well. If you're going to continue updating them, please keep the stickers how they are with the previous ones. Or give us an option to choose our bitmoji design, please.",
    "Revising edit. This apps bugs just keep getting worse. The black box is still soo annoying. And now, if you lose a streak it show that you left them on opened even if you responded. The bugs in this app are getting worse and worse and they need to go back to what they know and what has worked previously. Stop trying to do stuff that people don't like!",
    "So since my last review it seems the app has gotten worse. It never shows any notifications for me whenever I get them, it takes a while (like an hour after the message was sent) for it to send. It lags so much for me and sometimes whenever I try to send something it completely freezes up and deletes the message. You guys need to do better for us android users.",
    "After the new update, me and my friend are both experiencing a lot of issues. When we call, there's no audio, ever. We can't hear each other at all so we have to call on a different platform. And now when video snaps are sent, no audio can be heard. And we can no longer send audio messages because it won't work. Also, bring back the old bitmojis",
    "What possess you to give a notification when someone is typing? Who asked for that? It's unnecessary and annoying. We don't need two notifications for every message received. The least you could do is allow us to turn it off if we don't want that. Addition: Now it keeps crashing after watching even a single story. Not to mention that if anyone posts while I'm viewing something, it starts glitching and crashes on itself. Snapchat is quickly becoming obsolete, and you only have yourself to blame.",
    "Had my account since February of 22, and I've noticed some things, one, in order for it to run smoothly on a android, you kinda have to have snap +, even then it's still buggy, the calls suck, and it freezes alot still, and sometimes you can't see/open snaps until hours after they've been sent. but onto positive stuff, it's a good way to keep in touch with family and friends and also to find and follow snap stars",
    "I was on snap for a pretty long time. I liked everything about it, it was very easy to use and you can chat with friends and it just has so many functions that just make it better. But now the AI is annoying me and I can't get rid of it, and my bitmoji scares me. It looks possessed and it makes me uncomfortable. Change the bitmojis back to the old style please! And get rid of this AI!",
    "The discover is terrible, the spotlight is also bad, the AI is just annoying. Any ads they give or notifications for new features just gets in the way, and doesn't go away for a long time. There's also no point in trying to maintain streaks, as Snapchat will just take it away randomly, and asks for you to pay to restore it. And there is Snapchat+, which anything that you think would be accessible to the basic user, it's not. And stories repeatedly say you haven't watched them, even tho you have.",
    "10/2022: the AI feature is something I will never use and there should be an option remove or block it. I also don't like the notifications from the discover part of the app. Not sure if there's a way to turn them off, haven't looked yet. 9/2018: Multiple times the app never told my friends and I that are streak is about to break until it's too late",
    
    "I have never used, nor will I use the My AI. It's a waste of space. I absolutely hate that there is no way to get rid of it or have the option of turning it off? It's really annoying.🚩 It's biased to only let iPhone users have dark mode instantly, and me, an Android user pay for it? If it's free on iOS, I want it to be free everywhere else, please.🚩 Aside, I think my favorite thing about the app is the sticker maker. I haven't made any new stickers recently, but tossing in customized faces.❤",
    "Snapchat is an amazing app to respond and socialize with friends, even when they are far away! The app has everything. Such as the stories feature, the 'tiktoks', pictures, filters, and even a map. But, I really despise the AI bot. It is kind of disturbing and weird. Recently, it posted a story on its page, which is absurd for it to happen. Honestly, Snapchat should just make it an option to have it 😊🥰.",
    "No dark mode, no easy way to access camera gallery folders, and the AI feature is a harassment. The new update also forces you to accept their new privacy invasion and personal data collection of your face and other things. The app crashes and closes itself out often. However, I can almost always reopen and use it quickly. It's an easy app with friendly UI. It's the easiest way to talk to people when in different countries and I love using it on a daily basis. I really do love the app.",
    "I use Snapchat as a primary mode of communication with several of my friends. However, despite trying ALL of the fixes and tricks, ensuring that my Internet connection is strong and secure, and running the app on an up-to-date phone, it has still defaulted to taking FOREVER to send videos. When they do send, they're out of order. It really may just be time to delete this app.",
    "App is okay for quickly sharing non-important photos and videos, but it's seriously bloated with features that really aren't useful. The interface is a convoluted mess, and there's probably better options available. The most egregious addition was a My AI that is permanently pinned to the top of your messages/contacts. And you can not remove it unless you pay money. It's the tipping point for me, I've accidentally selected the AI so many times, and it's becoming incredibly frustrating.",
    "Used to be better. The Discover page is just clickbait and after you're done viewing your friends' stories it just plays the first thing on the page on its own. The 3D Bitmojis are kind of annoying, since there isn't an option to perfer showing the original ones. However, the addition of 5 streak restores per month is brilliant! It's very easy to do and a reasonable number of restores. Good job with that! When it comes to My AI, please stop forcing it. Give us the option to remove it. That's all",
    "Firstly, I have noticed that the search function on Snapchat is not loading properly. When I try to search for specific usernames or content, the search results take a long time to load, and sometimes they don't load at all. This has been happening consistently over the past few days.Secondly, I am facing difficulties with the sticker feature.When I try to add stickers to my Snaps, they are not appearing in the sticker drawer. I have tried reinstalling the app and updating to the latest version.",
    "The app is pretty cool with all the features and stuff, but it is a bit annoying that it creates so much cache so quickly. Like I use the app for half an hour and it has already built up like 450 megabytes of cache. Also, recently, whenever I try to open an ar lens, the app freezes, then a few seconds later, it crashes. Please fix your app.",
    "I love this app, been using it for a long time. love communicating with friends and groups. The filters are awesome and it's overall a good experience. However, ever since they updated the bitmojis and made them 3D it looks cringy and has kind of ruined it for me. I loved the 2D ones though please change it back to that. Also you should be able to access snapchat from the web on phone, but u can only edit basic things. It's still a good app but they could improve.",
    "Snapchat is a popular social media app known for its variety of filters to click photos and videos. It offers a fun and interactive way to stay connected with friends through snaps and Stories. However, its user interface can be a bit confusing for newcomers ,and privacy concerns have been raised in the past. Overall, Snapchat is great for sharing moments, but it may not be everyone's cup of tea.",
    
    "I lost my streaks even though I paid within the given time. I'm expecting to be given my money back, even if it is just 99p. Also, the AI chat is pretty pointless, and from what I know, you can't delete it. It can lag quite a lot too, especially when recording. However, it's alright for chatting and scrolling on videos, and the filters are fun.",
    "A FEW THINGS WRONG.... Every single filter freezes for a couple minutes when I try to take a video & it does it a couple times in a row before it goes back to normal, also sometimes the videos start out glitchy & slow.. Every single day for weeks now. Also, now the light switches to brighter for a second then goes back to normal when I'm taking a video too, I have to have my phone at a certain uncomfortable angle so the camera light doesn't do that.",
    "Sometimes when I try to change the settings on a friend's chat or react to something they said, it either takes too long for the menu to pop up, or I have to exit out of the app two or three times before it works. And I get tired of seeing people in the quick add section; can there be an option to turn that off? Since the last update, there has been a red dot where my bitmoji is in the upper left corner and it won't say why it's there which is annoying. Please get rid of it.",
    "This app won't let me get past a pop-up window that comes up as soon as I open the app that states, We believe the camera should be optimized for each snapchatter that uses it. To do that Snap uses info about your face, hands, and voice to make certain features work learn more, and if you WANT to agree and continue tap below. I don't want to agree to this. But it won't let me click out of it or use the app in any way. Uninstalling...very frustrating. This is ridiculous.",
    "Snapchat is a bit hit and miss these days. I like that you can send snaps, videos and voice messages as well as just speaking through text, and I like the dark mode setting and vast range of filters, however there seems to be more bugs in the app with every new update. Things like the app not sending me notifications for snaps or messages, completely freezing up where I have to force stop it, the send button not working, etc. The app would be better if they fixed all these bugs.",
    "For some reason after the AI was added I've been having constant issues with using the app. First you can't delete the ai and I hate that it's even there. Second I have to open and close the app when trying to view messages because they refuse to show up or be visible, I can keep trying to scroll up but it doesn't do anything, takes me a minute of this process to even see the message. I'm not happy. This has been my problem ever since this new update.",
    "this app is great, I love taking photos of me & family , friends ect!! the filters are amazing. I also love how you can record, add music and more because you can film videos for fun (and other stuff) , the only problem is that I hate how when you change your avatar it randomly sends you off so you have to change it again, and I hate how the 3d bitmoji when I try to change my avatar, I really think they should have a option to change it, but other than those things, this app is amazing!",
    "For the average young person, this app is perfect. In my opinion, it's the best way to communicate with people online. When it comes to the creator side, there's alot that can be improved. The main thing that bugs me is how many of your personal Spotlights you're able to see under the Spotlight tab under the public profile. It often only shows the first few recent Spotlights you post, which becomes a hassle when trying to select which Spotlights appear on your profile.",
    "This app is amazing! I always call and message this app. I love when new updates come as well because I always get very excited! If you can, please upgrade and make it so it's not too laggy! Maybe you can add a video recording, meaning there are voice messages in chat, maybe add it so you can video record in chat, we can send little messages with a video! just click on the microphone, and it will change to video recording in the chat! Thanks, I hope you like my idea. I hope it turns out amazing!",
    "Pros: - filters - ability to talk with friends - group chats Cons: - AI (don't want/need) - videos not loading for hours on end. - having to restart the app, or even restart my phone just for the videos to load. - my snaps not sending (sometimes for hours) if I switch screens (app is still active in the background).",
    
    "This app used to be amazing as is, perfect bitmojis nice and simple, easy access and fast loading the app was great. But now it's one of the worst apps out there, they ruined bitmoji and now it looks like a freaky doll, we have an ai that we cannot get rid of?, and the filters are all boring and take long to load, I recommend you bring back snapchat 2019, the best version of the app or no one will be enjoying this app much longer. Please consider this recommendation.",
    "The application suffers from a significant number of glitches. When attempting to edit my snap and then add a song, upon returning, the entire experience becomes disorganized and chaotic. The recipient of my snap is presented with a distorted image that I never intended, thanks to these glitches. Moreover, any information I edit is inaccurately placed, and the app automatically selects presets or filters that I was merely checking. This disrupts the intended user experience.",
    "Too many bots/ spam/ inaccurate friend suggestions Useless features and too much push to promote paying for features. There should be a way to disable friend suggestions that appear beside stories since they are majority of the time people I have no idea who they are. There should be a way to keep 2 different profiles on a same device separate (including message requests and friend suggestions). AI should be able to have a disable/delete feature. Many flaws in app make me re-consider using it.",
    "Since the last update, Snapchat can't open my camera. I've tried clearing the cache and settings and completely reinstalling the app, and no other apps are having this problem. Other parts of the app work fine, it just can't take photos and throws an error if I try to switch between front and rear cameras.",
    "Unless you plan to purchase a Snapchat+ plan, don't bother. Various features available to premium users will be advertised in big banners across the screen, impeding your ability to actually interact with the app. Want to click on a chat stream with some friends? Better do it quick, before the entire list is rearranged because they're trying to sell you some new feature! Don't want your messages screen cluttered up with their mediocre AI? Tough, you gotta pay to hide that.",
    "Inconvenient updates. Unnecessary changes that mean you can't always see content you follow. Way too many ads. Certainly not worth paying for. Content often freezes or gets stuck loading. Pics/videos sent from friends or family, you occasionally have to completely close out the app for them to pop up, or they never load. Developers keep removing helpful features and don't putting them back until after nearly a month, and they come back worse than before. Paid version not even worth free trial",
    "One of the features that I believe Snapchat should implement is the ability to view the online status of other users. This would allow users to know when their friends or contacts were last active on the app, and potentially increase the engagement and communication among them. I think this feature would be beneficial for several reasons. First, it would help users to avoid sending messages to someone who is not online or available, and reduce the frustration of waiting for a reply.",
    "My camrea is opening side ways and once I finish recording its upside down? and my messages on my snaps are deleting too so I have to retype everything again. The help center was no help as it was confusing and I was unable to find the issue I was looking for. Otherwise from this issue I would say this is great app to use to talk to friends and mess around with.",
    "It's a fun & easy way to take selfies & group pics to share with friends & family and to capture memorable moments bc you are able to download/save photos to your device. I love the filters! There must be 1,000s of filters & the app allows you to save your favs & share. There's a lot more you can do within the app if you take the time to explore.",
    "I have to agree to Snapchat using info about my face, hands, and voice to make certain features work by clicking Sounds Good. That doesn't sound good. I don't trust the app/company to use that information responsibly. I can't use the app at all until I agree, including accessing snaps/chats unless I access it through my phone notification. If I clear the notification before clicking it, I'm out of luck. If my friend group didn't use this app here and there I would have already deleted it.",
    
    "Good app, but it has privacy issues. Sbapchat plus members can now see when people half swipe into chat and a lot of other things, and the friends can not control it. It's not a spying app. You don't need to see almost everything that your friends are doing.",
    "The bitmojis are getting worse and worse. Nobody even asked for a 3d bitmoji. It would be better if there was an option to have it 2d or 3d, but right now, it just looks horrible. The AI thing they introduced is also annoying, being at the top of the messages list and not being able to remove it despite its uselessness.",
    "NO AUTO FILTERS. People should have the OPTION to choose a filter and not have filters and beauty standards forced down our throats every minute of the day. There is an auto filter that smooths pores and even enhances and changes the photo. It does this even on that of a kitten. Disgusting we can't even look at our own images anymore.",
    "Ever since the last update, the camera is in landscape mode as soon as I open the app, and there's no way to change it within the app itself. I have to either restart my phone, or uninstall and re-install the app, then it will open properly again, but only for a few uses before it opens in landscape mode again and I have to repeat the process.",
    "the AI is annoying and intrusive. there's no reason it should be forced into the top of ur friends list, and no reason why you can't delete it unless you PAY for it. I already thought locking pretty basic features behind a pay wall was cheap and dirty, but the AI makes it worse. if you don't have snapchat already-- don't get it.",
    "Having lots of problems with the camera working. I have to completely exit out of the app, when that doesn't work I have to completely shut my phone off and turn it back on and sometimes that won't even work I finally had to uninstall reinstall I'm hoping this will work but when my camera does work it's so glitchy that I can't even get a picture that I want when if it's in motion because it's glitched so bad that it takes about a minute to catch up",
    "I have snap+ but still encounter so many of the issues that are in the complaints. My snaps aren't posted in the right order, or it gets duplicated. When I watch people's stories, it doesn't mark them as watched. Recently, the conversion I have with one person is glitched and I can't see any of the messages I send because they just dissappear as well as the rest of the UI. If I open other chats after this one they all get affected, meaning I can't chat with this one person. btw, one of my bsf",
    "The is a great app don't get me wrong, I've had my account since 2018. The main issue I see as of now(probably been for alot of users recently) is the excessive ads when you watch stories from public profiles like Mr Beast Etc. Having the ads is understandable but to put ads every two or three clips just gets really annoying, sometimes I miss the old snapchat because of this",
    "The app is generally excellent, easy to use, and safe, however it is unfair to charge for all of the primary features. They're essentially pressuring us to purchase a subscription, and My AI is quite bothersome. There's no real use for it, so at the very least, provide a way to deactivate it so that users can unfriend it.",
    "A relatively functional app that feels like it's pushing microtransactions as hard as possible. I am asked every time I open the app to join a group, advertised news media stories, and now their exclusive AI is always at the top of my page, not my most recent message. The only way that I can find to remove these features is to pay $3.50 a month and unlock the buttons to say no thank you.",
    
    "After the November 9 update no quick access stickers such as time, location won't show up. I've reinstalled the app which cleared all the data in my phone and I've also tried clearing the cache but it doesn't work. The light balance darkens my photos so much during the day they're invisible.",
    "the very image of shallow consumerism. i only use it when my friends do, and it's under duress. otherwise it's a total waste of space. it attracts the dumbest people on earth, the ads are ridiculous and probably malware, the ai feature is moronic and intrusive, and no i will not pay for more mediocre features for a below average app. everything about it screams corporate suits trying to be hip and with it like they haven't replaced their humanity with bar graphs. awful.",
    "Horrible, it's so laggy sometimes and when I try to edit a camera roll video it sends the videos in individual clips and then I look stupid so I have to delete all of them, and then when you send a video it makes you go back to text instead of leaving you on the camera roll so I have to go all the way bag down to find something again",
    "DON'T SAVE VIDEOS OR PICTURES ON SNAPCHAT, After some time, the videos/fotos you save on snap, cant open anymore. So you lose them. This is not just with old videos you save but also with new videos. This is stupid. I use snapchat for 7 years, and I am disappointed with it. Thank me later.",
    "It helps you to connect with thousands of people across the world. And most Importantly AI (artificial intelligence) it helps an individual to get answers about any rising questions within a second. It is very really helpful for students, employees, etc. Enjoying with many more interesting lenses. Capturing every moment. I am having a great time using Snapchat, thankyou.",
    "this app has always been awesome, one thing i don't really like though is the snapchat plus, for me since i don't have it, it won't let my streaks go past or higher than 25, i also don't like the bitmoji update to make it look realistic, i like the old original bitmoji, and it does lag sometimes and go slow but not that often.",
    "Making the bitmojis and stickers 3d was an awful idea!👎 It doesn't seem fun anymore. All the reasom im using this app is for silly and fun interface and gimmicks of it. I think its the case for most other people too. Even besides that, the realistic ones look ugly. Dear team snapchat, please return them back to normal",
    "Now you can't share can't export to you phone this is really really bad app This app got the worst music and never have they change them. On top of takes allot of data just for opening it. I don't know why anything you do there goes data if not s little it's like 10 gig that's why I shot using it. Cause it makes no sense then there the update Why update if there's nothing new",
    "Cash grab garbage. I don't watch videos on the app because why would I ever? But my buddy tried to show me a short video and it was riddled with ads every 30 seconds. And then My AI. The feature a minority of people may have wanted. And if you didn't want it, too bad! The chat stays in your feed. BUT WAIT! We'll give you the ability to hide it if you give us money and get Snapchat+! Literally the worst business decision I've seen for a mobile app. Total garbage.",
    "Something changed recently to where when you post a memory to your story anyone viewing cant see what date/time its from. It only says how long ago it was added to the story. Ex: If I post an OLD memory it says 1 hour ago instead of August 14th 2021 in the top left under the users name. I find it frustrating because when I want to post a memory I want to also include when it's from.",
    
    "The app was great before the new updates, since the new 3D bitmojis the app doesn't look great nor unique it's basic and annoying and the streaks are getting lost even though we send before the 24-hours max and you have to be subscribed to save them or restore them which is annoying, please bring back the old bitmojis back",
    "HAS WAY TO MUCH ADS. Each time I'm watching Snapchat I get atleats 1 minute each video and get another ad. It's so frustrating. BUT NOW. They added a new feature that makes your avatar look realistic it's so ugly please change it back to normal I hate it and I wanna delete this app",
    "Since the new update I am facing difficulties with the sticker feature. When I try and add stickers to my Snapchat, they are not appearing in the sticker drawer section. I am also having difficulty with searching for any specific names or content, it takes a really long time to load or it doesn't load at all. I have checked all settings and everything is fine and I have deleted the app and reinstalled it and still not working so It's very frustrating for me and would like for this to be sorted",
    "It's a little bit unfair that android users don't get a chance to set their chats to dark theme. You know, one thing I absolutely hate about this app is that AI. The fact that we can't unfriend it. It barely gives me correct answers and just keeps repeating its answers... I have updated the app so many times, but I still don't get the features I updated it for. Honestly, I might uninstall this app. When I updated the app, I wasn't able to save stories in chat anymore. The app is getting worse.",
    "Latest version sometimes glitches with camera on pixel 7 pro after being open for a long time and jumping in and out of different apps prior to returning to snap. Camera becomes stretched and takes weird photos that are shown skewed to other people and isn't just displaying incorrectly During this oddity, using bultin camera app on phone, poses no issues",
    "Snapchat on Android is a disappointment. The app is buggy, crashes frequently, and the camera quality is terrible. It lags during snaps, making the experience frustrating. Updates seem to make it worse, not better. Overall, it is a subpar and frustrating social media platform. Update... App continues to get worse with bugs and even more things that many users dislike, such as the new bitmojis.",
    "I love Snapchat Always been a big fan of it. You can take pictures & videos & save those memories into your phone & into the app. You control who adds you & who sees your posts. It's a great way to connect with old friends & family & make silly memories especially with the funny filters. I just wish snap would add back the games because those games were fun..I don't really like the Snapchat+",
    "Recent photos (from every other application) now in the way when trying to take a snap; could just upload from Memories under camera roll? Moving the emojis to the left under friend icons creates too much clutter on the left side and is hard to look at. Please stop changing the UI you keep making it worse.",
    "Snapchat is good for straightforward chatting, but it has gotten worse lately. Features that you usually would need isn't there, like replay YOUR OWN snap... and it keeps getting changed! I would delete it if I could, but so many people are using it. Not email or something.",
    "Ever since Snap+ it's been horrible. Nothing new unless you pay, no improvements. Which is crazy because they are already double dipping with the ads and now the subscription service. Users were really shown with this one they aren't cared about unless they pay.",
    
    "The app would be much improved if it had a dark mode NOT BEHIND A PAY WALL. As well as more options for length of time messages are kept and not allowing them to be saved in chat. The filters are ok and the stories are ok. DARK MODE SHOULD NOT BE A PAY WALL",
    "Without a doubt the worst chat app I've ever seen in my life. I could say so much, but I'll sum this barely working greed gutter up by asking you this: if you wanted to push a subscription service that did nothing useful at all in the absolutely worst, verging on immoral, ways how would you do that? (And when all else fails, just take away features that have been free since day one. Then they HAVE to like it!)",
    "Hey I was using nothing Phone 2 and it seems the app is not optimised for the device as the snaps are having too much processing. Moreover there was a glitch with the night mode it was there when I first installed the app after that the night mode option is not showing at all. To bring it again I have to reinstall the app then it will show but it will only be for the 1st time I open the app.",
    "I love using Snapchat to talk to my friends and engage with others. The only gripe I have currently is somewhat of a bug. People complained when the back button was moved next to the call button and then it was moved back to the other side. Mine is still on the right side near the call but my other friends have it on the opposite side. If I can somehow fix this, I'll change my rating to 5 stars 🌟",
    "Banned for trying to log in, my wifi is slow, especially when everyone is on it. I was trying to get I to Snapchat to send my streaks, and due to slow wifi, j was unable to get in, it would just load. I exited and reentered the app a couple times, when I tried logging in the next day, it said my access was temporarily blocked due to multiple failed attempts to log in. I left it sit for over 24 hours, tried logging in and got the same message. It has now been over 2 days, and I am still banned.",
    "One of the most grippling application as there are multifarious filters along with emojis.However,we can not only stay in the touch of our friends but also our celebrities whose have a account on that application and the unique thing about this app it give a lot of functions in one platform such as stories,function of: streaks and spotlight ,AI. Having a chat with AI such a excellent thing as we can ask anything whether it could be about our profession or knowledge . So users should go with it.",
    "Great app. I text my girlfriend on it, but these past two days, it keeps saying sending on my messages. It never sends them, and when someone's texts me, itm get the notification about it, but the chat between me and the person doesn't show anything new. I knows it's not a problem with my data, because I can launch any other app perfectly fine. It can't be a problem with my account either, because I can still log in when I'm connected to WiFi. Please solve this",
    "I really love this app, I can message my friends, there are so many different filters, I can watch as well and I have maps. This app is really good. The only thing that is really annoying is the new update, it has made the avatars so bad. Especially the facial features such as eyes, facial shape and more. It's really annoying to look at the avatars of my friends in chat. This app would be 5 stars if the update would be removed. I think that there should be an option between 2D and 3D avatars.",
    "Stops in the mid of videos and doesn't even show them, just has a loading symbol. 😒 this day and age you'd think ppl would be smart enough to make a simple app. Update**.. sometimes it takes long time to upload a video.. and when trying to view my memories it doesn't load. And when I went to look at my memories b4, it showed me pics and videos I didn't even upload!!! Wonder if I can do something bout that. Uploading private pics and vids without my consent 🤔😤😤😡",
    "Camera won't orient itself upright. Takes pictures as if I'm holding the phone the opposite way that I am holding it. (portrait vs landscape orientation) the filters have also stopped working. What is the issue?"
]


# Convert new review text to TF-IDF features
new_reviews_tfidf = tfidf_vectorizer.transform(new_reviews)

# Predict labels for new reviews
predicted_labels = svm_classifier.predict(new_reviews_tfidf)

# Interpret the predictions
actionable_labels = {0: 'Not Actionable', 1: 'Actionable'}
predicted_labels = [actionable_labels[pred] for pred in predicted_labels]
# ... (previous code remains unchanged)

# Initialize a list to store actionable reviews
actionable_reviews_list = []

# Display the predictions and count the number of actionable reviews
num_actionable_reviews = 0
for review, label in zip(new_reviews, predicted_labels):
    print(f"Review: {review}\nPredicted Label: {label}\n")
    if label == 'Actionable':
        num_actionable_reviews += 1
        actionable_reviews_list.append(review)
        
# Print the list of actionable reviews
print("\nList of Actionable Reviews:")
for idx, review in enumerate(actionable_reviews_list, 1):
    print(f"{idx}. {review},\n")

# Output the number of actionable reviews
print(f"Number of Actionable Reviews: {num_actionable_reviews} out of {len(new_reviews)}")