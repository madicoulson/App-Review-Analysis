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
    
    "The app is amazing! Love the new features.",
    "Constant crashes. Needs urgent fixing.",
    "Great app overall, but a dark mode would be nice.",
    "Security vulnerability detected. Fix ASAP!",
    "App is user-friendly. No complaints.",
    "Frequent freezes on my phone. Please resolve.",
    "Love the simplicity. No need for changes.",
    "Outdated design. Needs a modern look.",
    "Decent app, but lacks essential features.",
    "The app crashes on startup. Immediate attention required.",
    
    "The latest update made the app slower. Improve performance!",
    "I like the design. Keep up the good work!",
    "Encountered a bug causing data loss. Fix urgently.",
    "Sleek design, but too expensive for limited features.",
    "The app's dark mode is fantastic. No changes needed.",
    "The UI is confusing. Consider simplifying.",
    "Reliable app. It has all the features I need.",
    "Frequent crashes. Needs immediate fixing.",
    "Love the app's performance. No complaints.",
    "Encountered a security flaw. Strengthen encryption.",
    
    "Perfect for my needs. No issues at all.",
    "Simple design, but could use more features.",
    "The app freezes on my phone. Optimize for better performance.",
    "Excellent customer support. Quick responses.",
    "Crashes occasionally. Investigate and fix.",
    "Perfect app. No need for changes.",
    "Lacks some essential features. Consider adding them.",
    "The app is worth the price. Provides all necessary features.",
    "App frequently freezes. Optimize for better performance.",
    "User-friendly interface. No complaints.",
    
    "Crashes on startup. Immediate attention required.",
    "Fantastic app! No need for changes.",
    "Outdated design. Needs a modern look.",
    "Decent app, but could use more features.",
    "The app crashes occasionally. Fix this issue.",
    "Amazing app! Love the new features.",
    "UI is confusing. Consider simplifying.",
    "The app is reliable. No complaints.",
    "Encountered a bug causing data loss. Implement better error handling.",
    "Sleek design, but too expensive for limited features.",
    
    "Love the app's dark mode. No need for changes.",
    "The UI needs improvement. Consider a redesign.",
    "App is reliable. It has all the features I need.",
    "Frequent crashes. Needs immediate fixing.",
    "App's performance is excellent. No complaints.",
    "Encountered a security flaw. Strengthen security measures.",
    "Simple design, but could use more features.",
    "App freezes on my phone. Optimize for better performance.",
    "Excellent customer support. Quick responses.",
    "Crashes occasionally. Investigate and fix.",
    
    "The app's latest update is fantastic! Great job!",
    "Frequent crashes. Fix this issue immediately.",
    "User-friendly interface. No complaints at all.",
    "Encountered a bug causing data corruption. Urgent fix needed.",
    "Love the simplicity. No need for changes.",
    "App freezes on my phone. Optimize for better performance.",
    "Outstanding customer support. Quick and helpful responses.",
    "Crashes occasionally. Investigate and fix.",
    "Perfect app for productivity. No complaints.",
    "The app's design is outdated. Consider a modern look.",
    
    "Great app overall, but a dark mode would be nice.",
    "Security vulnerability detected. Fix ASAP!",
    "App is easy to use. No issues at all.",
    "Frequent freezes on my phone. Please resolve.",
    "Love the simplicity. No need for changes.",
    "Outdated design. Needs a fresh, modern look.",
    "Decent app, but could use more features.",
    "The app crashes on startup. Immediate attention required.",
    "Perfect for my needs. No issues at all.",
    "The latest update made the app slower. Improve performance!",
    
    "The design is great. Keep up the good work!",
    "Encountered a bug causing data loss. Fix urgently.",
    "Sleek design, but too expensive for limited features.",
    "The app's dark mode is fantastic. No changes needed.",
    "The UI is confusing. Consider simplifying.",
    "Reliable app. It has all the features I need.",
    "Frequent crashes. Needs immediate fixing.",
    "Love the app's performance. No complaints.",
    "Encountered a security flaw. Strengthen encryption.",
    "Simple design, but could use more features.",
    
    "The app freezes on my phone. Optimize for better performance.",
    "Excellent customer support. Quick responses.",
    "Crashes occasionally. Investigate and fix.",
    "Perfect app. No need for changes.",
    "Lacks some essential features. Consider adding them.",
    "The app is worth the price. Provides all necessary features.",
    "App frequently freezes. Optimize for better performance.",
    "User-friendly interface. No complaints.",
    "Crashes on startup. Immediate attention required.",
    "Fantastic app! Love the new features.",
    
    "UI is confusing. Consider simplifying.",
    "The app is reliable. No complaints.",
    "Encountered a bug causing data loss. Implement better error handling.",
    "Sleek design, but too expensive for limited features.",
    "Love the app's dark mode. No need for changes.",
    "The UI needs improvement. Consider a redesign.",
    "App is reliable. It has all the features I need.",
    "Frequent crashes. Needs immediate fixing.",
    "App's performance is excellent. No complaints.",
    "Encountered a security flaw. Strengthen security measures.",
    
    "Simple design, but could use more features.",
    "App freezes on my phone. Optimize for better performance.",
    "Excellent customer support. Quick responses.",
    "Crashes occasionally. Investigate and fix.",
    "Innovative features make this app stand out. Excellent work!",
    "Experienced a minor glitch. Fixing it would improve the app.",
    "User interface needs improvement. It's a bit confusing.",
    "App crashed during a critical task. Urgent fix required.",
    "Loving the new update! The added features are fantastic.",
    "Encountered a security issue. Strengthen the app's protection.",
    
    "Impressive app design. The user interface is intuitive.",
    "Encountered a bug that needs urgent attention. App crashed unexpectedly.",
    "Lack of dark mode is a drawback. Consider adding this feature.",
    "This app is a game-changer. It's transformed how I manage tasks.",
    "The latest update made the app slower. Performance optimization needed.",
    "Sleek and modern design. One of the best apps on the market.",
    "Security concerns. A vulnerability was identified. Immediate fix required.",
    "Fantastic app! It has all the features I need for productivity.",
    "The app's speed is outstanding. Responds quickly to user commands.",
    "A bit pricey for the limited features. Consider adjusting the pricing.",
    
    "Crashes occasionally. Investigate and resolve the issue promptly.",
    "I appreciate the regular updates. Developers are committed to improvement.",
    "This app exceeded my expectations. Feature-packed and easy to use.",
    "User-friendly interface. But it lacks some essential functionalities.",
    "Regret purchasing. Does not meet advertised capabilities.",
    "Frequent freezes. Optimize for better performance.",
    "The app's reliability is impressive. No issues so far.",
    "Outdated design. Consider a modern redesign.",
    "Crashes on startup. Urgent attention needed.",
    "A must-have app for productivity. Simplifies workflow seamlessly.",
    
    "Decent app overall, but customer support is lacking.",
    "Security vulnerability detected. Strengthen encryption measures.",
    "New features needed in the next update.",
    "Simple and effective app. Ideal for daily tasks.",
    "Crashes frequently. Frustrating user experience.",
    "Customer support is responsive and helpful.",
    "I love the simplicity of the app. No unnecessary complications.",
    "A bug causing data loss. Implement better error handling.",
    "This app is perfect for my needs. Love the speed and design.",
    "The latest update improved performance significantly. Good work!",
    
    "App's design is sleek. Appreciate the modern look.",
    "I use this app daily, and it works very well for me.",
    "App's performance on my device improved after the recent update.",
    "A gem of an app. Simplifies complex tasks effortlessly.",
    "Not satisfied with the app's performance. Lags frequently.",
    "Highly recommend this app! User-friendly and feature-packed.",
    "Reliable app. No issues encountered so far.",
    "The app crashes frequently, making it frustrating to use.",
    "Great app overall, but it lacks a dark mode. Consider adding.",
    "App is user-friendly, but occasional lags hinder the experience.",
    
    "I've been a loyal user for years. Consistently meets expectations.",
    "App is not worth the price. Lacks essential features and functionality.",
    "Latest update introduced new features. Enhancing functionality.",
    "Interface is confusing. Consider a redesign for better user experience.",
    "Regret purchasing. Doesn't live up to promises.",
    "App's speed is impressive. Responds quickly to user commands.",
    "Subscription cost is too high for limited features provided.",
    "Recommend this app to everyone. User-friendly and feature-packed.",
    "Frequent crashes, especially when multitasking. Address urgently.",
    "App crashes every time I open it. Please fix this issue!",
    
    "App is a lifesaver! Incredibly useful and efficient.",
    "User interface is confusing. Please redesign for better usability.",
    "Great app, but it would be better with a dark mode option.",
    "App's performance is excellent. No issues so far.",
    "Intuitive app design. Easy to navigate and use.",
    "Unexpected crashes ruin the experience. Urgent fix required.",
    "The lack of a dark mode is disappointing. Consider adding it.",
    "This app is a lifesaver for managing daily tasks efficiently.",
    "Latest update significantly improved app speed. Impressive!",
    "Stylish and modern interface. One of my favorite apps.",
    
    "Detected a security vulnerability. Immediate attention needed.",
    "Amazing app! All the features I need for productivity.",
    "Fast and responsive. The app speed is exceptional.",
    "The app is overpriced for the limited features it offers.",
    "Frequent crashes hinder usability. Investigate and fix.",
    "Regular updates show dedication to improvement. Appreciate it.",
    "Exceeded my expectations. Feature-packed and user-friendly.",
    "User interface lacks essential functionalities. Needs improvement.",
    "Regret purchasing. The app does not meet its advertised capabilities.",
    "Frequent freezes affect performance. Optimize for better user experience.",
    
    "Impressed with the app's reliability. No issues encountered so far.",
    "Outdated design. A modern redesign would enhance the user experience.",
    "Crashes on startup. Immediate attention required.",
    "Must-have app for productivity. Simplifies tasks seamlessly.",
    "Decent app overall, but customer support is lacking.",
    "Detected a security vulnerability. Strengthen encryption measures.",
    "New features needed in the upcoming update.",
    "Simple and effective app. Ideal for daily use.",
    "Frequent crashes frustrate users. Investigate and fix the issue.",
    "Responsive and helpful customer support.",
    
    "Love the simplicity of the app. No unnecessary complications.",
    "A bug causing data loss needs immediate attention. Implement better error handling.",
    "Perfect app for my needs. Fast and sleek design.",
    "The latest update significantly improved performance. Great work!",
    "Sleek app design. Appreciate the modern look.",
    "I use this app daily, and it works very well for me.",
    "App performance improved after the recent update.",
    "A gem of an app. Simplifies complex tasks effortlessly.",
    "Not satisfied with the app's performance. Lags frequently.",
    "Highly recommend this app! User-friendly and feature-packed.",
    
    "Smooth and efficient app. No complaints so far.",
    "An essential tool for productivity. Highly recommend.",
    "Crashes frequently. Developers, please fix this issue.",
    "Intuitive interface. Easy to use for all skill levels.",
    "The latest update introduced unexpected bugs. Needs immediate attention.",
    "Great app overall, but the lack of customer support is disappointing.",
    "Impressive features, but the app is slow to load.",
    "User-friendly design. Perfect for managing daily tasks.",
    "Regular updates keep the app fresh and reliable.",
    "Security flaw detected. Please address for user safety.",
    
    "The app's simplicity is its strength. No unnecessary clutter.",
    "Unexpected crashes during crucial tasks. Urgent fix needed.",
    "This app is a game-changer. Simplifies complex processes.",
    "Frequent updates indicate active development. Thumbs up!",
    "Annoying ads detract from the user experience. Consider a paid option.",
    "Responsive customer support. Quick resolution to issues.",
    "Love the minimalist design. A pleasure to use daily.",
    "App performance is outstanding. No lags or slowdowns.",
    "Critical bug causing data loss. Immediate action required.",
    "This app does exactly what it promises. No complaints at all.",
    
    "App crashes upon startup. Cannot use it. Fix urgently needed.",
    "Innovative features set this app apart. Well done!",
    "The lack of dark mode is a downside. Please add this feature.",
    "Reliable app for day-to-day tasks. Highly recommended.",
    "App freezes during use. Makes it frustrating to navigate.",
    "Streamlined design enhances user experience. Thumbs up!",
    "Buggy and unreliable. Not recommended for serious use.",
    "The app has become a staple in my daily routine.",
    "Unexpected errors disrupt workflow. Developers, please investigate.",
    "Simple and effective. Does exactly what I need.",
    
    "App frequently forgets settings. Annoying and needs fixing.",
    "Great job on the recent update! Improved many aspects.",
    "Easy to use, but the lack of certain features is noticeable.",
    "The app lacks essential features. Consider adding more functionality.",
    "Crashes persist even after updates. Fix urgently needed.",
    "This app exceeded my expectations. Well worth the download.",
    "Lack of updates is concerning. Is the app still supported?",
    "User interface needs improvement. Too many clicks for basic functions.",
    "Fantastic app! No issues encountered so far.",
    "App crashes randomly. Frustrating and needs fixing.",
    
    "Sleek design and excellent functionality. Highly recommended.",
    "The latest update ruined the app's performance. Disappointed.",
    "Effective app for task management. No complaints from me.",
    "Crucial features missing. Please consider adding them.",
    "Frequent updates keep the app secure and reliable.",
    "App crashes when multitasking. Developers, please investigate.",
    "Highly intuitive design. A joy to use daily.",
    "Lags and slowdowns make the app frustrating to use.",
    "The lack of a search function is a major drawback. Consider adding it.",
    "App crashes during video playback. Urgent fix needed.",
    
    "The recent update improved app stability. Great work!",
    "Crashes persist after the latest update. Unusable.",
    "Love the new features introduced in the latest release.",
    "The app's performance has been exceptional. No complaints.",
    "An essential app for anyone seeking productivity tools.",
    "Buggy and unreliable. Constant crashes need fixing.",
    "Intuitive design makes the app easy to navigate.",
    "The lack of customer support is a significant drawback.",
    "Frequent updates keep the app secure and up-to-date.",
    "The latest version is a step backward. Many issues introduced.",
    
    "Great app for organizing tasks. A must-have for productivity.",
    "Security flaw detected. Immediate action required.",
    "Sleek design and efficient functionality. Highly recommended.",
    "Unexpected errors during use. Developers, please investigate.",
    "App performance is sluggish. Needs optimization.",
    "Annoying ads detract from the overall experience. Consider ad-free options.",
    "Simple and effective. Does exactly what it promises.",
    "Crashes upon opening. Cannot use it at all.",
    "The lack of dark mode is a major downside. Please add this feature.",
    "Fantastic app! Improved my workflow significantly.",
    
    "Unresponsive customer support. Frustrating experience.",
    "User-friendly design. A pleasure to use daily.",
    "Frequent crashes make the app frustrating to use.",
    "Regular updates enhance app functionality. Thumbs up!",
    "Effective app for managing daily tasks. No complaints.",
    "Lack of essential features. Needs improvement.",
    "The app has become an integral part of my routine.",
    "Innovative features set this app apart from others.",
    "Unexpected bugs after the latest update. Developers, please address.",
    "Outstanding app performance. No lag or slowdowns.",
    
    "Crucial features missing. Please consider adding them.",
    "App crashes during video playback. Urgent fix needed.",
    "Great job on the recent update! Improved many aspects.",
    "Lags and slowdowns make the app frustrating to use.",
    "User interface needs improvement. Too many clicks for basic functions.",
    "Sleek design and excellent functionality. Highly recommended.",
    "The latest update ruined the app's performance. Disappointed.",
    "Effective app for task management. No issues encountered.",
    "Crucial features missing. Developers, please take note.",
    "Responsive customer support. Quick resolution to issues.",
    
    "Love the minimalist design. A pleasure to use daily.",
    "App performance is outstanding. No lags or slowdowns.",
    "Critical bug causing data loss. Immediate action required.",
    "This app does exactly what it promises. No complaints at all.",
    "App crashes upon startup. Cannot use it. Fix urgently needed.",
    "Innovative features set this app apart. Well done!",
    "The lack of dark mode is a downside. Please add this feature.",
    "Reliable app for day-to-day tasks. Highly recommended.",
    "App freezes during use. Makes it frustrating to navigate.",
    "Streamlined design enhances user experience. Thumbs up!"     
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
          1, 1, 1, 1, 0, 1, 1, 1, 1, 1,       
          0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
          0, 1, 0, 1, 1, 1, 0, 0, 0, 0,
          0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
          0, 0, 1, 0, 0, 0, 1, 0, 0, 1,
          1, 1, 0, 0, 0, 0, 1, 0, 0, 0,
          0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
          1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
          1, 0, 1, 1, 1, 0, 0, 0, 0, 1,
          0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
          1, 0, 0, 1, 1, 1, 0, 0, 0, 0,        
          1, 0, 0, 0, 0, 0, 1, 0, 0, 0,
          1, 0, 1, 0, 0, 1, 0, 0, 0, 0,
          0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
          1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
          0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
          0, 1, 1, 0, 1, 0, 1, 0, 0, 1,
          0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
          0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 1, 0, 0, 0, 0, 0,          
          0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
          0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
          0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
          1, 0, 0, 0, 0, 0, 1, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
          0, 0, 1, 0, 0, 1, 0, 0, 1, 0,
          0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
          1, 0, 0, 0, 0, 0, 1, 0, 0, 1]


# Convert text data to TF-IDF features
tfidf_vectorizer = TfidfVectorizer(max_features=10000)  # You can adjust the number of features
X = tfidf_vectorizer.fit_transform(reviews)

# Create and train the SVM model
svm_classifier = SVC(kernel='linear', C=1, random_state=42)
svm_classifier.fit(X, labels)

# Actionable Snapchat Reviews to test
snapchat_reviews = [
    "I loved this app, but ever since the last update the filters don't work. Any filter that alters someone's face, I can take a picture but I can't take video anymore. The audio will come up normally and yet the video is like it's doing a slideshow, each frame holds for 5 seconds before switching to the next. Most of the app functions normally but filters that alter people's faces just will not work and it's driving me insane. They worked a couple weeks ago so what did you guys do????",
 
    "Please for the love of God fix the notification bug! It's been over a year and I STILL can't get notifications from people trying to message me. I may as well unfriend them so they don't think I'm ignoring them when it's been two weeks before I even notice it's there.. I have tried EVERYTHING from reinstalling, clearing the cache, turning off do not disturb, making sure notifications are on and at max volume. Nothing. It's utterly impossible to stay in touch with anyone on this platform.",
    
    "This app is really annoying. And it takes all of your data even if you set the permissions to not allow it into your contacts. It still takes it. And the notifications don't stop even if you set them to only one a day, the only option to not be annoyed with their notifications all day long is to turn them completely off, so then you don't know when your friends have sent you a message.",
    
    "My AI is so annoying. I've clicked on it by accident many times. There should be a free option to get rid of it. I don't think it's fair that iPhone users get dark mode for free, but android users have to pay for premium to have the option. Go back to the simplicity. I've had snap since 2013. It's a great way to send photos & videos to friends & family & keep in touch. The Discover Page & tiktok pages are also annoying. If i wanna watch tiktoks I'll go to tiktok where the videos are clear.",
    
    "This app is super buggy. Every time I add continuous video it either sends the cut clips in the wrong order, or duplicates the clip twice. Along side this I find the discover feed thing to be annoying, as well as the quick add feature. I feel like they're always trying to shove random people down my throat whenever I use the app. I just want you communicate with my friends. I wish there was an option to turn it off. The amount of times I almost sent personal stuff to random people is crazy.",
    
    "The advertisement of Snapchat+ is seriously invasive and annoying. I also find it hilarious that at first you had to pay for My AI, but now you have to pay to get rid of it. I also wish there was an option to turn off the discover page since I find it so distracting. The one thing I'll never stop complaining about is the new bitmojis. I don't think anyone was asking for creepy 3D ones, and the 2D ones were far more charming. It feels like you're trying to copy Meta (Facebook), disappointing.",
    
    "Absolutely terrible experience on Android. I have been a long-time user of this app on Android and have been waiting for the experience to improve. For one singular app version, it supported my 3x and 10x zoom cameras. I updated it a couple of days later, and it was just gone. They have proven that they have the ability for good camera integration but refuse to roll it out promptly. The fact that I use one of the most popular Android phones and they still can't get it right is astonishing to me.",
    
    "The updates are making this app fade into obscurity and is absolutely full of bugs, instability, loss of useful and popular features, totally bizarre and unusable touch screen controls that are not synced with the actual place where you are supposed to touch. On my s22ultra I have to hover a 1/4 of an inch above the icon for it to possibly work, but some on screen items will never be able to utilized from the ridiculous discrepancy of syncingthevm layout to the touchscreen.",
    
    "There is a bug where after you type a message onto a snap, it just disappears when you close the keyboard. I have reported this bug multiple times and it hasn't been fixed in months. I use the app a lot and it happens several times a day, for at least half of all snaps I send. It's extremely annoying, and it seems like there is no attempt to fix these bugs.",
    
    "it's a easy way to send pictures and talk to other people but now it's just getting to be too much, an off-brand tiktok page, the bitmoji updates are starting to look a bit weird, I just wish we're the way they used to be. The AI is getting really annoying too, of there was a option to remove it completely that would be nice because sometimes when I click on a completely different conversation, it open the AI. The AI part of snap is all making me phone glitch a lot, but it could just be me.",
    
    "Horeible. Why can't you fix your app so that when a person makes a continuous video. When you click on there page to get the second video you can't find it.!!! Can you please make it easier for user to find the continuous video. This really upsets me. I watch a video and they say they have to continue it in another video and we'll you can't find it. It makes me mad!!. Gosh make it easier for your users!! I will change ratings once it's done!!",
    
    "The app functionality has been alright, not really here to comment on that. The bitmojis..I did not see the need to update them again, and they look worse. The extra detail really isn't needed when the previous ones actually looked good. Plus, the way the new one looks while editing and how it stands is definitely worse as well. If you're going to continue updating them, please keep the stickers how they are with the previous ones. Or give us an option to choose our bitmoji design, please.",
    
    "What possess you to give a notification when someone is typing? Who asked for that? It's unnecessary and annoying. We don't need two notifications for every message received. The least you could do is allow us to turn it off if we don't want that. Addition: Now it keeps crashing after watching even a single story. Not to mention that if anyone posts while I'm viewing something, it starts glitching and crashes on itself. Snapchat is quickly becoming obsolete, and you only have yourself to blame.",
    
    "Had my account since February of 22, and I've noticed some things, one, in order for it to run smoothly on a android, you kinda have to have snap +, even then it's still buggy, the calls suck, and it freezes alot still, and sometimes you can't see/open snaps until hours after they've been sent. but onto positive stuff, it's a good way to keep in touch with family and friends and also to find and follow snap stars",
    
    "I was on snap for a pretty long time. I liked everything about it, it was very easy to use and you can chat with friends and it just has so many functions that just make it better. But now the AI is annoying me and I can't get rid of it, and my bitmoji scares me. It looks possessed and it makes me uncomfortable. Change the bitmojis back to the old style please! And get rid of this AI!",
    
    "The discover is terrible, the spotlight is also bad, the AI is just annoying. Any ads they give or notifications for new features just gets in the way, and doesn't go away for a long time. There's also no point in trying to maintain streaks, as Snapchat will just take it away randomly, and asks for you to pay to restore it. And there is Snapchat+, which anything that you think would be accessible to the basic user, it's not. And stories repeatedly say you haven't watched them, even tho you have.",
    
    "10/2022: the AI feature is something I will never use and there should be an option remove or block it. I also don't like the notifications from the discover part of the app. Not sure if there's a way to turn them off, haven't looked yet. 9/2018: Multiple times the app never told my friends and I that are streak is about to break until it's too late",
    
    "I have never used, nor will I use the My AI. It's a waste of space. I absolutely hate that there is no way to get rid of it or have the option of turning it off? It's really annoying.🚩 It's biased to only let iPhone users have dark mode instantly, and me, an Android user pay for it? If it's free on iOS, I want it to be free everywhere else, please.🚩 Aside, I think my favorite thing about the app is the sticker maker. I haven't made any new stickers recently, but tossing in customized faces.❤",
    
    "Snapchat is an amazing app to respond and socialize with friends, even when they are far away! The app has everything. Such as the stories feature, the 'tiktoks', pictures, filters, and even a map. But, I really despise the AI bot. It is kind of disturbing and weird. Recently, it posted a story on its page, which is absurd for it to happen. Honestly, Snapchat should just make it an option to have it 😊🥰.",
    
    "App is okay for quickly sharing non-important photos and videos, but it's seriously bloated with features that really aren't useful. The interface is a convoluted mess, and there's probably better options available. The most egregious addition was a My AI that is permanently pinned to the top of your messages/contacts. And you can not remove it unless you pay money. It's the tipping point for me, I've accidentally selected the AI so many times, and it's becoming incredibly frustrating.",

    "Used to be better. The Discover page is just clickbait and after you're done viewing your friends' stories it just plays the first thing on the page on its own. The 3D Bitmojis are kind of annoying, since there isn't an option to perfer showing the original ones. However, the addition of 5 streak restores per month is brilliant! It's very easy to do and a reasonable number of restores. Good job with that! When it comes to My AI, please stop forcing it. Give us the option to remove it. That's all",
    
    "Firstly, I have noticed that the search function on Snapchat is not loading properly. When I try to search for specific usernames or content, the search results take a long time to load, and sometimes they don't load at all. This has been happening consistently over the past few days.Secondly, I am facing difficulties with the sticker feature.When I try to add stickers to my Snaps, they are not appearing in the sticker drawer. I have tried reinstalling the app and updating to the latest version.",
    
    "I love this app, been using it for a long time. love communicating with friends and groups. The filters are awesome and it's overall a good experience. However, ever since they updated the bitmojis and made them 3D it looks cringy and has kind of ruined it for me. I loved the 2D ones though please change it back to that. Also you should be able to access snapchat from the web on phone, but u can only edit basic things. It's still a good app but they could improve.",
    
    "Snapchat is a popular social media app known for its variety of filters to click photos and videos. It offers a fun and interactive way to stay connected with friends through snaps and Stories. However, its user interface can be a bit confusing for newcomers, and privacy concerns have been raised in the past. Overall, Snapchat is great for sharing moments, but it may not be everyone's cup of tea.",
    
    "I lost my streaks even though I paid within the given time. I'm expecting to be given my money back, even if it is just 99p. Also, the AI chat is pretty pointless, and from what I know, you can't delete it. It can lag quite a lot too, especially when recording. However, it's alright for chatting and scrolling on videos, and the filters are fun.",
    
    "A FEW THINGS WRONG.... Every single filter freezes for a couple minutes when I try to take a video & it does it a couple times in a row before it goes back to normal, also sometimes the videos start out glitchy & slow.. Every single day for weeks now. Also, now the light switches to brighter for a second then goes back to normal when I'm taking a video too, I have to have my phone at a certain uncomfortable angle so the camera light doesn't do that.",
    
    "This app won't let me get past a pop-up window that comes up as soon as I open the app that states, We believe the camera should be optimized for each snapchatter that uses it. To do that Snap uses info about your face, hands, and voice to make certain features work learn more, and if you WANT to agree and continue tap below. I don't want to agree to this. But it won't let me click out of it or use the app in any way. Uninstalling...very frustrating. This is ridiculous.",
    
    "For the average young person, this app is perfect. In my opinion, it's the best way to communicate with people online. When it comes to the creator side, there's a lot that can be improved. The main thing that bugs me is how many of your personal Spotlights you're able to see under the Spotlight tab under the public profile. It often only shows the first few recent Spotlights you post, which becomes a hassle when trying to select which Spotlights appear on your profile.",
    
    "The application suffers from a significant number of glitches. When attempting to edit my snap and then add a song, upon returning, the entire experience becomes disorganized and chaotic. The recipient of my snap is presented with a distorted image that I never intended, thanks to these glitches. Moreover, any information I edit is inaccurately placed, and the app automatically selects presets or filters that I was merely checking. This disrupts the intended user experience.",
    
    "Too many bots/ spam/ inaccurate friend suggestions Useless features and too much push to promote paying for features. There should be a way to disable friend suggestions that appear beside stories since they are majority of the time people I have no idea who they are. There should be a way to keep 2 different profiles on a same device separate (including message requests and friend suggestions). AI should be able to have a disable/delete feature. Many flaws in app make me re-consider using it.",

    "Unless you plan to purchase a Snapchat+ plan, don't bother. Various features available to premium users will be advertised in big banners across the screen, impeding your ability to actually interact with the app. Want to click on a chat stream with some friends? Better do it quick, before the entire list is rearranged because they're trying to sell you some new feature! Don't want your messages screen cluttered up with their mediocre AI? Tough, you gotta pay to hide that.",
    
    "Inconvenient updates. Unnecessary changes that mean you can't always see content you follow. Way too many ads. Certainly not worth paying for. Content often freezes or gets stuck loading. Pics/videos sent from friends or family, you occasionally have to completely close out the app for them to pop up, or they never load. Developers keep removing helpful features and don't putting them back until after nearly a month, and they come back worse than before. Paid version not even worth free trial",
    
    "My camera is opening sideways and once I finish recording its upside down? and my messages on my snaps are deleting too so I have to retype everything again. The help center was no help as it was confusing and I was unable to find the issue I was looking for. Otherwise from this issue I would say this is great app to use to talk to friends and mess around with.",
    
    "It's a fun & easy way to take selfies & group pics to share with friends & family and to capture memorable moments bc you are able to download/save photos to your device. I love the filters! There must be 1,000s of filters & the app allows you to save your favs & share. There's a lot more you can do within the app if you take the time to explore.",
    
    "The bitmojis are getting worse and worse. Nobody even asked for a 3d bitmoji. It would be better if there was an option to have it 2d or 3d, but right now, it just looks horrible. The AI thing they introduced is also annoying, being at the top of the messages list and not being able to remove it despite its uselessness.",
    
    "Ever since the last update, the camera is in landscape mode as soon as I open the app, and there's no way to change it within the app itself. I have to either restart my phone, or uninstall and re-install the app, then it will open properly again, but only for a few uses before it opens in landscape mode again and I have to repeat the process.",
    
    "The AI is annoying and intrusive. there's no reason it should be forced into the top of ur friends list, and no reason why you can't delete it unless you PAY for it. I already thought locking pretty basic features behind a pay wall was cheap and dirty, but the AI makes it worse. if you don't have snapchat already-- don't get it.",
    
    "A relatively functional app that feels like it's pushing microtransactions as hard as possible. I am asked every time I open the app to join a group, advertised news media stories, and now their exclusive AI is always at the top of my page, not my most recent message. The only way that I can find to remove these features is to pay $3.50 a month and unlock the buttons to say no thank you.",
    
    "The very image of shallow consumerism. I only use it when my friends do, and it's under duress. Otherwise, it's a total waste of space. It attracts the dumbest people on earth, the ads are ridiculous and probably malware, the AI feature is moronic and intrusive, and no I will not pay for more mediocre features for a below-average app. Everything about it screams corporate suits trying to be hip and with it like they haven't replaced their humanity with bar graphs. Awful.",
    
    "Horrible, it's so laggy sometimes and when I try to edit a camera roll video it sends the videos in individual clips and then I look stupid so I have to delete all of them, and then when you send a video it makes you go back to text instead of leaving you on the camera roll so I have to go all the way back down to find something again",

    "DON'T SAVE VIDEOS OR PICTURES ON SNAPCHAT, After some time, the videos/fotos you save on snap, cant open anymore. So you lose them. This is not just with old videos you save but also with new videos. This is stupid. I use snapchat for 7 years, and I am disappointed with it. Thank me later.",
    
    "It helps you to connect with thousands of people across the world. And most Importantly AI (artificial intelligence) it helps an individual to get answers about any rising questions within a second. It is very really helpful for students, employees, etc. Enjoying with many more interesting lenses. Capturing every moment. I am having a great time using Snapchat, thankyou.",
    
    "Making the bitmojis and stickers 3d was an awful idea!👎 It doesn't seem fun anymore. All the reasom im using this app is for silly and fun interface and gimmicks of it. I think its the case for most other people too. Even besides that, the realistic ones look ugly. Dear team snapchat, please return them back to normal.",
    
    "Now you can't share can't export to your phone this is really really bad app This app got the worst music and never have they change them. On top of takes a lot of data just for opening it. I don't know why anything you do there goes data if not s little it's like 10 gig that's why I shot using it. Cause it makes no sense then there the update Why update if there's nothing new",
    
    "Cash grab garbage. I don't watch videos on the app because why would I ever? But my buddy tried to show me a short video and it was riddled with ads every 30 seconds. And then My AI. The feature a minority of people may have wanted. And if you didn't want it, too bad! The chat stays in your feed. BUT WAIT! We'll give you the ability to hide it if you give us money and get Snapchat+! Literally the worst business decision I've seen for a mobile app. Total garbage.",
    
    "Something changed recently to where when you post a memory to your story anyone viewing cant see what date/time its from. It only says how long ago it was added to the story. Ex: If I post an OLD memory it says 1 hour ago instead of August 14th 2021 in the top left under the users name. I find it frustrating because when I want to post a memory I want to also include when it's from.",
    
    "The app was great before the new updates, since the new 3D bitmojis the app doesn't look great nor unique it's basic and annoying and the streaks are getting lost even though we send before the 24-hours max and you have to be subscribed to save them or restore them which is annoying, please bring back the old bitmojis back",
    
    "HAS WAY TO MUCH ADS. Each time I'm watching Snapchat I get at least 1 minute each video and get another ad. It's so frustrating. BUT NOW. They added a new feature that makes your avatar look realistic it's so ugly please change it back to normal I hate it and I wanna delete this app",
    
    "Since the new update I am facing difficulties with the sticker feature. When I try and add stickers to my Snapchat, they are not appearing in the sticker drawer section. I am also having difficulty with searching for any specific names or content, it takes a really long time to load or it doesn't load at all. I have checked all settings and everything is fine and I have deleted the app and reinstalled it and still not working so It's very frustrating for me and would like for this to be sorted",
    
    "It's a little bit unfair that android users don't get a chance to set their chats to dark theme. You know, one thing I absolutely hate about this app is that AI. The fact that we can't unfriend it. It barely gives me correct answers and just keeps repeating its answers... I have updated the app so many times, but I still don't get the features I updated it for. Honestly, I might uninstall this app. When I updated the app, I wasn't able to save stories in chat anymore. The app is getting worse.",
    
    "Snapchat on Android is a disappointment. The app is buggy, crashes frequently, and the camera quality is terrible. It lags during snaps, making the experience frustrating. Updates seem to make it worse, not better. Overall, it is a subpar and frustrating social media platform. Update... App continues to get worse with bugs and even more things that many users dislike, such as the new bitmojis.",
    
    "I love Snapchat Always been a big fan of it. You can take pictures & videos & save those memories into your phone & into the app. You control who adds you & who sees your posts. It's a great way to connect with old friends & family & make silly memories especially with the funny filters. I just wish snap would add back the games because those games were fun..I don't really like the Snapchat+.",
    
    "Recent photos (from every other application) now in the way when trying to take a snap; could just upload from Memories under camera roll? Moving the emojis to the left under friend icons creates too much clutter on the left side and is hard to look at. Please stop changing the UI you keep making it worse.",
    
    "Snapchat is good for straightforward chatting, but it has gotten worse lately. Features that you usually would need isn't there, like replay YOUR OWN snap... and it keeps getting changed! I would delete it if I could, but so many people are using it. Not email or something.",
    
    "The app would be much improved if it had a dark mode NOT BEHIND A PAY WALL. As well as more options for length of time messages are kept and not allowing them to be saved in chat. The filters are ok and the stories are ok. DARK MODE SHOULD NOT BE A PAY WALL.",
    
    "Hey I was using nothing Phone 2 and it seems the app is not optimised for the device as the snaps are having too much processing. Moreover there was a glitch with the night mode it was there when I first installed the app after that the night mode option is not showing at all. To bring it again I have to reinstall the app then it will show but it will only be for the 1st time I open the app.",
    
    "I love using Snapchat to talk to my friends and engage with others. The only gripe I have currently is somewhat of a bug. People complained when the back button was moved next to the call button and then it was moved back to the other side. Mine is still on the right side near the call but my other friends have it on the opposite side. If I can somehow fix this, I'll change my rating to 5 stars 🌟.",
    
    "Banned for trying to log in, my wifi is slow, especially when everyone is on it. I was trying to get I to Snapchat to send my streaks, and due to slow wifi, j was unable to get in, it would just load. I exited and reentered the app a couple times, when I tried logging in the next day, it said my access was temporarily blocked due to multiple failed attempts to log in. I left it sit for over 24 hours, tried logging in and got the same message. It has now been over 2 days, and I am still banned.",
    
    "One of the most grippling application as there are multifarious filters along with emojis.However,we can not only stay in touch with our friends but also our celebrities whose have an account on that application and the unique thing about this app it gives a lot of functions in one platform such as stories,function of: streaks and spotlight ,AI. Having a chat with AI such an excellent thing as we can ask anything whether it could be about our profession or knowledge. So users should go with it.",
    
    "Great app. I text my girlfriend on it, but these past two days, it keeps saying sending on my messages. It never sends them, and when someone texts me, it get the notification about it, but the chat between me and the person doesn't show anything new. I know it's not a problem with my data, because I can launch any other app perfectly fine. It can't be a problem with my account either, because I can still log in when I'm connected to WiFi. Please solve this.",
    
    "I really love this app, I can message my friends, there are so many different filters, I can watch as well and I have maps. This app is really good. The only thing that is really annoying is the new update, it has made the avatars so bad. Especially the facial features such as eyes, facial shape and more. It's really annoying to look at the avatars of my friends in chat. This app would be 5 stars if the update would be removed. I think that there should be an option between 2D and 3D avatars.",
    
    "Stops in the mid of videos and doesn't even show them, just has a loading symbol. 😒 this day and age you'd think people would be smart enough to make a simple app. Update**.. sometimes it takes a long time to upload a video.. and when trying to view my memories it doesn't load. And when I went to look at my memories b4, it showed me pics and videos I didn't even upload!!! Wonder if I can do something bout that. Uploading private pics and vids without my consent 🤔😤😤😡.",
    
    "Camera won't orient itself upright. Takes pictures as if I'm holding the phone the opposite way that I am holding it. (portrait vs landscape orientation) the filters have also stopped working. What is the issue?",
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