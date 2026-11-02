
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

data = {
    "message": [
        "Congratulations you won a free prize",
        "Claim your free cash reward now",
        "You have won a lottery",
        "Click here to claim your gift",
        "Win a free iPhone today",
        "Urgent claim your reward now",
        "You won free tickets",
        "Get free money by clicking this link",
        "Limited offer win cash now",
        "Congratulations claim your prize",
        "Hey are we meeting today",
        "Please call me when you are free",
        "Can you send me the notes",
        "Your assignment is due tomorrow",
        "I will reach home by evening",
        "Are you coming to college today",
        "Let's have lunch together",
        "Please check your email",
        "Happy birthday have a great day",
        "I will call you later",
        "Meeting starts at ten",
        "Don't forget to bring your notebook",
        "Thanks for helping me yesterday",
        "Where are you right now",
        "I have completed the project",
        "Can we talk after class",
        "Your package will arrive tomorrow",
        "The lecture starts in five minutes",
        "Please share the location",
        "See you tomorrow morning",
        "Free entry in a weekly prize draw",
        "You have been selected for a cash reward",
        "Click now to get a free voucher",
        "Claim your exclusive bonus today",
        "Win big prizes with this special offer",
        "Congratulations you are our lucky winner",
        "Get your free gift card now",
        "You have won a free vacation",
        "Earn cash instantly click here",
        "Exclusive deal claim your reward",
        "Can you help me with this question",
        "I am waiting outside the library",
        "The class has been cancelled today",
        "Send me the homework when possible",
        "I will be late for the meeting",
        "Did you eat lunch",
        "Let's study together this evening",
        "Please pick up some groceries",
        "I reached college safely",
        "Thank you for your message"
    ],
    "label": [
        "spam", "spam", "spam", "spam", "spam",
        "spam", "spam", "spam", "spam", "spam",
        "ham", "ham", "ham", "ham", "ham",
        "ham", "ham", "ham", "ham", "ham",
        "ham", "ham", "ham", "ham", "ham",
        "ham", "ham", "ham", "ham", "ham",
        "spam", "spam", "spam", "spam", "spam",
        "spam", "spam", "spam", "spam", "spam",
        "ham", "ham", "ham", "ham", "ham",
        "ham", "ham", "ham", "ham", "ham"
    ]
}

df = pd.DataFrame(data)

X_train, X_test, y_train, y_test = train_test_split(
    df["message"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

with open("spam_model.pkl", "wb") as file:
    pickle.dump(model, file)

print(f"Model trained successfully!")
print(f"Test accuracy on this small sample: {accuracy:.2%}")
print("Saved model as spam_model.pkl")
