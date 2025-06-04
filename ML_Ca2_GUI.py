import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tkinter import *
from tkinter import messagebox

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load dataset
data = pd.read_csv(r"C:\Users\Dipali\Downloads\twitter_training.csv (1)\twitter_training.csv", header=None)

data.columns = ['ID', 'Topic', 'Sentiment', 'Tweet']

# Text preprocessing function
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)

# Preprocess tweets
data['clean_text'] = data['Tweet'].astype(str).apply(preprocess)

# Feature extraction and label encoding
X = data['clean_text']
y = data['Sentiment']

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train-test split and model training
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Accuracy calculation
accuracy = accuracy_score(y_test, model.predict(X_test))

# GUI Functions
def analyze_sentiment():
    user_input = entry.get()
    if not user_input:
        messagebox.showwarning("Input Error", "Please enter a sentence.")
        return
    cleaned = preprocess(user_input)
    vec_input = vectorizer.transform([cleaned])
    prediction = model.predict(vec_input)[0]
    result_label.config(text=f"Predicted Sentiment: {prediction}", fg="blue")

def clear_input():
    entry.delete(0, END)
    result_label.config(text="")

# GUI Setup
window = Tk()
window.title("Twitter Sentiment Analysis - Logistic Regression")
window.geometry("600x400")
window.config(bg="#e6f2ff")

Label(window, text="Enter Tweet Text:", font=("Helvetica", 14), bg="#e6f2ff").pack(pady=10)
entry = Entry(window, width=70, font=("Helvetica", 12))
entry.pack(pady=10)

Button(window, text="Analyze Sentiment", command=analyze_sentiment, font=("Helvetica", 12), bg="#80c1ff").pack(pady=10)
Button(window, text="Clear", command=clear_input, font=("Helvetica", 12), bg="#ff9999").pack(pady=5)

result_label = Label(window, text="", font=("Helvetica", 14), bg="#e6f2ff")
result_label.pack(pady=10)

Label(window, text=f"Model Accuracy: {accuracy * 100:.2f}%", font=("Helvetica", 12), bg="#e6f2ff", fg="green").pack(pady=10)

window.mainloop()
