from flask import Flask, render_template, request
import mysql.connector
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Function to fetch training data from MySQL
def fetch_training_data():
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='chatbot_db'
    )
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT keyword, response FROM responses")
    data = cursor.fetchall()
    conn.close()
    return data

# Prepare the Naive Bayes model with TF-IDF Vectorizer
def train_model():
    data = fetch_training_data()
    X = [item['keyword'] for item in data]
    y = [item['response'] for item in data]

    vectorizer = TfidfVectorizer()
    X_vectorized = vectorizer.fit_transform(X)

    model = MultinomialNB()
    model.fit(X_vectorized, y)

    return model, vectorizer

model, vectorizer = train_model()

# Function to preprocess text (e.g., remove punctuation, convert to lowercase)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"(ass?alam(u'?alaikum)?)", "salam", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    return text

# Function to get response from the model
def chatbot_response(text):
    if model and vectorizer:
        text_preprocessed = preprocess_text(text)
        text_vectorized = vectorizer.transform([text_preprocessed])
        response = model.predict(text_vectorized)
        return response[0]
    else:
        return 'Maaf. Saya Tidak Tahu.'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get', methods=['POST'])
def get_bot_response():
    userText = request.form['msg']
    return chatbot_response(userText)

if __name__ == "__main__":
    app.run(debug=True)
