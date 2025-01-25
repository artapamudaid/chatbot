from flask import Flask, render_template, request
import nltk
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Function to fetch training data from JSON
def fetch_training_data():
    with open('data/responses.json', 'r') as file:
        data = json.load(file)
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
    # Convert to lowercase
    text = text.lower()

    text = re.sub(r"(ass?alam(u'?alaikum)?)", "salam", text)
    
    return text

# Function to get response from the model
def chatbot_response(text):
    if model and vectorizer:
        # Preprocess the input text to handle variations
        text_preprocessed = preprocess_text(text)

        # Vectorize the preprocessed text
        text_vectorized = vectorizer.transform([text_preprocessed])
        
        # Predict response from model
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
    nltk.download('punkt')
    nltk.download('stopwords')
    app.run(debug=True)
