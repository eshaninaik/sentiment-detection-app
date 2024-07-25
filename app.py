from flask import Flask, request, jsonify, render_template
import pickle
import re
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

app = Flask(__name__)

# Load the model and vectorizer
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    text = re.sub(r'\d+', '', text)
    text = ' '.join(word for word in text.split() if word not in stopwords.words('english'))
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data['text']
    preprocessed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([preprocessed_text])
    prediction = model.predict(vectorized_text)[0]
    return jsonify({'sentiment': prediction})

if __name__ == '__main__':
    app.run(debug=True)
