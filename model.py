import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords
import re
import pickle

nltk.download('stopwords')

data = {
    'text': [
        'I love this product! It works great and is exactly what I needed.',
        'Terrible service. I am very disappointed.',
        'Not bad, could be better.',
        'Absolutely fantastic! Exceeded my expectations.',
        'It is okay, nothing special.',
        'Worst experience ever. Will not be buying again.',
        'Decent product for the price.',
        'Horrible! Do not waste your money.',
        'I am satisfied with the purchase.',
        'Great value for the money.'
    ],
    'sentiment': [
        'positive',
        'negative',
        'neutral',
        'positive',
        'neutral',
        'negative',
        'neutral',
        'negative',
        'positive',
        'positive'
    ]
}

df = pd.DataFrame(data)

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    text = re.sub(r'\d+', '', text)
    text = ' '.join(word for word in text.split() if word not in stopwords.words('english'))
    return text

df['text'] = df['text'].apply(preprocess_text)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['sentiment']

model = LogisticRegression()
model.fit(X, y)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
