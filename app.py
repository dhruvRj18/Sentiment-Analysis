from flask import Flask, render_template, request
import joblib
import numpy as np
from gensim.models.doc2vec import Doc2Vec
import demoji
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

demoji.download_codes()
nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__)

doc2vec_model = Doc2Vec.load('./data/doc2vec_model.pkl')
xgboost_model = joblib.load('./data/xgboost_model.pkl')
svc_model = joblib.load('./data/svc_model.pkl')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        text_content = request.form['text_content']

        if text_content:
            text_vector = infer_doc2vec_vector(text_content)
            sentiment_label = predict_sentiment(text_vector)

            return render_template('result.html', sentiment=sentiment_label)

    return render_template('index.html')


def preprocess_text(text):
    text = text.lower()

    text = re.sub(r'https?:\/\/\S+', '', text)
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    text = re.sub(r'[^A-Za-z0-9\s]+', '', text)
    text = demoji.replace(text, '')

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    preprocessed_text = ' '.join(tokens)

    return preprocessed_text


def infer_doc2vec_vector(text_content):
    words = text_content.split()
    return doc2vec_model.infer_vector(words)


def decode_sentiment(sentiment_label):
    if sentiment_label == 0:
        return "Negative"
    elif sentiment_label == 1:
        return "Neutral"
    elif sentiment_label == 2:
        return "Positive"
    else:
        return "Unknown"


def predict_sentiment(text_vector):
    text_vector = np.array(text_vector).reshape(1, -1)
    sentiment_label = svc_model.predict(text_vector)[0]

    return decode_sentiment(sentiment_label)


if __name__ == '__main__':
    app.run(debug=True)
