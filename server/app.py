import hug
import pickle
import re

import tensorflow_hub as hub
import tensorflow as tf
import numpy as np

from collections import OrderedDict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"

# Create graph and finalize (finalizing optional but recommended).
g = tf.Graph()
with g.as_default():
    # We will be feeding 1D tensors of text into the graph.
    text_input = tf.placeholder(dtype=tf.string, shape=[None])
    embed = hub.Module(module_url, trainable=True)
    embedded_text = embed(text_input)
    init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
g.finalize()

# Create session and initialize.
session = tf.Session(graph=g)
session.run(init_op)

analyzer = SentimentIntensityAnalyzer()

with open('/usr/src/app/sentiment/models/pickles/SVC.pickle', 'rb') as f:
    clf = pickle.load(f)

def clean_tweet(text):
    cleaned = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())
    return cleaned

def combined_embeddings(text):
    use_embeddings = session.run(embedded_text, feed_dict={text_input: [text]}).tolist()
    vs = analyzer.polarity_scores(text)
    vs = OrderedDict(sorted(vs.items()))
    vader_embeddings = list(vs.values())
    combined_embeddings = np.append(use_embeddings, [vader_embeddings]).reshape(1, -1)
    return combined_embeddings

@hug.get('/health')
def health():
    """Checks status of application"""
    return "OK"

@hug.post('/predict')
def predict(body):
    """Predicts probabilities of class of text as positive, negative, neutral, not_relevant"""
    text = clean_tweet(body['text'].decode('utf-8'))
    embeddings = combined_embeddings(text)
    prediction = clf.predict_proba(embeddings)
    probabilities = dict(zip(['negative', 'neutral', 'positive', 'not_relevant'], prediction[0]))
    predicted_class = max(probabilities, key=probabilities.get)
    return{
        'predicted class': predicted_class,
        'class probability': probabilities[predicted_class],
        'probabilities': probabilities
    }

