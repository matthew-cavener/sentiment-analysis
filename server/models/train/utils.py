import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

import re

from collections import OrderedDict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def generate_hub_embeddings(text, module):
    if module == 'USE':
        module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
    elif module == 'ELMo':
        module_url = "https://tfhub.dev/google/elmo/2"
    embed = hub.Module(module_url, trainable=True)
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        embeddings = session.run(embed(text))
    return embeddings

def generate_vader_embeddings(text):
    analyzer = SentimentIntensityAnalyzer()
    vader_embeddings = []
    for sample in text:
        vs = analyzer.polarity_scores(sample)
        vs = OrderedDict(sorted(vs.items()))
        vader_embeddings.append(list(vs.values()))
    return vader_embeddings

def combined_embeddings(text, module):
    hub_embeddings = generate_hub_embeddings(text, module)
    vader_embeddings = generate_vader_embeddings(text)
    combined_embeddings = np.append(hub_embeddings, vader_embeddings, axis=1)
    return combined_embeddings

def clean_tweet(tweet):
    cleaned = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())
    return cleaned

def clean_sentiment(sentiment):
    if sentiment == 'not_relevant':
        return -1
    return int(sentiment)

def read_data():
    return pd.read_csv('/usr/src/app/sentiment/data/Apple-Twitter-Sentiment-DFE.csv',
                        encoding='latin-1',
                        usecols=['text', 'sentiment'],
                        converters={'text': clean_tweet})