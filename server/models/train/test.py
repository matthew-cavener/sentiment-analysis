import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

import re

from collections import OrderedDict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def generate_elmo_embeddings(text):
    module_url = "https://tfhub.dev/google/elmo/2"
    embed = hub.Module(module_url, trainable=True)
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        embeddings = session.run(embed(text))
    return embeddings

text = [
    'I have 2 cats',
    'I enjoy development'
]

print(generate_elmo_embeddings(text).shape)