import hug

import tensorflow_hub as hub
import tensorflow as tf

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

@hug.get('/health')
def health():
    """Checks status of application"""
    return "OK"

