"""
script to download tf hub module to be cached before starting server.
"""
import tensorflow_hub as hub

module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
embed = hub.Module(module_url)