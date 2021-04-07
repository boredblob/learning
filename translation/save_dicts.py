import tensorflow as tf
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os
import io
import time

dataset_path = "./fra.txt"

def unicode_to_ascii(s):
  return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def preprocess(w):
  w = unicode_to_ascii(w.lower().strip())
  w = re.sub(r"([?.!,¿¡])", r" \1 ", w)
  w = re.sub(r'[" "]+', " ", w) # putting a space between words and punctuation
  w = re.sub(r"[^a-zA-Z?.!,¿¡]+", " ", w) # replacing unwanted chars with white space

  w = w.strip()
  w = '<start> ' + w + ' <end>'
  return w

def create_dataset(path, num_examples):
  lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
  word_pairs = [[preprocess(w) for w in line.split('\t')[:2]] for line in lines[:num_examples]]
  return zip(*word_pairs)

def tokenise(lang):
  lang_tokeniser = tf.keras.preprocessing.text.Tokenizer(filters='')
  lang_tokeniser.fit_on_texts(lang)

  tensor = lang_tokeniser.texts_to_sequences(lang)
  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

  return tensor, lang_tokeniser

def load_dataset(path, num_examples=None):
  target_lang, input_lang = create_dataset(path, num_examples)

  input_tensor, input_lang_tokeniser = tokenise(input_lang)
  target_tensor, target_lang_tokeniser = tokenise(target_lang)

  return input_tensor, target_tensor, input_lang_tokeniser, target_lang_tokeniser

num_examples = 500000
units = 1024
input_tensor, target_tensor, input_lang, target_lang = load_dataset(dataset_path, num_examples)
target_max_length, input_max_length = target_tensor.shape[1], input_tensor.shape[1]

with open('./models/dicts/input_wi.json', 'w') as f:
  print(input_lang.word_index, file=f)
with open('./models/dicts/input_iw.json', 'w') as f:
  print(input_lang.index_word, file=f)
with open('./models/dicts/target_wi.json', 'w') as f:
  print(target_lang.word_index, file=f)
with open('./models/dicts/target_iw.json', 'w') as f:
  print(target_lang.index_word, file=f)

print("\n")
print(input_max_length)
print(target_max_length)