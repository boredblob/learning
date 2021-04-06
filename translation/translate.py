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
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

encoder = tf.keras.models.load_model('./models/encoder.h5', compile=False)
decoder = tf.keras.models.load_model('./models/decoder.h5', compile=False)

def evaluate(sentence):
  sentence = preprocess(sentence)

  inputs = [input_lang.word_index[i] for i in sentence.split(' ')]
  inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=input_max_length, padding='post')
  print(inputs)
  inputs = tf.convert_to_tensor(inputs)

  result = ''

  hidden = tf.zeros((1, units))
  _, enc_hidden = encoder([inputs, hidden])

  dec_hidden = enc_hidden
  dec_input = tf.expand_dims([target_lang.word_index['<start>']], 0)

  for t in range(target_max_length):
    predictions, dec_hidden = decoder([dec_input, dec_hidden])
    predicted_id = tf.argmax(predictions[0,0]).numpy()
    print(predicted_id)

    if target_lang.index_word[predicted_id] == '<end>':
      return result, sentence

    result += target_lang.index_word[predicted_id] + ' '


    # the predicted ID is fed back into the model
    dec_input = tf.expand_dims([predicted_id], 0)

  return result, sentence

def translate(sentence):
  result, sentence = evaluate(sentence)

  print('Input:', sentence)
  print('Predicted translation:', result)

translate(input("Enter phrase\n"))
print("\n")