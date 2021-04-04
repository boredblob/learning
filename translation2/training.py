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
input_tensor, target_tensor, input_lang, target_lang = load_dataset(dataset_path, num_examples)

BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
EPOCHS = 20
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_input_size = len(input_lang.word_index)+1
vocab_target_size = len(target_lang.word_index)+1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

encoder_inputs = Input(shape=(None, vocab_input_size))
encoder = layers.GRU(embedding_dim, return_state=True)
encoder_outputs, state_h = encoder(encoder_inputs)

decoder_inputs = Input(shape=(None, vocab_target_size))
decoder_gru = layers.GRU(embedding_dim, return_sequences=True)
decoder_outputs = decoder_gru(decoder_inputs, initial_state=state_h)
decoder_dense = Dense(vocab_target_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit(
  [input_tensor, target_tensor],
  target_tensor,
  batch_size=batch_size,
  epochs=epochs,
  validation_split=0.2
)

tf.resha