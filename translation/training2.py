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
target_max_length, input_max_length = target_tensor.shape[1], input_tensor.shape[1]
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

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

print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))


def make_encoder_model(vocab_size, embedding_dim, enc_units, batch_sz):
  embedding = layers.Embedding(vocab_size, embedding_dim)
  gru = layers.GRU(enc_units,
    return_sequences=True,
    return_state=True,
    recurrent_initializer='glorot_uniform'
  )
  inputs = layers.Input(shape=(None, vocab_size))
  hidden = layers.Input(shape=(batch_sz, enc_units))
  x = embedding(inputs)
  print(gru.get_initial_state(x))
  print(x.shape)
  print(hidden.shape)
  output, state = gru(x, initial_state=hidden)
  model = tf.keras.Model(inputs=[inputs, hidden], outputs=[output, state], name='encoder')
  return model

def make_decoder_model(vocab_size, embedding_dim, dec_units, batch_sz):
  # attention = BahdanauAttention(dec_units)
  embedding = layers.Embedding(vocab_size, embedding_dim)
  gru = layers.GRU(dec_units,
    return_sequences=True,
    return_state=True,
    recurrent_initializer='glorot_uniform'
  )
  fc = layers.Dense(vocab_size)

  inputs = layers.Input(shape=(None, vocab_size))
  hidden = layers.Input(shape=(batch_sz, dec_units))
  x = embedding(inputs)
  # context_vector, attention_weights = attention(hidden, enc_output)
  #x = tf.concat([tf.expand_dims(context_vector, 1), inputs], axis=-1)
  output, state = gru(x, initial_state=hidden)
  output = tf.reshape(output, (-1, output.shape[2]))
  output = fc(output)
  # model = tf.keras.Model(inputs=inputs, outputs=[output, state, attention_weights], name='decoder')
  model = tf.keras.Model(inputs=[inputs, hidden], outputs=[output, state], name='decoder')
  return model

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

@tf.function
def train_step(inp, targ, enc_hidden):
  loss = 0
  with tf.GradientTape() as tape:
    _, enc_hidden = encoder(inp, enc_hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([target_lang.word_index['<start>']] * BATCH_SIZE, 1)

    for t in range(1, targ.shape[1]):
      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden) # target is next input
      loss += loss_function(targ[:, t], predictions)
      dec_input = tf.expand_dims(targ[:, t], 1)

  batch_loss = (loss / int(targ.shape[1]))
  variables = encoder.trainable_variables + decoder.trainable_variables
  gradients = tape.gradient(loss, variables)
  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss

encoder = make_encoder_model(vocab_input_size, embedding_dim, units, BATCH_SIZE)
decoder = make_decoder_model(vocab_target_size, embedding_dim, units, BATCH_SIZE)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

checkpoint_dir = './training_checkpoints'
checkpoint = tf.train.Checkpoint(
  step=tf.Variable(0),
  optimizer=optimizer,
  encoder=encoder,
  decoder=decoder
)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3, checkpoint_name="checkpoint")

encoder.summary()
decoder.summary()

# training
checkpoint.restore(manager.latest_checkpoint)
time.sleep(5)
if manager.latest_checkpoint:
  print("Restored from {}".format(manager.latest_checkpoint))
else:
  print("Initializing from scratch.")
for epoch in range(EPOCHS - int(checkpoint.step)):
  start = time.time()
  checkpoint.step.assign_add(1)

  enc_hidden = tf.zeros((BATCH_SIZE, units))
  print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
  print(enc_hidden.shape)
  total_loss = 0

  for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
    batch_loss = train_step(inp, targ, enc_hidden)
    total_loss += batch_loss

    if batch % 100 == 0:
      print(f'Epoch {int(checkpoint.step)} Batch {batch} Loss {batch_loss.numpy():.4f}')
  # saving (checkpoint) the model every 2 epochs
  if (epoch + 1) % 2 == 0:
    manager.save()
    print("Saved checkpoint for step {}: {}".format(int(checkpoint.step), checkpoint_dir))

  print(f'Epoch {int(checkpoint.step)} Loss {total_loss/steps_per_epoch:.4f}')
  print(f'Time taken for 1 epoch {time.time()-start:.2f} sec\n')

encoder.save("./models/encoder.h5")
decoder.save("./models/decoder.h5")