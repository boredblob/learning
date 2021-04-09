import tensorflow as tf
from tensorflow.keras import layers
import time
import io
import re
from sklearn.model_selection import train_test_split

# "[^ a-zA-Z0-9\t.,?!'-’àâéèêïœôç ]+"
chars = "abcdefghijklmnopqrstuvwxyz0123456789àâéèêïœôç.,?!'’- "
dataset_path = "./fra.txt"

def preprocess(w):
  w = w.lower().strip()
  w = re.sub(r'[ ]+', " ", w)
  return w

def create_dataset(path, num_examples):
  lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
  sentence_pairs = [[preprocess(w) for w in line.split('\t')[:2]] for line in lines[:num_examples]]
  return zip(*sentence_pairs)

def tokenise(lang):
  lang_tokeniser = tf.keras.preprocessing.text.Tokenizer(filters='', char_level=True)
  lang_tokeniser.fit_on_texts(lang)

  tensor = lang_tokeniser.texts_to_sequences(lang)
  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

  return tensor, lang_tokeniser

def load_dataset(path, num_examples=None):
  input_lang, target_lang = create_dataset(path, num_examples)

  input_tensor, input_lang_tokeniser = tokenise(input_lang)
  target_tensor, target_lang_tokeniser = tokenise(target_lang)

  return input_tensor, target_tensor, input_lang_tokeniser, target_lang_tokeniser

num_examples = 1000
input_tensor, target_tensor, input_lang, target_lang = load_dataset(dataset_path, num_examples)
target_max_length, input_max_length = target_tensor.shape[1], input_tensor.shape[1]
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

BUFFER_SIZE = len(input_tensor_train)
units = 1024
embedding_dim = 20
batch_size = 64
epochs = 20
steps_per_epoch = len(input_tensor_train)//batch_size
input_vocab_size = len(input_lang.word_index) + 1
target_vocab_size = len(target_lang.word_index) + 1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(batch_size, drop_remainder=True)

print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))

def create_encoder_model():
  embedding = layers.Embedding(input_vocab_size, embedding_dim)
  gru = layers.GRU(units,
    return_sequences=True,
    return_state=True,
    recurrent_initializer='glorot_uniform',
    reset_after=False
  )

  inputs = layers.Input((None,), dtype='int32')
  hidden = layers.Input((units,), dtype='float32')

  input_embeddings = embedding(inputs)
  output, state = gru(input_embeddings, initial_state=hidden)
  model = tf.keras.Model(inputs=[inputs, hidden], outputs=[output, state], name='encoder')
  return model

def create_decoder_model():
  embedding = layers.Embedding(target_vocab_size, embedding_dim)
  attention = layers.AdditiveAttention()
  gru = layers.GRU(units,
    return_sequences=True,
    return_state=True,
    recurrent_initializer='glorot_uniform',
    reset_after=False
  )
  pooling = layers.GlobalAveragePooling1D()
  reshape = layers.Reshape((1, units))
  dense = layers.Dense(units)

  inputs = layers.Input((None,), dtype='int32')
  hidden = layers.Input((units,), dtype='float32')
  encoder_output = layers.Input((None, units), dtype='float32')

  input_embeddings = embedding(inputs)
  context_vector = attention([hidden, encoder_output])
  context_vector = pooling(context_vector)
  context_vector = reshape(context_vector)
  x = layers.Concatenate(axis=-1)([context_vector, input_embeddings])

  output, state = gru(x, initial_state=hidden)
  output = dense(output)
  model = tf.keras.Model(inputs=[inputs, hidden, encoder_output], outputs=[output, state], name='decoder')
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
    enc_output, enc_hidden = encoder([inp, enc_hidden])
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims(targ[:, 0], 1)

    for t in range(1, targ.shape[1]):
      predictions, dec_hidden = decoder([dec_input, dec_hidden, enc_output]) # target is next input
      loss += loss_function(targ[:, t], predictions)
      dec_input = tf.expand_dims(targ[:, t], 1)

  batch_loss = (loss / int(targ.shape[1]))
  variables = encoder.trainable_variables + decoder.trainable_variables
  gradients = tape.gradient(loss, variables)
  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss

encoder = create_encoder_model()
decoder = create_decoder_model()

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

# training
checkpoint.restore(manager.latest_checkpoint)
time.sleep(5)
if manager.latest_checkpoint:
  print("Restored from {}".format(manager.latest_checkpoint))
else:
  print("Initializing from scratch.")
for epoch in range(epochs - int(checkpoint.step)):
  start = time.time()
  checkpoint.step.assign_add(1)

  enc_hidden = tf.zeros((batch_size, units))
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