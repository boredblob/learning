import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import glob
from IPython import display

BUFFER_SIZE = 70000
BATCH_SIZE = 32
EPOCHS = 300
noise_dim = 100
example_num = 16
example_seed = tf.random.normal([example_num, noise_dim])
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
#   './ffhq/thumbnails128x128/',
#   seed=123,
#   image_size=(128, 128),
#   batch_size=BATCH_SIZE
# )
# train_dataset = train_dataset.shuffle(1000)

def load_image(path):
  img = tf.io.read_file(path)
  img = tf.io.decode_png(img, channels=3)
  img = tf.cast(img, tf.float32)
  img = (img - 127.5) / 127.5 # Normalise to [-1, 1]
  return img

list_ds = tf.data.Dataset.list_files('./ffhq/thumbnails128x128/*/*', shuffle=False)
# list_ds = list_ds.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)
train_dataset = list_ds.map(load_image, num_parallel_calls=BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)
print("loaded dataset")

# image_list = glob.glob('./ffhq/thumbnails128x128/*/*')
# train_dataset = tf.data.Dataset.from_tensor_slices(image_list).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


# (train_images, train_labels), (_, _) = tf.keras.datasets.fashion_mnist.load_data() # LOAD NEW DATASET
# train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
# train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
# train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# def normalise(img):
#   img = (img - 127.5) / 127.5
#   return img

# img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./127.5, preprocessing_function=normalise)
# train_dataset = tf.data.Dataset.from_generator(
#   lambda: img_gen.flow_from_directory('./ffhq/thumbnails128x128/'),
#   output_signature=tf.TensorSpec(shape=(32, 128, 128, 3), dtype=tf.float32)
# )

def make_generator_model():
  model = tf.keras.Sequential()
  model.add(layers.Dense(8*8*256, use_bias=False, input_shape=(100,)))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())
  model.add(layers.Dropout(0.3))

  model.add(layers.Reshape((8, 8, 256)))
  assert model.output_shape == (None, 8, 8, 256)

  model.add(layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', use_bias=False))
  assert model.output_shape == (None, 16, 16, 256)
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', use_bias=False))
  assert model.output_shape == (None, 32, 32, 128)
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', use_bias=False))
  assert model.output_shape == (None, 64, 64, 128)
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', use_bias=False))
  assert model.output_shape == (None, 128, 128, 64)
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2DTranspose(32, (3, 3), strides=(1, 1), padding='same', use_bias=False))
  assert model.output_shape == (None, 128, 128, 32)
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2DTranspose(3, (3, 3), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
  assert model.output_shape == (None, 128, 128, 3)

  return model

def make_discriminator_model():
  model = tf.keras.Sequential()
  model.add(layers.experimental.preprocessing.Rescaling(1./127.5, offset=-1, input_shape=[128, 128, 3]))
  model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=[128, 128, 3])) # 64x64x64
  model.add(layers.LeakyReLU())
  model.add(layers.Dropout(0.4))

  model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')) # 32x32x128
  model.add(layers.LeakyReLU())
  model.add(layers.Dropout(0.4))

  model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')) # 16x16x128
  model.add(layers.LeakyReLU())
  model.add(layers.Dropout(0.4))

  model.add(layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same')) # 8x8x256
  model.add(layers.LeakyReLU())
  model.add(layers.Dropout(0.4))

  model.add(layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same')) # 4x4x512
  model.add(layers.LeakyReLU())
  model.add(layers.Dropout(0.4))

  model.add(layers.Flatten())
  model.add(layers.Dense(1, activation='sigmoid'))

  return model

def generator_loss(fake_output):
  return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
  real_loss = cross_entropy(tf.ones_like(real_output), real_output)
  fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
  total_loss = real_loss + fake_loss
  return total_loss

generator = make_generator_model()
discriminator = make_discriminator_model()
generator_optimizer = tf.keras.optimizers.Adam()
discriminator_optimizer = tf.keras.optimizers.Adam()

checkpoint_dir = './training_checkpoints'
checkpoint = tf.train.Checkpoint(
  step=tf.Variable(20),
  generator_optimizer=generator_optimizer,
  discriminator_optimizer=discriminator_optimizer,
  generator=generator,
  discriminator=discriminator
)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5, checkpoint_name="checkpoint")

@tf.function
def train_step(images):
  noise = tf.random.normal([BATCH_SIZE, noise_dim])

  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    generated_images = generator(noise, training=True)

    real_output = discriminator(images, training=True)
    fake_output = discriminator(generated_images, training=True)

    gen_loss = generator_loss(fake_output)
    disc_loss = discriminator_loss(real_output, fake_output)

    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

    return gen_loss, disc_loss

def train(dataset, epochs, manager):
  generate_examples(generator, 0, example_seed)
  checkpoint.restore(manager.latest_checkpoint)
  if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
  else:
    print("Initializing from scratch.")
  for epoch in range(epochs - checkpoint.step):
    start = time.time()
    checkpoint.step.assign_add(1)
    batches = 0
    gen_losses = 0
    disc_losses = 0
    print("epoch started")
    for image_batch in dataset:
      batches += 1
      gen_loss, disc_loss = train_step(image_batch)
      gen_losses += gen_loss
      disc_losses += disc_loss

    display.clear_output(wait=True)
    generate_examples(generator, checkpoint.step, example_seed)

    if (checkpoint.step) % 10 == 0:
      manager.save()
      print("Saved checkpoint for step {}: {}".format(int(checkpoint.step), checkpoint_dir))

    print('Epoch {} - gen_loss: {}, disc_loss: {}, time: {}'.format(
      checkpoint.step,
      gen_losses / batches,
      disc_losses / batches,
      time.time() - start)
    )

def generate_examples(model, epoch, seed):
  predictions = model(seed, training=False)
  plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, :] * 0.5 + 0.5)
      plt.axis('off')
  plt.savefig('./images_1/image_at_epoch_{:04d}.png'.format(epoch))
  plt.close('all')

train(train_dataset, EPOCHS, manager)
generator.save('trained_model.h5')