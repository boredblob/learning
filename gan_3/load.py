import tensorflow as tf
import matplotlib.pyplot as plt

noise_dim = 100
example_num = 16
example_seed = tf.random.normal([example_num, noise_dim])

model = tf.keras.models.load_model('./trained_model.h5')

predictions = model(example_seed, training=False)
plt.figure(figsize=(4, 4))

for i in range(predictions.shape[0]):
    plt.subplot(4, 4, i+1)
    plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='binary')
    plt.colorbar()
    plt.axis('off')

plt.show()