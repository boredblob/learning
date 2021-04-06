import tensorflow as tf
import tensorflowjs as tfjs

encoder = tf.keras.models.load_model('./models/encoder.h5', compile=False)
decoder = tf.keras.models.load_model('./models/decoder.h5', compile=False)
tfjs.converters.save_keras_model(encoder, './models/models/encoder')
tfjs.converters.save_keras_model(decoder, './models/models/decoder')