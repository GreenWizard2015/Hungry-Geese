import tensorflow as tf
import keras.backend as K
import keras.layers as layers
import keras

keras.utils.get_custom_objects().update({
  'DQN_LReLu': keras.layers.LeakyReLU(alpha=0.2)
})

def entropyOf(y):
  eps = K.epsilon()
  probs = y / (tf.reduce_sum(y, axis=1, keepdims=True) + eps)
  return -tf.reduce_sum(probs * tf.math.log(probs + eps), axis=1)

def combineQValues(QNets):
  predictions = [
    1 + tf.nn.softmax(x - tf.math.reduce_min(x, axis=1, keepdims=True)) for x in QNets
  ]

  return tf.nn.softmax(tf.reduce_prod(predictions, axis=1))

def stackedMean(values):
  stacked = K.stack(values, axis=-1)
  return tf.reduce_mean(stacked, axis=-1)

def convBlock(prev, sz, filters):
  conv_1 = layers.Convolution2D(filters, (sz, sz), padding="same", activation='DQN_LReLu')(prev)
  conv_1 = layers.Dropout(0.1)(conv_1)
  conv_1 = layers.BatchNormalization()(conv_1)
  return conv_1

def downsamplingBlock(res, sz, filters, hiddenLayers=1):
  for _ in range(hiddenLayers):
    res = convBlock(res, sz, filters)
  res = layers.Convolution2D(filters, (2,2), strides=2, padding="same", activation='DQN_LReLu')(res)
  return res