import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow as tf

keras.utils.get_custom_objects().update({
  'DQN_LReLu': keras.layers.LeakyReLU(alpha=0.2)
})

def convBlock(prev, sz, filters):
  conv_1 = layers.Convolution2D(filters, (sz, sz), padding="same", activation='DQN_LReLu')(prev)
  conv_1 = layers.Dropout(0.1)(conv_1)
  conv_1 = layers.BatchNormalization()(conv_1)
  return conv_1

def createModel(shape):
  inputs = res = layers.Input(shape=shape)
 
  for i in range(9):
    res = convBlock(res, 3, filters=(i * 4)+8)
  res = layers.Convolution2D(96, (2,2), strides=2, padding="same", activation='DQN_LReLu')(res)

  raw = res = layers.Flatten()(res)
  
  res = layers.Dense(256, activation='DQN_LReLu')(res)
  res = layers.Dense(256, activation='DQN_LReLu')(res)
  res = layers.Dense(128, activation='DQN_LReLu')(res)
  res = layers.Concatenate()([raw, res])
  
  # dueling dqn
  valueBranch = layers.Dense(128, activation='DQN_LReLu')(res)
  valueBranch = layers.Dense(64, activation='DQN_LReLu')(valueBranch)
  valueBranch = layers.Dense(32, activation='DQN_LReLu')(valueBranch)
  valueBranch = layers.Dense(1, activation='linear')(valueBranch)
  
  actionsBranch = layers.Dense(128, activation='DQN_LReLu')(res)
  actionsBranch = layers.Dense(64, activation='DQN_LReLu')(actionsBranch)
  actionsBranch = layers.Dense(64, activation='DQN_LReLu')(actionsBranch)
  actionsBranch = layers.Dense(64, activation='DQN_LReLu')(actionsBranch)
  actionsBranch = layers.Concatenate()([raw, actionsBranch])
  actionsBranch = layers.Dense(64, activation='DQN_LReLu')(actionsBranch)
  actionsBranch = layers.Dense(32, activation='DQN_LReLu')(actionsBranch)
  actionsBranch = layers.Dense(16, activation='DQN_LReLu')(actionsBranch)
  actionsBranch = layers.Dense(8, activation='DQN_LReLu')(actionsBranch)
  actionsBranch = layers.Dense(3, activation='linear')(actionsBranch)
  
  res = layers.Lambda(
    lambda x: x[1] + (x[0] - tf.reduce_mean(x[0], axis=-1, keepdims=True))
  )([actionsBranch, valueBranch])

  return keras.Model(inputs=inputs, outputs=res)
