import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow as tf

def createModel(shape):
  inputs = res = layers.Input(shape=shape)
  raw = res = layers.Flatten()(res)
  res = layers.Dense(256, activation='relu')(res)
  res = layers.Dense(256, activation='relu')(res)
  res = layers.Dense(128, activation='relu')(res)
  res = layers.Concatenate()([raw, res])
  
  # dueling dqn
  valueBranch = layers.Dense(128, activation='relu')(res)
  valueBranch = layers.Dense(64, activation='relu')(valueBranch)
  valueBranch = layers.Dense(32, activation='relu')(valueBranch)
  valueBranch = layers.Dense(1, activation='linear')(valueBranch)
  
  actionsBranch = layers.Dense(128, activation='relu')(res)
  actionsBranch = layers.Dense(64, activation='relu')(actionsBranch)
  actionsBranch = layers.Concatenate()([raw, actionsBranch])
  actionsBranch = layers.Dense(64, activation='relu')(actionsBranch)
  actionsBranch = layers.Dense(64, activation='relu')(actionsBranch)
  actionsBranch = layers.Dense(3, activation='linear')(actionsBranch)
  
  res = layers.Lambda(
    lambda x: x[1] + (x[0] - tf.reduce_mean(x[0], axis=-1, keepdims=True))
  )([actionsBranch, valueBranch])

  return keras.Model(inputs=inputs, outputs=res)