import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow as tf
from Utils.TFUtils import downsamplingBlock

def DDQNBranchBlock(data, name):
  res = data
  for sz in range(data.shape[-1], 16, -32):
    res = layers.Dense(sz, activation='DQN_LReLu')(res)
    res = layers.Dropout(.1)(res)
  
  res = layers.Concatenate(axis=-1)([res, data])
  for sz in range(data.shape[-1], 16, -24):
    res = layers.Dense(sz, activation='DQN_LReLu')(res)
    res = layers.Dropout(.1)(res)
  return res

def DuelingDQNBlock(data, name, actionsN=3):
  # value branch
  valueBranch = layers.Dense(1, activation='DQN_LReLu', name='%s_value' % name)(
    DDQNBranchBlock(data, name='%s_valueBranch' % name)
  )
  # actions branch
  actionsBranch = layers.Dense(actionsN, activation='linear', name='%s_actions' % name)(
    DDQNBranchBlock(data, name='%s_actionsBranch' % name)
  )
  # combine branches  
  return layers.Lambda(
    lambda x: x[1] + (x[0] - tf.reduce_mean(x[0], axis=-1, keepdims=True)),
    name='%s_Q' % name
  )([actionsBranch, valueBranch])

def createModel(shape):
  gameState = res = layers.Input(shape=shape)

  res = downsamplingBlock(res, sz=5, filters=16, hiddenLayers=8)
  res = downsamplingBlock(res, sz=4, filters=32, hiddenLayers=8)
  res = downsamplingBlock(res, sz=3, filters=64, hiddenLayers=8)
  res = layers.Flatten()(res)

  return keras.Model(
    inputs=[gameState],
    outputs=[DuelingDQNBlock(res, actionsN=3, name='DQN')]
  )
