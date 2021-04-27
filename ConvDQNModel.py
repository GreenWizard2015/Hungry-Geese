import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers
import tensorflow as tf

def convBlock(prev, sz, filters):
  conv_1 = layers.Convolution2D(filters, (sz, sz), padding="same")(prev)
  conv_1 = layers.LeakyReLU(alpha=0.2)(conv_1)
  conv_1 = layers.BatchNormalization()(conv_1)
  # conv_1 = layers.Dropout(0.1)(conv_1)
  return conv_1

def downsamplingBlock(res, sz, filters, hiddenLayers=1):
  for _ in range(hiddenLayers):
    res = convBlock(res, sz, filters)
  res = layers.Convolution2D(filters, (2,2), strides=2, padding="same")(res)
  return layers.LeakyReLU(alpha=0.2)(res)

def ConvSkipBlock(data, name):
  out = data
  inF = data.shape[-1]
  for sz, filters in [(1, inF*2), (2, 3 * inF), (3, 4 * inF)]:
    out = convBlock(out, sz, int(filters))
  return out

def ConvDDQNBranchBlock(data, name):
  branchA = ConvSkipBlock(data, name='%s_A' % name)
  # branchB = ConvSkipBlock(branchA, name='%s_B' % name)
  # branchSum = layers.Add(name='%s_Sum' % name)([branchA, branchB, data]) # 11x11x10
  branchD1 = downsamplingBlock(branchA, 1, filters=6) # 6x6x6
  branchD2 = downsamplingBlock(branchD1, 1, filters=3) # 3x3x3
  branchD3 = downsamplingBlock(branchD2, 1, filters=2) # 2x2x2
  return layers.Flatten()(branchD3)

def ConvDuelingDQNBlock(data, name, actionsN=3):
  # value branch
  valueBranch = layers.Dense(1, activation='linear', name='%s_value' % name)(
    ConvDDQNBranchBlock(data, name='%s_valueBranch' % name)
  )
  # actions branch
  actionsBranch = layers.Dense(actionsN, activation='linear', name='%s_actions' % name)(
    ConvDDQNBranchBlock(data, name='%s_actionsBranch' % name)
  )
  # combine branches  
  return layers.Lambda(
    lambda x: x[1] + (x[0] - tf.reduce_mean(x[0], axis=-1, keepdims=True)),
    name='%s_Q' % name
  )([actionsBranch, valueBranch])

def createModel(shape):
  gameState = res = layers.Input(shape=shape)

  res = convBlock(res, sz=2, filters=64)
  res = ConvDuelingDQNBlock(res, actionsN=3, name='DQN')  
  return keras.Model(
    inputs=[gameState],
    outputs=[res]
  )
