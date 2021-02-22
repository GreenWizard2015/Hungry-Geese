import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers
import tensorflow as tf

keras.utils.get_custom_objects().update({
  'DQN_LReLu': keras.layers.LeakyReLU(alpha=0.2)
})

def convBlock(prev, sz, filters):
  conv_1 = layers.Convolution2D(filters, (sz, sz), padding="same", activation='DQN_LReLu')(prev)
  conv_1 = layers.Dropout(0.1)(conv_1)
#   conv_1 = layers.BatchNormalization()(conv_1)
  return conv_1

def downsamplingBlock(res, sz, filters, hiddenLayers=1):
  for _ in range(hiddenLayers):
    res = convBlock(res, sz, filters)
  res = layers.Convolution2D(filters, (2,2), strides=2, padding="same", activation='DQN_LReLu')(res)
  return res

def ConvSkipBlock(data, name):
  out = data
  inF = data.shape[-1]
  for sz, filters in [(1, inF), (2, 1.5 * inF), (3, 2 * inF), (4, 2 * inF), (5, 2 * inF), (4, 2 * inF), (3, 2 * inF), (2, 1.5 * inF), (1, inF)]:
    out = convBlock(out, sz, filters)
  return layers.Add(name='%s_CSB_out' % name)([data, out])

def ConvDDQNBranchBlock(data, name):
  branchA = ConvSkipBlock(data, name='%s_A' % name)
  branchB = ConvSkipBlock(branchA, name='%s_B' % name)
  branchSum = layers.Add(name='%s_Sum' % name)([branchA, branchB, data]) # 11x11x10
  branchD1 = downsamplingBlock(branchSum, 1, filters=6) # 6x6x6
  branchD2 = downsamplingBlock(branchD1, 1, filters=3) # 3x3x3
  branchD3 = downsamplingBlock(branchD2, 1, filters=2) # 2x2x2
  return layers.Flatten()(branchD3)

def ConvDuelingDQNBlock(data, name, actionsN=3):
  # value branch
  valueBranch = layers.Dense(1, activation='DQN_LReLu', name='%s_value' % name)(
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

def combineQValues(QNets):
  Good, Bad = QNets
  Good = 1 + tf.nn.softmax(Good - tf.math.reduce_min(Good, axis=1, keepdims=True))
  Bad = 2 - tf.nn.softmax(Bad - tf.math.reduce_min(Bad, axis=1, keepdims=True))

  return tf.nn.softmax(Good * Bad)

def createModel(shape, fixDimensions=True):
  gameState = res = layers.Input(shape=shape)
  
  if fixDimensions:
    res = layers.Lambda(lambda x: K.permute_dimensions(x, (0, 2, 3, 1)))(gameState)

  res = convBlock(res, sz=2, filters=64)
  GoodDQN = ConvDuelingDQNBlock(res, actionsN=3, name='GoodDQN')
  BadDQN = ConvDuelingDQNBlock(res, actionsN=3, name='BadDQN')

  ensembledQValues = layers.Lambda(combineQValues, name='EnsembledQ')([GoodDQN, BadDQN])
  
  return keras.Model(
    inputs=[gameState],
    outputs=[ensembledQValues, GoodDQN, BadDQN]
  )
