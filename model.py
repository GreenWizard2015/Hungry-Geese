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
  conv_1 = layers.BatchNormalization()(conv_1)
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

def DDQNBranchBlock(data, name):
  res = data
  for sz in range(data.shape[-1], 16, -32):
    res = layers.Dense(sz, activation='DQN_LReLu')(res)
    res = layers.Dropout(.1)(res)
  
  res = layers.Concatenate(axis=-1)([res, data])
  mid = res
  res = data
  for sz in range(data.shape[-1], 16, -24):
    res = layers.Dense(sz, activation='DQN_LReLu')(res)
    res = layers.Dropout(.1)(res)
  return res, mid

def DuelingDQNBlock(data, name, actionsN=3):
  # value branch
  valueBranch, VBR = DDQNBranchBlock(data, name='%s_valueBranch' % name)
  valueBranch = layers.Dense(1, activation='DQN_LReLu', name='%s_value' % name)(
    valueBranch
  )
  # actions branch
  actionsBranch, ABR = DDQNBranchBlock(data, name='%s_actionsBranch' % name)
  actionsBranch = layers.Dense(actionsN, activation='linear', name='%s_actions' % name)(
    actionsBranch
  )
  # combine branches  
  return layers.Lambda(
    lambda x: x[1] + (x[0] - tf.reduce_mean(x[0], axis=-1, keepdims=True)),
    name='%s_Q' % name
  )([actionsBranch, valueBranch]), [ABR,  VBR]

def combineQValues(QNets):
  Good, Bad = QNets
  Good = 1 + tf.nn.softmax(Good - tf.math.reduce_min(Good, axis=1, keepdims=True))
  Bad = 2 - tf.nn.softmax(Bad - tf.math.reduce_min(Bad, axis=1, keepdims=True))

  return tf.nn.softmax(Good * Bad)

def regularizationBlock(data, N):
  res = layers.Flatten()(data)
  res = layers.Dense(16, activation='tanh')(res)
  return layers.Dense(N, activation='tanh')(res)
  
def stackedMean(values):
  stacked = K.stack(values, axis=-1)
  return tf.reduce_mean(stacked, axis=-1)

def createModel(shape):
  gameState = res = layers.Input(shape=shape)

  res = downsamplingBlock(res, sz=5, filters=8, hiddenLayers=8)
  res = downsamplingBlock(res, sz=4, filters=64, hiddenLayers=8)
  res = downsamplingBlock(res, sz=3, filters=128, hiddenLayers=8)
  res = layers.Flatten()(res)
  DDQN, R1 = DuelingDQNBlock(res, actionsN=3, name='DQN')
  
  regLayers = R1
  AUX = layers.Lambda(stackedMean, name='AUX')([
    regularizationBlock(x, 3) for x in regLayers
  ])
    
  return keras.Model(
    inputs=[gameState],
    outputs=[DDQN, AUX]
  )
