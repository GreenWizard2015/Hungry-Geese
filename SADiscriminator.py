import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers
import numpy as np
import time

def soft_labels_acc(maxDelta=0.1):
  def f(y_true, y_pred):
    return K.mean(
      K.cast(K.abs(y_true - y_pred) < maxDelta, tf.float16),
      axis=-1
    )

  f.__name__ = 'soft acc'
  return f

LEAKY_RELU = {'activation': keras.layers.LeakyReLU(alpha=0.2)}
def convBlock(prev, sz, filters):
  conv_1 = layers.Convolution2D(filters, (sz, sz), padding="same", **LEAKY_RELU)(prev)
  conv_1 = layers.Dropout(0.1)(conv_1)
  conv_1 = layers.BatchNormalization()(conv_1)
  return conv_1

def downsamplingBlock(res, sz, filters):
  for _ in range(3):
    res = convBlock(res, sz, filters)
  res = layers.Convolution2D(filters, (2,2), strides=2, padding="same", **LEAKY_RELU)(res)
  return res

def discriminatorModel(stateShape, actionsN, labelsMargin, fixDimensions=True):
  states = layers.Input(shape=stateShape)
  actions = layers.Input(shape=(actionsN,))

  S = states
  if fixDimensions:
    S = layers.Lambda(lambda x: K.permute_dimensions(x, (0, 2, 3, 1)))(S)
    
  SBranch = downsamplingBlock(S, 5, 32)
  SBranch = downsamplingBlock(SBranch, 3, 16)
  SBranch = downsamplingBlock(SBranch, 2, 8)
  SBranch = layers.Flatten()(SBranch)

  ABranch = layers.Dense(actionsN, **LEAKY_RELU)(actions)
  ittr = 1
  while ABranch.shape[1] < SBranch.shape[1]:
    ittr += 1
    ABranch = layers.Dense(actionsN * ittr, **LEAKY_RELU)(ABranch)
  
  merged = layers.Concatenate()([ABranch, SBranch])
  while 4 < merged.shape[1]:
    merged = layers.Dense(int(merged.shape[1]/3) + 1, activation='sigmoid')(merged)

  model = keras.Model(
    inputs=[states, actions],
    outputs=layers.Dense(1, activation='sigmoid')(merged),
    name='D'
  )
  model.compile(
    loss='mse', # because we use soft labels
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    metrics=[soft_labels_acc(labelsMargin)]
  )
  return model

class CDiscriminator:
  def __init__(self, stateShape, actionsN, labelsMargin=0.1):
    self._labelsMargin = labelsMargin
    self.network = discriminatorModel(stateShape, actionsN, labelsMargin)
    self.network.trainable = False
    return
  
  def train(
    self, samplesProvider, batchSize, batchesPerEpoch,
    epochs=100, minAcc=0.95, minEpochs=10, minLoss=float('inf')
  ):
    self.network.trainable = True
    
    for epoch in range(epochs):
      T = time.time()
      losses = []
      acc = []
      def fit(X, isReal):
        labelShift = 1. -  self._labelsMargin if isReal else 0.0
        Y = labelShift + np.random.random_sample((X[0].shape[0], )) * self._labelsMargin
        res = self.network.fit(X, Y, verbose=0).history
      
        losses.append(res['loss'][0])
        acc.append(res['soft acc'][0])
        return
      
      for _ in range(batchesPerEpoch):
        # train on good samples
        fit(samplesProvider(batchSize, True), True)
        # train on bad samples
        fit(samplesProvider(batchSize, False), False)

      #
      avgAcc = np.mean(acc)
      avgLoss = np.mean(losses)
      print(
        'Epoch %d. Time: %.1f sec. Avg. loss: %.4f. Avg. acc: %.2f.' % (
          epoch, time.time() - T, avgLoss, avgAcc
        )
      )
      if (minEpochs <= epoch) and (minAcc < avgAcc) and (avgLoss < minLoss): break
      
    self.network.trainable = False
    return avgLoss

  def predict(self, states, actions):
    self.network.trainable = False
    return self.network([states, actions]).numpy()[:, 0]