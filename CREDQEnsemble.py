import numpy as np
import tensorflow as tf
from Agents.CAgentState import EMPTY_OBSERVATION

def combineQ(W, stackedQ):
  alter = (1.0 - W) * stackedQ.dtype.max
  masked = (stackedQ * W[:, :, None]) + alter[:, :, None]
  return tf.reduce_min(masked, axis=-2)

def createEnsemble(submodel, N_models, compileModel):
  layers = tf.keras.layers
  state = layers.Input(shape=EMPTY_OBSERVATION.shape)
  submodelsW = layers.Input(shape=(N_models, )) # for randomness during training
  
  submodels = [submodel(EMPTY_OBSERVATION.shape) for _ in range(N_models)]
  submodelsQ = [x(state) for x in submodels]
  
  combined = combineQ(submodelsW, tf.stack(submodelsQ, axis=1))
  
  model = tf.keras.Model(
    inputs=[state, submodelsW],
    outputs=[combined, *submodelsQ]
  )
  
  if compileModel:
    for X in submodels:
      X.compile(optimizer=tf.optimizers.Adam(lr=1e-4, clipnorm=1.), loss=None)
      X.summary()
  return model

def createTrainStep(model, targetModel, minibatchSize=64):
  submodels = [x for x in model.layers if x.name.startswith('model')]
  
  @tf.function
  def _trainBatch(states, actions, rewards, nextStates, nextStateScoreMultiplier, nextQScore):
    allW = tf.ones((tf.shape(states)[0], len(submodels)))
    nextAction = tf.argmax(model([nextStates, allW], training=False)[0], axis=-1)
    futureScores = tf.gather(nextQScore, nextAction, batch_dims=1)
    nextRewards = rewards + futureScores * nextStateScoreMultiplier
    tf.assert_rank(nextRewards, 1) # [None, ]
    
    with tf.GradientTape(persistent=True) as tape:
      predictions = model([states, allW])
      ########
      targets = predictions[0]
      targets = (nextRewards[:, None] * actions) + (targets * (1.0 - actions))
      ########
      tf.assert_equal(tf.shape(targets), tf.shape(predictions[-1]))
      losses = [
        tf.reduce_mean(tf.keras.losses.huber(targets, x)) for x in predictions[1:]
      ]
      
    for submodel in submodels:
      grads = tape.gradient(losses, submodel.trainable_weights)
      submodel.optimizer.apply_gradients(zip(grads, submodel.trainable_weights))
    
    return tf.reduce_mean(losses), nextRewards

  @tf.function
  def step(states, actions, rewards, nextStates, nextStateScoreMultiplier, W, tau):
    rewards = tf.cast(rewards, tf.float32)
    loss = 0.0
    nextRewards = tf.zeros((tf.shape(W)[0],))
    indices = tf.reshape(tf.range(minibatchSize), (-1, 1))
    nextStateScoreMultiplier = tf.cast(nextStateScoreMultiplier, tf.float32)
    actions = tf.one_hot(actions, tf.shape(W)[-1])

    for i in tf.range(0, tf.shape(states)[0], minibatchSize):
      nextQScore = targetModel([nextStates[i:i+minibatchSize], W[i:i+minibatchSize]], training=False)[0]
      
      bLoss, NR = _trainBatch(
        states[i:i+minibatchSize],
        actions[i:i+minibatchSize],
        rewards[i:i+minibatchSize],
        nextStates[i:i+minibatchSize],
        nextStateScoreMultiplier[i:i+minibatchSize],
        nextQScore
      )
      nextRewards = tf.tensor_scatter_nd_update(nextRewards, i + indices, NR)
      loss += bLoss

      # soft update
      for (a, b) in zip(targetModel.trainable_variables, model.trainable_variables):
        a.assign(b * tau + a * (1 - tau))

    newValues = model([nextStates, tf.ones_like(W)], training=False)[0]
    errors = tf.abs(nextRewards - tf.reduce_sum(actions * newValues, axis=-1))
    return errors, loss
  
  return step

###########
class CREDQEnsemble:
  def __init__(self, NModels, train=False, model=None, submodel=None):
    self._N = NModels
    self._model = model
    if not model:
      self._model = createEnsemble(submodel, self._N, compileModel=train) 
    return
  
  def predict(self, X):
    W = np.ones((len(X), self._N))
    return self._model([X, W])[0]
  
  def __call__(self, X):
    return self.predict(X)
  
  def load(self, filepath):
    self._model.load_weights(filepath)
    return
  
  def save(self, filepath):
    self._model.save_weights(filepath)
    return
  
  def summary(self):
    self._model.summary()
    return
###########
class CREDQEnsembleTrainable(CREDQEnsemble):
  def __init__(self, submodel, NModels, M):
    super().__init__(NModels, train=True, submodel=submodel)

    self._M = M
    self._submodel = submodel
    self._targetModel = self._cloneModel()
    self._trainStep = createTrainStep(self._model, self._targetModel, minibatchSize=64) 
    return
  
  def fit(self, states, actions, rewards, nextStates, nextStateScoreMultiplier, tau=0.005):
    # select M random models per sample (because we can)
    W = np.zeros((len(states), self._N))
    for i in range(len(states)):
      ind = np.random.choice(self._N, self._M, replace=False)
      W[i, ind] = 1.0
      
    return self._trainStep(states, actions, rewards, nextStates, nextStateScoreMultiplier, W, tau)

  @tf.function
  def _emaUpdate(self, targetV, srcV, tau):
    for (a, b) in zip(targetV, srcV):
      a.assign(b * tau + a * (1 - tau))
    return
  
  def updateTargetModel(self, tau=1.0):
    return
    if 1.0 <= tau:
      self._targetModel.set_weights(self._model.get_weights())
    else:
      self._emaUpdate(self._targetModel.trainable_variables, self._model.trainable_variables, tau)
    return

  def _cloneModel(self):
    res = createEnsemble(self._submodel, self._N, compileModel=False)
    res.set_weights(self._model.get_weights())
    return res
  
  def clone(self):
    return CREDQEnsemble(NModels=self._N, model=self._cloneModel())