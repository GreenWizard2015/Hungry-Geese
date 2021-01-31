import sys
import os
import tensorflow as tf

if 'COLAB_GPU' in os.environ:
  # fix resolve modules
  from os.path import dirname
  sys.path.append(dirname(dirname(dirname(__file__))))
else: # local GPU
  gpus = tf.config.experimental.list_physical_devices('GPU')
  tf.config.experimental.set_virtual_device_configuration(
    gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3 * 1024)]
  )

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber

from CHGEnvironment import CHGEnvironment
import model
from Utils import collectReplays
from ExperienceBuffers.CebWeightedLinear import CebWeightedLinear
import numpy as np
from Utils.CNoisedNetwork import CNoisedNetwork
import time
import Utils

def collectExperience(env, model, memory, params):
  BOOTSTRAPPED_STEPS = params['bootstrapping steps']
  GAMMA = params['gamma']
  bootstrapping = GAMMA ** np.arange(BOOTSTRAPPED_STEPS) 
  
  FOOD_LAYER = 1
  PLAYER_LAYER = 2
  trajectories, allScores = collectReplays(model, agentsN=4, envN=params['episodes'])
  allScores = []
  for traj in trajectories:
    prevState, actions, rewards, nextStates, alive = zip(*traj)
    # bootstrap
    alive = np.array(alive, np.float16)
    actions = np.array(actions, np.int8)
    rewards = np.array(rewards, np.float16)
    prevState = np.array(prevState, np.float16)
    nextStates = np.array(nextStates, np.float16)
    prevFoodReward = 0
    
    for i in range(len(rewards)):
      r = rewards[i:i+BOOTSTRAPPED_STEPS]
      sz = len(r)
      
      foodD = prevState[i][PLAYER_LAYER, 0 < prevState[i][FOOD_LAYER]]
      foodReward = .1 * (1 - foodD.min())
      rewards[i] = (r * bootstrapping[:sz]).sum() + foodReward - prevFoodReward
      prevFoodReward = foodReward
      
      nextStates[i] = nextStates[i+sz-1]
      alive[i] *= GAMMA ** sz
    ########
    memory.store(list(zip(prevState, actions, rewards, nextStates, alive)))
    allScores.append(rewards.sum())
  return allScores
###############
def train(model, memory, params):
  modelClone = tf.keras.models.clone_model(model)
  modelClone.set_weights(model.get_weights()) # use clone model for stability
  
  ALPHA = params.get('alpha', 1.0)
  rows = np.arange(params['batch size'])
  lossSum = 0
  for _ in range(params['episodes']):
    (states, actions, rewards, nextStates, nextStateScoreMultiplier), W = memory.sampleBatch(
      batch_size=params['batch size']
    )
  
    futureScores = modelClone.predict(nextStates).max(axis=-1) * nextStateScoreMultiplier
    targets = modelClone.predict(states)
    delta = rewards + futureScores - targets[rows, actions]
    W.update(delta)
    targets[rows, actions] += ALPHA * delta
    
    lossSum += model.fit(states, targets, epochs=1, verbose=0).history['loss'][0]
    ###

  return lossSum / params['episodes']

def learn_environment(model, params):
  NAME = params['name']
  BATCH_SIZE = params['batch size']
  GAMMA = params['gamma']
  BOOTSTRAPPED_STEPS = params['bootstrapped steps']
  metrics = {}

  memory = CebWeightedLinear(maxSize=350000)
  ######################################################
  def testModel(EXPLORE_RATE):
    return collectExperience(
      CHGEnvironment({'agents': 4}),
      CNoisedNetwork(network, EXPLORE_RATE),
      memory,
      {
        'gamma': GAMMA,
        'bootstrapping steps': BOOTSTRAPPED_STEPS,
        'episodes': params['test episodes']
      }
    )
  ######################################################
  # collect some experience
  for _ in range(2):
    testModel(EXPLORE_RATE=0.8)
  #######################
  bestModelScore = -float('inf')
  for epoch in range(params['epochs']):
    T = time.time()
    
    EXPLORE_RATE = params['explore rate'](epoch)
    alpha = params.get('alpha', lambda _: 1)(epoch)
    print(
      '[%s] %d/%d epoch. Explore rate: %.3f. Alpha: %.5f.' % (NAME, epoch, params['epochs'], EXPLORE_RATE, alpha)
    )
    print('Samples: %d.' % (len(memory), ))
    ##################
    # Training
    trainLoss = train(
      model, memory,
      {
        'batch size': BATCH_SIZE,
        'episodes': params['train episodes'](epoch),
        'alpha': alpha
      }
    )
    print('Avg. train loss: %.4f' % trainLoss)
    ##################
    # test
    print('Testing...')
    scores = testModel(EXPLORE_RATE)
    Utils.trackScores(scores, metrics)
    ##################
    
    scoreSum = sum(scores)
    print('Scores sum: %.5f' % scoreSum)
    if bestModelScore < scoreSum:
      print('save best model (%.2f => %.2f)' % (bestModelScore, scoreSum))
      bestModelScore = scoreSum
      os.makedirs('weights', exist_ok=True)
      model.save_weights('weights/%s.h5' % NAME)
    ##################
    os.makedirs('charts', exist_ok=True)
    Utils.plotData2file(metrics, 'charts/%s.jpg' % NAME)
    print('Epoch %d finished in %.1f sec.' % (epoch, time.time() - T))
    print('------------------')
############

network = model.createModel(shape=(6, 11, 11))
network.compile(optimizer=Adam(lr=1e-4), loss=Huber(delta=1))

DEFAULT_LEARNING_PARAMS = {
  'batch size': 256,
  'gamma': 0.99,
  'bootstrapped steps': 10,
  
  'epochs': 1000,
  'train episodes': lambda _: 64,
  'test episodes': 256,

  'alpha': lambda _: 1,
  'explore rate': lambda e: max((.05 * .99**e, 1e-3)),
}
#######################
for i in range(1):
  learn_environment(
    network,
    {
      **DEFAULT_LEARNING_PARAMS,
      'name': 'agent-%d' % i
    }
  )