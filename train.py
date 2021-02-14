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

import model as M
from Utils import collectReplays
from ExperienceBuffers.CebWeightedLinear import CebWeightedLinear
from ExperienceBuffers.CebPrioritized import CebPrioritized
import numpy as np
from Utils.CNoisedNetwork import CNoisedNetwork
import time
import Utils

def collectExperience(model, memory, params):
  BOOTSTRAPPED_STEPS = params['bootstrapping steps']
  GAMMA = params['gamma']
  discounts = GAMMA ** np.arange(BOOTSTRAPPED_STEPS + 1) 
  
  trajectories, stats = collectReplays(
    model,
    agentsN=params.get('agents', 4),
    envN=params['episodes'],
    envParams=params.get('env', {})
  )
  
  scores = stats['scores']
  kinds = stats['kinds']
  kindsSet = set(kinds)
  
  deathBy = stats['death by']
  deathReasons = set(deathBy)
  
  for kind in kindsSet:
    print(kind, {
      reason: sum(
        1 for i, x in enumerate(deathBy) if (x==reason) and (kind==kinds[i])
      ) for reason in deathReasons
    })
  
  RLRewards = []
  Ages = []
  for traj in trajectories:
    prevState, actions, rewards, nextStates, alive = (np.array(x, np.float16) for x in zip(*traj))
    Ages.append(len(rewards))
    RLRewards.append(rewards.sum())
    # bootstrap & discounts
    discounted = []
    for i in range(len(rewards)):
      r = rewards[i:i+BOOTSTRAPPED_STEPS]
      N = len(r)

      discounted.append( (r * discounts[:N]).sum() )
      nextStates[i] = nextStates[i+N-1]
      alive[i] *= discounts[N]
    
    rewards = np.array(discounted, np.float16)
    actions = actions.astype(np.int8)
    ########
    memory.store(list(zip(prevState, actions, rewards, nextStates, alive)))

  stats = {}
  for kind in kinds:
    replaysID = [i for i, k in enumerate(kinds) if k == kind]
    stats.update({
      'Age_%s' % kind: [Ages[i] for i in replaysID],
      'Score_%s' % kind: [scores[i] for i in replaysID],
      'RLRewards_%s' % kind: [RLRewards[i] for i in replaysID],
    })
  return stats
###############
def train(model, memory, params):
  modelClone = params['model clone'](model)
  modelClone.set_weights(model.get_weights()) # use clone model for stability
  
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
    targets[rows, actions] += delta
    
    lossSum += model.fit(states, targets, epochs=1, verbose=0).history['loss'][0]
    ###

  return lossSum / params['episodes']

def learn_environment(model, params):
  NAME = params['name']
  BATCH_SIZE = params['batch size']
  GAMMA = params['gamma']
  BOOTSTRAPPED_STEPS = params['bootstrapped steps']
  metrics = {}

  memory = CebPrioritized(maxSize=5000)
  ######################################################
  def testModel(EXPLORE_RATE):
    return collectExperience(
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
    print('[%s] %d/%d epoch. Explore rate: %.3f.' % (NAME, epoch, params['epochs'], EXPLORE_RATE))
    ##################
    # Training
    trainLoss = train(
      model, memory,
      {
        'batch size': BATCH_SIZE,
        'episodes': params['train episodes'](epoch),
        'model clone': params['model clone']
      }
    )
    print('Avg. train loss: %.4f' % trainLoss)
    ##################
    # test
    print('Testing...')
    stats = testModel(EXPLORE_RATE)
    for k, v in stats.items():
      Utils.trackScores(v, metrics, metricName=k)
    ##################
    
    scoreSum = sum(stats['Score_network'])
    print('Scores sum: %.5f' % scoreSum)
    if bestModelScore < scoreSum:
      print('save best model (%.2f => %.2f)' % (bestModelScore, scoreSum))
      bestModelScore = scoreSum
      os.makedirs('weights', exist_ok=True)
      model.save_weights('weights/%s.h5' % NAME)
    ##################
    os.makedirs('charts/%s' % NAME, exist_ok=True)
    for metricName in metrics.keys():
      Utils.plotData2file(metrics, 'charts/%s/%s.jpg' % (NAME, metricName), metricName)
    print('Epoch %d finished in %.1f sec.' % (epoch, time.time() - T))
    print('------------------')
############

MODEL_SHAPE = (11, 11, 3)
network = M.createModel(shape=MODEL_SHAPE)
network.summary()
network.compile(optimizer=Adam(lr=1e-4, clipnorm=1.), loss=Huber(delta=1.))

DEFAULT_LEARNING_PARAMS = {
  'model clone': lambda _: M.createModel(shape=MODEL_SHAPE),
  'batch size': 256,
  'gamma': 0.95,
  'bootstrapped steps': 3,
  
  'epochs': 1000,
  'train episodes': lambda _: 64,
  'test episodes': 64,

  'explore rate': lambda e: max((.05 * .9**e, 1e-3)),
  
  'env': {
    'move reward': lambda x: 1./20,
    'grow reward': lambda x: max((0.95 ** (len(x) - 1), 0.5)),
    'kill reward': 2,
  }
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