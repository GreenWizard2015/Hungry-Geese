import sys
import os
import tensorflow as tf
import Agents

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
from ExperienceBuffers.CHGExperienceStorage import CHGExperienceStorage
import numpy as np
from Utils.CNoisedNetwork import CNoisedNetwork
import time
import Utils
import math
import SADiscriminator
from Agents.CAgentState import LO_SHAPE
from collections import defaultdict

def collectExperience(agents, memory, params):
  trajectories, stats = collectReplays(
    models=[x[0] for x in agents],
    agentsKinds=[x[1] for x in agents],
    envN=params['episodes'],
    envParams=params.get('env', {})
  )
  
  scores = stats['scores']
  ranks = stats['ranks']
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
  for traj, rank in zip(trajectories, ranks):
    Ages.append(len(traj))
    RLRewards.append(sum(x[2] for x in traj))
    memory.store(traj, rank)

  for replay in stats['raw replays']:
    memory.storeReplay(replay)
    
  winRates = {}
  stats = {}
  for kind in kinds:
    replaysID = [i for i, k in enumerate(kinds) if k == kind]
    stats.update({
      'Age_%s' % kind: [Ages[i] for i in replaysID],
      'Score_%s' % kind: [scores[i] for i in replaysID],
      'RLRewards_%s' % kind: [RLRewards[i] for i in replaysID],
    })
  
    winN = sum(1 for i in replaysID if ranks[i] == 1)
    winRates[kind] = winN / float(len(replaysID))
    
  print('Win rates: ', winRates)
  return stats, winRates
###############
def train(model, targetModel, expertModel, memory, params):
#   EAcc = expertModel.train(
#     samplesProvider=memory.sampleExpertsActions, 
#     batchSize=64, batchesPerEpoch=8,
#     epochs=15,
#     minAcc=0.9, minEpochs=3,
#   )
  
  lossSum = defaultdict(int)
  for _ in range(params['episodes']):
    states, actions, rewards, nextStates, nextStateScoreMultiplier, futureActions, isStarve = memory.sampleReplays()
    dummyV = np.zeros((states.shape[0], 3))
    rows = np.arange(states.shape[0])
  
#     expertScores = 0.05 * EAcc * expertModel.predict(states, futureActions)
#     rewards += expertScores
    
    _, GoodDQNFuture, BadDQNFuture = targetModel.predict(nextStates)
    GFutureScores = GoodDQNFuture.max(axis=-1) * nextStateScoreMultiplier
    BFutureScores = BadDQNFuture.max(axis=-1) * nextStateScoreMultiplier
    ########
    _, GTargets, BTargets = targetModel.predict(states)
    
    GTargets[rows, actions] += (rewards + GFutureScores) - GTargets[rows, actions]
    BTargets[rows, actions] += (-rewards + BFutureScores) - BTargets[rows, actions]
    
    history = model.fit(
      states, 
      [dummyV, GTargets, BTargets],
      epochs=1, verbose=0
    ).history
    
    for k, v in history.items():
      lossSum[k] += v[0]
    ###

  return {k: v / params['episodes'] for k, v in lossSum.items()}

def learn_environment(model, params):
  NAME = params['name']
  metrics = {}
  wrHistory = {
    'network': []
  }

  memory = CHGExperienceStorage(params['experience storage'])
  expertModel = SADiscriminator.CDiscriminator(
    stateShape=params['shape'],
    actionsN=params['experience storage']['experts actions']['range']
  )
  ######################################################
  lastBestModels = [
    (Utils.DummyNetwork, Agents.CGreedyAgent),
    (Utils.DummyNetwork, Agents.CGreedyAgent),
  ]
  
  def testModel(EXPLORE_RATE):
    T = time.time()
    res = collectExperience(
      [ # agents
        (CNoisedNetwork(network, EXPLORE_RATE), Agents.CAgent),
        *lastBestModels,
        (Utils.DummyNetwork, Agents.CGreedyAgent)
      ],
      memory,
      {
        'episodes': params['test episodes'],
        'env': params.get('env', {})
      }
    )
    print('Testing finished in %.1f sec.' % (time.time() - T))
    return res
  ######################################################
  # collect some experience
  for epoch in range(2):
    testModel(EXPLORE_RATE=0.8)
    
  #######################
  targetModel = params['model clone'](model)
  targetModel.set_weights(model.get_weights())
  for epoch in range(params['epochs']):
    T = time.time()

    EXPLORE_RATE = params['explore rate'](epoch)
    print('[%s] %d/%d epoch. Explore rate: %.3f.' % (NAME, epoch, params['epochs'], EXPLORE_RATE))
    ##################
    # Training
    if params.get('target update', lambda _: True)(epoch):
      targetModel.set_weights(model.get_weights())
    trainLoss = train(
      model, targetModel, expertModel, memory,
      {
        'episodes': params['train episodes'](epoch),
        'model clone': params['model clone']
      }
    )
    print('Losses:')
    for k, v in trainLoss.items():
      print('Avg. %s: %.4f' % (k, v))
    print('')
    ##################
    # test
    print('Testing...')
    stats, winRates = testModel(EXPLORE_RATE)
    for k, v in stats.items():
      Utils.trackScores(v, metrics, metricName=k)
    
    for k, v in winRates.items():
      if k not in wrHistory:
        wrHistory[k] = [0] * epoch
      wrHistory[k].append(v)
    ##################
    
    print('Scores sum: %.5f' % sum(stats['Score_network']))
    
    os.makedirs('weights', exist_ok=True)
    model.save_weights('weights/%s-latest.h5' % NAME)
    if params['min win rate'] <= winRates['network']:
      print('save model (win rate: %.2f%%)' % (100.0 * winRates['network']))
      model.save_weights('weights/%s-epoch-%06d.h5' % (NAME, epoch))
      ########
      LBM = params['model clone'](model)
      LBM.set_weights(model.get_weights())
      lastBestModels = [
        (CNoisedNetwork(LBM, noise=0.0), lambda: Agents.CAgent(kind='LBM')),
        lastBestModels[0]
      ]
    
    ##################
    os.makedirs('charts/%s' % NAME, exist_ok=True)
    for metricName in metrics.keys():
      Utils.plotData2file(metrics, 'charts/%s/%s.jpg' % (NAME, metricName), metricName)
    Utils.plotSeries2file(wrHistory, 'charts/%s/win_rates.jpg' % (NAME,), 'Win rates')
    print('Epoch %d finished in %.1f sec.' % (epoch, time.time() - T))
    print('------------------')
############

MODEL_SHAPE = LO_SHAPE
network  = M.createModel(shape=MODEL_SHAPE)
network.summary()

def entropyOf(y_pred):
  probs = y_pred / (tf.reduce_sum(y_pred , axis=1, keepdims=True) + 1e-8)
  return -tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=1)

network.compile(optimizer=Adam(lr=1e-4, clipnorm=1.), loss=[
  lambda _,x: tf.math.reduce_mean(entropyOf(x)), # ensembledQValues
  Huber(delta=1.), # GoodDuelDQN
  Huber(delta=1.), # BadDuelDQN
], loss_weights=[0.05, 1, 1])

# calc GAMMA so  +-1 reward after N steps would give +-0.001 for current step
GAMMA = math.pow(0.001, 1.0 / 50.0)
print('Gamma: %.5f' % GAMMA)

DEFAULT_LEARNING_PARAMS = {
  'shape': MODEL_SHAPE,
  'model clone': lambda _: M.createModel(shape=MODEL_SHAPE),
  'experience storage': {
    'gamma': GAMMA,
    'bootstrapped steps': 2,
    'replays batch size': 64,
    
    'experts actions': {
      'range': 3,
      'mask': np.inf
    },
    
    'replays': {
      'folder': os.path.join(os.path.dirname(__file__), 'replays'),
      'replays per chunk': 1000,
    },
  },
  
  'epochs': 1000,
  'train episodes': lambda _: 64,
  'test episodes': 128,

  'explore rate': lambda e: max((.05 * .9**e, 1e-3)),
  
  'env': {
    'episode steps': 200,
    'min players': 2,
    ##############
    'survived reward': +10,
    'kill reward': +.5,
    'grow reward': lambda x: 0.85 ** (len(x) - 1),
    'starve reward': -10,
    'death reward': -10,
    'opponent death reward': +5,
    'killed reward': -1,
  },
  
  'min win rate': .5,
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