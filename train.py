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
    gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1 * 1024)]
  )
  
import Agents
import ConvDQNModel
from CREDQEnsemble import CREDQEnsembleTrainable

from ExperienceBuffers.CHGExperienceStorage import CHGExperienceStorage
import numpy as np
from Utils.CNoisedNetwork import CNoisedNetwork
import time
import Utils
import math
import random
from collections import defaultdict

def train(model, memory, params):
  T = time.time()
  lossSum = defaultdict(int) 
  for _ in range(params['episodes']):
    batch, Err = memory.sampleReplays(1024)
    states, actions, rewards, nextStates, nextStateScoreMultiplier = batch[:5]
    
    states = Utils.restoreStates(states)
    nextStates = Utils.restoreStates(nextStates)
    actions = actions.astype(np.int)
    
    ###############
    errors, loss = model.fit(states, actions, rewards, nextStates, nextStateScoreMultiplier)
    lossSum['loss'] += loss
    Err.update(errors)
    ###
    model.updateTargetModel(0.01)

  print('Training finished in %.1f sec.' % (time.time() - T))
  trainLoss = {k: v / params['episodes'] for k, v in lossSum.items()}
  print('Losses:')
  for k, v in trainLoss.items():
    print('Avg. %s: %.4f' % (k, v))
  print('')
  return

def forkAgent(model, epoch, params):
  LBM = model.clone()
  nm = 'LBM-%d' % epoch
  return (
    CNoisedNetwork(LBM, noise=.1+random.random() * 0.2),
    lambda world: Agents.CAgent(world, kind=nm)
  )

def learn_environment(model, params):
  NAME = params['name']
  metrics = {}
  wrHistory = {
    'network': []
  }

  memory = CHGExperienceStorage(params['experience storage'])
  ######################################################
  lastBestModels = [forkAgent(model, 0, params)] * 3

  def testModel(EXPLORE_RATE, epoch):
    T = time.time()
    opponents = [
      (Utils.DummyNetwork, Agents.CGreedyAgent),
      (Utils.DummyNetwork, Agents.CGreedyAgent),
      (Utils.DummyNetwork, Agents.CGreedyAgent),
    ] if 0 == (epoch % 2) else lastBestModels

    res = Utils.collectExperience(
      [ # agents
        (CNoisedNetwork(model, EXPLORE_RATE), Agents.CAgent),
        *opponents
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
    testModel(EXPLORE_RATE=0.8, epoch=0)

  #######################
  for epoch in range(params['epochs']):
    T = time.time()

    EXPLORE_RATE = params['explore rate'](epoch)
    print('[%s] %d/%d epoch. Explore rate: %.3f.' % (NAME, epoch, params['epochs'], EXPLORE_RATE))
    ##################
    # Training
#     if params.get('target update', lambda _: True)(epoch):
#       model.updateTargetModel()
    
    train(model, memory, { 'episodes': params['train episodes'](epoch) })

    ##################
    os.makedirs('weights', exist_ok=True)
    model.save('weights/%s-latest.h5' % NAME)
    # test
    if (epoch % params['test interval']) == 0:
      print('Testing...')
      stats, winRates = testModel(EXPLORE_RATE, epoch)
      for k, v in stats.items():
        Utils.trackScores(v, metrics, metricName=k)
      
      for k, v in winRates.items():
        if k not in wrHistory:
          wrHistory[k] = [0] * epoch
        wrHistory[k].append(v)
      ##################
      
      print('Scores sum: %.5f' % sum(stats['Score_network']))
      
      if (0 < (epoch % 2)) and (params['min win rate'] <= winRates['network']):
        print('save model (win rate: %.2f%%)' % (100.0 * winRates['network']))
        model.save('weights/%s-epoch-%06d.h5' % (NAME, epoch))
        ########
        lastBestModels.insert(0, forkAgent(model, epoch, params))
        modelsHistory = params.get('models history', 3)
        lastBestModels = lastBestModels[:modelsHistory]
    
      os.makedirs('charts/%s' % NAME, exist_ok=True)
      for metricName in metrics.keys():
        Utils.plotData2file(metrics, 'charts/%s/%s.jpg' % (NAME, metricName), metricName)
      Utils.plotSeries2file(wrHistory, 'charts/%s/win_rates.jpg' % (NAME,), 'Win rates')
    ##################
    print('Epoch %d finished in %.1f sec.' % (epoch, time.time() - T))
    print('------------------')
  return
############

network = CREDQEnsembleTrainable(
  submodel=ConvDQNModel.createModel,
  NModels=3, M=2
)
network.summary()

# calc GAMMA so  +-1 reward after N steps would give +-0.001 for current step
GAMMA = math.pow(0.001, 1.0 / 50.0)
print('Gamma: %.5f' % GAMMA)

ENVIRONMENT_SETTINGS ={
  'episode steps': 200,
  'min players': 2,
  ##############
  'survived reward': +5,
  'kill reward': +0,
  'grow reward': lambda x: 0.1,
  'starve reward': -10,
  'death reward': -0,
  'opponent death reward': +0,
  'killed reward': -0,
  'rank reward': {
    1: 11,
    2: 5,
    3: -5,
    4: -10
  }
}

DEFAULT_LEARNING_PARAMS = {
  'experience storage': {
    'batch size': 256,
    'gamma': GAMMA,
    'bootstrapped steps': 1,
    'fetch replays': {
      'replays': 256 * 1,
      'batch interval': 2000,
    },
    
    'replays': {
      'disabled': True,
      'folder': os.path.join(os.path.dirname(__file__), 'replays'),
      'replays per chunk': 1000,
      'env': ENVIRONMENT_SETTINGS,
    },
    
    'low level policy': {
    },
    
    'high level policy': {
      'steps': 5,
      'samples': 25,
    },
  },

  'epochs': 10000,
  'train episodes': lambda _: 16,
  'test interval': 1,
  'test episodes': 1,

  'explore rate': lambda e: 0.0,
  
  'env': ENVIRONMENT_SETTINGS,
  'min win rate': 0.55,
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