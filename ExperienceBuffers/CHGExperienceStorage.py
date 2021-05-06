import numpy as np
from ExperienceBuffers.CMulticastUpdater import collectSamples, sampleFromBuffer
from ExperienceBuffers.CebWeightedLinear import CebWeightedLinear
from ExperienceBuffers.CHGReplaysStorage import CHGReplaysStorage
import time
from Agents.CWorldState import CWorldState, CGlobalWorldState
from Utils.ActionsEncoding import actions2coords

class CHGExperienceStorage:
  def __init__(self, params):
    self._batchSize = params.get('batch size', None)
    
    self._GAMMA = params.get('gamma', 1.0)
    BOOTSTRAPPED_STEPS = params.get('bootstrapped steps', 0)
    self._DISCOUNTS = self._GAMMA ** np.arange(BOOTSTRAPPED_STEPS + 1)
    
    self._LLParams = params['low level policy']
    MEM_SIZE = self._LLParams.get('memory size per rank', 20 * 2000)
    self._LLSamples = {
      1: CebWeightedLinear(maxSize=MEM_SIZE),
      2: CebWeightedLinear(maxSize=MEM_SIZE),
      3: CebWeightedLinear(maxSize=MEM_SIZE),
      4: CebWeightedLinear(maxSize=MEM_SIZE),
    }
    
    self._HLParams = params['high level policy']
    MEM_SIZE = self._HLParams.get('memory size per rank', 20 * 2000)
    self._HLSamples = {
      1: CebWeightedLinear(maxSize=MEM_SIZE),
      2: CebWeightedLinear(maxSize=MEM_SIZE),
      3: CebWeightedLinear(maxSize=MEM_SIZE),
      4: CebWeightedLinear(maxSize=MEM_SIZE),
    }
    
    self._replaysStorage = CHGReplaysStorage(params['replays'])
    self._fetchReplayN = params['fetch replays']['replays']
    self._fetchReplayInterval = params['fetch replays']['batch interval']
    self._batchID = 0
    return

  def _createLL(self, replay):
    BOOTSTRAPPED_STEPS = len(self._DISCOUNTS) - 1
    
    unzipped = list(zip(*replay))
    prevState, actions, rewards, nextStates, alive = (np.array(x, np.float16) for x in unzipped)
    
    # bootstrap & discounts
    discounted = []
    L = len(rewards)
    for i in range(L):
      r = rewards[i:i+BOOTSTRAPPED_STEPS]
      N = len(r)

      discounted.append( (r * self._DISCOUNTS[:N]).sum() )
      nextStates[i] = nextStates[i+N-1]
      alive[i] *= self._DISCOUNTS[N]
      #######

    rewards = np.array(discounted, np.float16)
    actions = actions.astype(np.int8)
    ########
    return [prevState, actions, rewards, nextStates, alive]

  def _createHL(self, replay):
    unzipped = list(zip(*replay))
    prevState, actions, rewards, nextStates, alive = (np.array(x, np.float16) for x in unzipped)
    
    STEPS = self._HLParams['steps']
    SAMPLES = self._HLParams.get('samples', len(prevState))
    
    # bootstrap & discounts
    startStatesIndex = list(set(np.random.choice(
      np.arange(len(prevState)),
      min(len(prevState), SAMPLES)
    )))
    rS, rA, rR, rNS, rM = [], [], [], [], []
    for ind in startStatesIndex:
      r = rewards[ind:ind+STEPS]
      N = len(r)

      relShift = actions2coords(actions[ind:ind+N].astype(np.int))
      dist = np.linalg.norm(relShift)
      
      rS.append(prevState[ind])
      rA.append( relShift )
      rR.append( (r[::-1] * (self._GAMMA ** np.arange(N))).sum() + dist )
      rNS.append(nextStates[ind+N-1])
      rM.append(alive[ind + N - 1] * self._GAMMA)
      #######

    rS, rA, rR, rNS, rM = [np.array(x, np.float16) for x in [rS, rA, rR, rNS, rM]]
    rA = rA.astype(np.int8)
    ########
    print(rA)
    exit()
    return [rS, rA, rR, rNS, rM]
  
  def _storeGame(self, game):
    for replay, rank in game:
      LLData = self._createLL(replay)
      self._LLSamples[rank].store(list(zip(*LLData)))
      
      HLData = self._createHL(replay)
      self._LLSamples[rank].store(list(zip(*HLData)))
    return
  
  def store(self, replays, save=True):
    _, data = replays
    for game in data['games']:
      self._storeGame(game)
    
    if not save: return
    for replay in data['raw replays']:
      self.storeReplay(replay)
    return
  
  def _sample(self, buffers, batch_size):
    if (self._batchID % self._fetchReplayInterval) == 0:
      self.fetchStoredReplays(self._fetchReplayN)
    self._batchID += 1

    # sample from each of buffer
    def sampler():
      while True:
        for buffer in buffers:
          yield sampleFromBuffer(buffer)
        ########
      return

    return collectSamples(
      batch_size, sampler,
      samplesPerCall=1 + int(batch_size / (len(buffers) * 4))
    )
  
  def sampleLowLevel(self, batch_size):      
    return self._sample(self._LLSamples.values(), batch_size)
  
  def sampleHighLevel(self, batch_size):
    return self._sample(self._HLSamples.values(), batch_size)
  
  def storeReplay(self, replay):
    self._replaysStorage.store(replay)
    return

  def fetchStoredReplays(self, replaysN):
    T = time.time()
    N = 0
    for _ in range(replaysN):
      res = self._replaysStorage.sampleReplay()
      if res:
        self.store(res, save=False)
        N += 1
    print('Fetched %d from storage in %.1fms' % (N, (time.time() - T) * 1000.0))
    return