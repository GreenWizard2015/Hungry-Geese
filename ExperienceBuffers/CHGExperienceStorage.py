import numpy as np
from .CebEpisodic import CebEpisodic
from ExperienceBuffers.CMulticastUpdater import collectSamples, sampleFromBuffer
from ExperienceBuffers.CebWeightedLinear import CebWeightedLinear
from ExperienceBuffers.CHGReplaysStorage import CHGReplaysStorage
import time

class CHGExperienceStorage:
  def __init__(self, params):
    self._batchSize = params.get('batch size', None)
    
    GAMMA = params['gamma']
    BOOTSTRAPPED_STEPS = params['bootstrapped steps']
    self._DISCOUNTS = GAMMA ** np.arange(BOOTSTRAPPED_STEPS + 1)
    
    MEM_SIZE = 2 * 200 * 1000
    commonMemory = CebWeightedLinear(maxSize=MEM_SIZE)
    self._byRank = {
      1: CebWeightedLinear(maxSize=MEM_SIZE),
      2: CebWeightedLinear(maxSize=MEM_SIZE),
      3: commonMemory,
      4: commonMemory,
    }
    self._replaysBatchSize = params['replays batch size']
    
    self._replaysStorage = CHGReplaysStorage(params['replays'])
    self._fetchReplayN = params['fetch replays']['replays']
    self._fetchReplayInterval = params['fetch replays']['batch interval']
    self._batchID = 0
    return

  def _preprocess(self, replay, rank):
    BOOTSTRAPPED_STEPS = len(self._DISCOUNTS) - 1
    
    unzipped = list(zip(*replay))
    # details = unzipped[-1]
    prevState, actions, rewards, nextStates, alive = (np.array(x, np.float16) for x in unzipped[:-1])
    
    # bootstrap & discounts
    discounted = []
    L = len(rewards)
    for i in range(L):
      r = rewards[i:i+BOOTSTRAPPED_STEPS]
      N = len(r)

      discounted.append( (r * self._DISCOUNTS[:N]).sum() )
      nextStates[i] = nextStates[i+N-1]
      alive[i] *= self._DISCOUNTS[N]

    rewards = np.array(discounted, np.float16)
    actions = actions.astype(np.int8)
    ########
    return list(zip(prevState, actions, rewards, nextStates, alive))
  
  def store(self, replay, rank):
    self._byRank[rank].store(self._preprocess(replay, rank))
    return
  
  def sampleReplays(self, batch_size=None):
    if batch_size  is None:
      batch_size = self._replaysBatchSize
      
    if (self._batchID % self._fetchReplayInterval) == 0:
      self.fetchStoredReplays(self._fetchReplayN)
    # sample from each of ranks buffer
    def sampler():
      while True:
        for buffer in self._byRank.values():
          yield sampleFromBuffer(buffer)
        ########
      return

    data, _ = collectSamples(
      batch_size, sampler,
      samplesPerCall=1 + int(batch_size / (len(self._byRank) * 4))
    )
    self._batchID += 1      
    return data
  
  def storeReplay(self, replay):
    self._replaysStorage.store(replay)
    return
  
  def fetchStoredReplays(self, replaysN):
    T = time.time()
    N = 0
    for _ in range(replaysN):
      res = self._replaysStorage.sampleReplay()
      if res:
        trajectories, info = res
        for traj, rank in zip(trajectories, info['ranks']):
          self.store(traj, rank)
        N += 1
    print('Fetched %d from storage in %.1fms' % (N, (time.time() - T) * 1000.0))
    return