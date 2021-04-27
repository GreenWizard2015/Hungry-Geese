import numpy as np
from ExperienceBuffers.CMulticastUpdater import collectSamples, sampleFromBuffer
from ExperienceBuffers.CebWeightedLinear import CebWeightedLinear
from ExperienceBuffers.CHGReplaysStorage import CHGReplaysStorage
import time

class CHGExperienceStorage:
  def __init__(self, params):
    self._batchSize = params.get('batch size', None)
    
    GAMMA = params.get('gamma', 1.0)
    BOOTSTRAPPED_STEPS = params.get('bootstrapped steps', 0)
    self._LOOKAHEAD_STEPS = params.get('lookahead', 1)
    self._DISCOUNTS = GAMMA ** np.arange(BOOTSTRAPPED_STEPS + 1)
    
    MEM_SIZE = 200 * 2000
    self._byRank = {
      1: CebWeightedLinear(maxSize=MEM_SIZE),
      2: CebWeightedLinear(maxSize=MEM_SIZE),
      3: CebWeightedLinear(maxSize=MEM_SIZE),
      4: CebWeightedLinear(maxSize=MEM_SIZE),
    }
    self._replaysBatchSize = params['batch size']
    
    self._replaysStorage = CHGReplaysStorage(params['replays'])
    self._fetchReplayN = params['fetch replays']['replays']
    self._fetchReplayInterval = params['fetch replays']['batch interval']
    self._batchID = 0
    return

  def _preprocess(self, replay):
    BOOTSTRAPPED_STEPS = len(self._DISCOUNTS) - 1
    
    unzipped = list(zip(*replay))
    prevState, actions, rewards, nextStates, alive = (np.array(x, np.float16) for x in unzipped)
    
    # bootstrap & discounts
    discounted = []
    lookahead = []
    L = len(rewards)
    for i in range(L):
      r = rewards[i:i+BOOTSTRAPPED_STEPS]
      N = len(r)

      discounted.append( (r * self._DISCOUNTS[:N]).sum() )
      nextStates[i] = nextStates[i+N-1]
      alive[i] *= self._DISCOUNTS[N]
      #######
      a = actions[i:i+self._LOOKAHEAD_STEPS]
      N = len(a)
      la = np.full((self._LOOKAHEAD_STEPS,), -1)
      la[:N] = a
      lookahead.append(la)
      #######

    rewards = np.array(discounted, np.float16)
    actions = actions.astype(np.int8)
    lookahead = np.array(lookahead, np.int8)
    ########
    return [prevState, actions, rewards, nextStates, alive, lookahead]
  
  def _storeGame(self, game):
    playerData = [self._preprocess(replay) for replay, _ in game]
    for data, (_, rank) in zip(playerData, game):
      self._byRank[rank].store(list(zip(*data)))
    return
  
  def store(self, replays, save=True):
    _, data = replays
    for game in data['games']:
      self._storeGame(game)
    
    if not save: return
    for replay in data['raw replays']:
      self.storeReplay(replay)
    return
  
  def _sample(self, buffers, batch_size=None):
    if (self._batchID % self._fetchReplayInterval) == 0:
      self.fetchStoredReplays(self._fetchReplayN)
    self._batchID += 1
    
    if batch_size  is None:
      batch_size = self._replaysBatchSize
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
  
  def sampleReplays(self, batch_size=None):      
    return self._sample(self._byRank.values(), batch_size)
  
  def storeReplay(self, replay):
    self._replaysStorage.store(replay)
    return
  
  def sampleIL(self, batch_size=None, rank=1):
    (prevState, _, _, _, _, lookahead, _), _ = self._sample([self._byRank[rank]], batch_size)
    return prevState, lookahead

  def sampleStates(self, batch_size=None):
    res = self.sampleReplays(batch_size)
    return res[0][0]

  def sampleActionsLookahead(self, batch_size=None):
    (prevState, _, _, _, _, lookahead, _), _ = self.sampleReplays(batch_size)
    return prevState, lookahead
  
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