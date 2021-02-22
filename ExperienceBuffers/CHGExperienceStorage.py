import numpy as np
from .CebEpisodic import CebEpisodic
from ExperienceBuffers.CMulticastUpdater import collectSamples, sampleFromBuffer
from ExperienceBuffers.CebWeightedLinear import CebWeightedLinear

class CHGExperienceStorage:
  def __init__(self, params):
    self._batchSize = params.get('batch size', None)
    
    GAMMA = params['gamma']
    BOOTSTRAPPED_STEPS = params['bootstrapped steps']
    self._DISCOUNTS = GAMMA ** np.arange(BOOTSTRAPPED_STEPS + 1)
    
    commonMemory = CebWeightedLinear(maxSize=200 * 1000)
    self._byRank = {
      # 1: CebEpisodic(maxSize=1000),
      1: CebWeightedLinear(maxSize=200 * 1000),
      # 2: CebEpisodic(maxSize=2000),
      2: CebWeightedLinear(maxSize=200 * 1000),
      3: commonMemory,
      4: commonMemory,
    }
    self._replaysBatchSize = params['replays batch size']
    
    self._expertsActionsN = params['experts actions']['range']
    self._expertsActionsMask = params['experts actions']['mask']
    return
  
  def _encodeActions(self, acts):
    masked = np.full((self._expertsActionsN, ), self._expertsActionsMask)
    masked[:len(acts)] = acts[:self._expertsActionsN] - 1. # 0..2 => -1..1
    return masked.astype(np.int8)

  def _preprocess(self, replay, rank):
    BOOTSTRAPPED_STEPS = len(self._DISCOUNTS) - 1
    
    unzipped = list(zip(*replay))
    details = unzipped[-1]
    prevState, actions, rewards, nextStates, alive = (np.array(x, np.float16) for x in unzipped[:-1])
    
    # bootstrap & discounts
    discounted = []
    futureActions = []
    for i in range(len(rewards)):
      r = rewards[i:i+BOOTSTRAPPED_STEPS]
      N = len(r)

      discounted.append( (r * self._DISCOUNTS[:N]).sum() )
      nextStates[i] = nextStates[i+N-1]
      alive[i] *= self._DISCOUNTS[N]
      futureActions.append(self._encodeActions(actions[i:]))
    
    rewards = np.array(discounted, np.float16)
    rewards[-1] + rank
    actions = actions.astype(np.int8)
    futureActions = np.array(futureActions, np.int8)
    ########
    isStarve = [x['starve'] for x in details]
    return list(zip(prevState, actions, rewards, nextStates, alive, futureActions, isStarve))
  
  def store(self, replay, rank):
    self._byRank[rank].store(self._preprocess(replay, rank))
    return
  
  def sampleReplays(self, batch_size=None):
    if batch_size  is None:
      batch_size = self._replaysBatchSize      
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
    return data
  
  def sampleExpertsActions(self, batch_size, sampleExpert):
    buffer = self._byRank[1] if  sampleExpert else self._byRank[4]
    (states, _, _, _, _, futureActions, _), _ = buffer.sample(batch_size)
    return [states, futureActions]
  