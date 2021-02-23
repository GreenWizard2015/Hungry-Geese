import random
import numpy as np
import math
import itertools

_WEIGHTS_MODES = {
  'abs': math.fabs,
  'reward': lambda x: x,
  'same': lambda _: 1
}

class CDummyWU:
  def update(self, w):
    return
  
class CebPrioritized:
  def __init__(self, maxSize, sampleWeight='same'):
    self.maxSize = maxSize
    self.sizeLimit = math.floor(maxSize * 1.1)
    self.episodes = []
    self.minScore = -math.inf
    self._sampleWeight = _WEIGHTS_MODES.get(sampleWeight, sampleWeight)
  
  def store(self, replay):
    score = sum(x[2] for x in replay) # state, action, 2 - reward
    if score < self.minScore: return
    self.episodes.append((replay, score))

    if self.sizeLimit < len(self.episodes):
      self.update()
    return

  def update(self):
    self.episodes = list(
      sorted(self.episodes, key=lambda x: x[1], reverse=True)
    )[:self.maxSize]
    self.minScore = self.episodes[-1][1]
    return 
    
  def __len__(self):
    return len(self.episodes)
  
  def _sampleIndexes(self, episode, maxSamples):
    return set(random.choices(
      np.arange(len(episode)),
      weights=[self._sampleWeight(x[2]) for x in episode],
      k=min((maxSamples, len(episode)))
    ))
  
  def _createBatch(self, batch_size, sampler):
    batchSize = 0
    cumweights = list(itertools.accumulate(x[1] for x in self.episodes))
    res = []
    while batchSize < batch_size:
      Episode = random.choices(self.episodes, cum_weights=cumweights, k=1)[0]
      for sample in sampler(Episode, batch_size - batchSize):
        while len(res) < len(sample):  res.append([])
        for i, value in enumerate(sample):
          res[i].append(value)
        batchSize += 1

    return [np.array(values) for values in res]
    
  def sampleBatch(self, batch_size, maxSamplesFromEpisode=5):
    def sampler(Episode, limit):
      limit = min((maxSamplesFromEpisode, limit))
      episode, _ = Episode
      minibatchIndexes = self._sampleIndexes(episode, limit)
      for ind in minibatchIndexes:
        yield episode[ind]
      return
        
    return self._createBatch(batch_size, sampler), CDummyWU()