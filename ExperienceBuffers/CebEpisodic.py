import random
import numpy as np
import math
import itertools
from collections import defaultdict
from .Baseline_RB import PrioritizedReplayBuffer
from .CMulticastUpdater import CMulticastUpdater
from ExperienceBuffers.CebWeightedLinear import CebWeightedLinear
from ExperienceBuffers.CMulticastUpdater import collectSamples, sampleFromBuffer

_WEIGHTS_MODES = {
  'by score': lambda x: x[2],
  'same': lambda _: 1
}
 
class CebEpisodic:
  def __init__(self, maxSize, sampleWeight='same'):
    self.maxSize = maxSize
    self.sizeLimit = math.floor(maxSize * 1.1)
    self.episodes = []
    self.minScore = -math.inf
    self._sampleWeight = _WEIGHTS_MODES.get(sampleWeight, sampleWeight)
  
  def store(self, replay, score=None):
    if score is None:
      score = sum(x[2] for x in replay) # state, action, 2 - reward
    if score < self.minScore: return
    
    buffer = CebWeightedLinear(len(replay))
    buffer.store(replay)
    self.episodes.append((buffer, score))

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

  def sample(self, batch_size, maxSamplesFromEpisode=5):
    def sampler():
      cumweights = list(itertools.accumulate(abs(1 + self._sampleWeight(x)) for x in self.episodes))
      episodesPerLoop = 1 + (len(self.episodes) // 10)
      
      while True:
        episodes = random.choices(self.episodes, cum_weights=cumweights, k=episodesPerLoop)
        for episode in episodes:
          yield sampleFromBuffer(episode[0])
        ########
      return

    return collectSamples(batch_size, sampler, samplesPerCall=maxSamplesFromEpisode)