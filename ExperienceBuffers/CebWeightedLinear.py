import numpy as np
from ExperienceBuffers.Baseline_RB import PrioritizedReplayBuffer, ReplayBuffer

class CWeightsUpdater:
  def __init__(self, updater):
    self._updater = updater
    return
  
  def update(self, values):
    return self._updater(values)
  
class CebWeightedLinear:
  def __init__(self, maxSize, **kwargs):
    self._buffer = PrioritizedReplayBuffer(size=maxSize, **kwargs)
    self._normWeights = kwargs.get('normWeights', lambda x: np.abs(x) + 1.)
    return
 
  def _updateSamples(self, samplesID, values):
    self._buffer.update_priorities(samplesID, self._normWeights(values))
    return

  def store(self, samples):
    for sample in samples:
      self._buffer.add(sample)
    return
    
  def __len__(self):
    return len(self._buffer)

  def sample(self, batch_size, beta=None, weights=False):
    res = self._buffer.sample(batch_size, beta, weights)
    samples = res[:-1]
    indexes = res[-1]
    return samples, CWeightsUpdater(lambda v: self._updateSamples(indexes, v))
