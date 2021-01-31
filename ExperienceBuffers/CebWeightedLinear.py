import random
import numpy as np
import itertools
from collections import defaultdict

class CWeightsUpdater:
  def __init__(self, updater):
    self._updater = updater
    return
  
  def update(self, values):
    return self._updater(values)
  
class CebWeightedLinear:
  def __init__(self, maxSize=None):
    self._maxSize = maxSize
    self._samples = []
    return
  
  def _normWeights(self, weights):
    eps = 1e-3
    return np.abs(np.power(np.abs(weights) + eps, 0.6) + eps)
 
  def _updateSamples(self, samplesID, values):
    for ind, value in zip(samplesID, self._normWeights(values)):
      self._samples[ind][-1] = value
    return

  def store(self, samples, weights=None):
    if weights is None:
      weights = np.ones((len(samples), )) * float('inf')
    else:
      weights = self._normWeights(weights)
    ######
    for sample, w in zip(samples, weights):
      self._samples.append([*sample, w])
    
    self.update()
    return

  def update(self):
    if self._maxSize * 1.25 < len(self._samples):
      self._samples = sorted(
        self._samples,
        key=lambda x: x[-1],
        reverse=True
      )[:self._maxSize]
    return
    
  def __len__(self):
    return len(self._samples)

  def _weights(self):
    return list(itertools.accumulate(x[-1] for x in self._samples))
 
  def _createBatch(self, batch_size):
    results = defaultdict(list)
    indexes = []
    
    samplesLeft = batch_size
    cumweights = self._weights() 
    indexRange = np.arange(len(self._samples))
    while 0 < samplesLeft:
      samplesInd = set(random.choices(
        indexRange, cum_weights=cumweights,
        k=min((samplesLeft, len(self._samples)))
      ))
      
      for i in samplesInd:
        sample = self._samples[i]
        indexes.append(i)
        for iv, value in enumerate(sample[:-1]):
          results[iv].append(value)
        samplesLeft -= 1
    ###
    return results, indexes
    
  def sampleBatch(self, batch_size):
    results, indexes = self._createBatch(batch_size)
    return [np.array(x) for x in results.values()], CWeightsUpdater(lambda v: self._updateSamples(indexes, v))