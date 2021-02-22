from collections import defaultdict
import numpy as np

class CMulticastUpdater:
  def __init__(self):
    self._updaters = []
    self._index = 0
    return
  
  def update(self, values):
    for updater, From, Size in self._updaters:
      updater.update(values[From:From+Size])
    return
  
  def addUpdater(self, updater, chunkSize):
    self._updaters.append((updater, self._index, chunkSize))
    self._index += chunkSize
    return

def collectSamples(batch_size, sampler, samplesPerCall=None):
  if samplesPerCall is None:
    samplesPerCall = batch_size
  
  samplesN = 0
  updater = CMulticastUpdater()
  res = defaultdict(list)
  for gen in sampler():
    indStart = samplesN
    samples, subUpdater = gen(min((samplesPerCall, batch_size - samplesN)))

    for i, values in enumerate(samples):
      res[i].extend(values)
    
    samplesN = len(res[0])
    updater.addUpdater(subUpdater, samplesN - indStart)
    if batch_size <= samplesN:
      break
    ########
  return [np.array(values) for values in res.values()], updater

def sampleFromBuffer(buffer):
  def f(N):
    return buffer.sample(min((N, len(buffer))))
  return f
