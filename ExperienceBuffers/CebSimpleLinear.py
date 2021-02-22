import numpy as np
import random

class CebSimpleLinear:
  def __init__(self, maxSize):
    self._buffer = []
    self._maxSize = maxSize
    self._index = 0
    return

  def _put(self, data):
    if len(self._buffer) < self._maxSize:
      self._buffer.append(data)
    else:
      self._buffer[self._index] = data
    self._index = (self._index + 1) % self._maxSize
    return
  
  def store(self, samples):
    for sample in samples:
      self._put(sample)
    return
    
  def __len__(self):
    return len(self._buffer)

  def sample(self, batch_size):
    return random.choices(self._buffer, k=batch_size)
