import numpy as np
import random
from collections import defaultdict
from ExperienceBuffers.segment_tree import SumSegmentTree, MinSegmentTree

class CCompressNumpyHack:
  def __init__(self, arr):
    self.shape = arr.shape
    self.coords = np.array(np.nonzero(arr))
    self.values = arr[(*self.coords, )]
    
  def get(self):
    res = np.zeros(self.shape)
    res[(*self.coords, )] = self.values
    return res

"""
Code mainly copied from https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
but with small tweaks 
"""
class ReplayBuffer(object):
  def __init__(self, size):
    """Create Replay buffer.

    Parameters
    ----------
    size: int
      Max number of transitions to store in the buffer. When the buffer
      overflows the old memories are dropped.
    """
    self._storage = []
    self._maxsize = size
    self._next_idx = 0

  def __len__(self):
    return len(self._storage)

  def _readSample(self, index):
    def f(x):
      if isinstance(x, CCompressNumpyHack):
        return x.get()
      return x
    return [f(el) for el in self._storage[index]]
  
  def _packSample(self, data):
    def f(x):
      if isinstance(x, np.ndarray) and (100 < x.size):
        return CCompressNumpyHack(x)
      return x
    return [f(el) for el in data]
  
  def add(self, data):
    data = self._packSample(data)
    
    if len(self._storage) <= self._next_idx:
      self._storage.append(data)
    else:
      self._storage[self._next_idx] = data
    self._next_idx = (self._next_idx + 1) % self._maxsize
  
  def _encode_sample(self, idxes):
    res = defaultdict(list)
    for i in idxes:
      data = self._readSample(i)
      for i, value in enumerate(data):
        res[i].append(value)

    return [np.array(x) for x in res.values()]

  def sample(self, batch_size):
    """Sample a batch of experiences.

    Parameters
    ----------
    batch_size: int
      How many transitions to sample.

    Returns
    -------
    Tuple with N columns
    """
    idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
    return self._encode_sample(idxes)

class PrioritizedReplayBuffer(ReplayBuffer):
  def __init__(self, size, alpha=0.6, beta=0.4):
    """Create Prioritized Replay buffer.

    Parameters
    ----------
    size: int
      Max number of transitions to store in the buffer. When the buffer
      overflows the old memories are dropped.
    alpha: float
      how much prioritization is used
      (0 - no prioritization, 1 - full prioritization)
    beta: float
      To what degree to use importance weights (default value for sample())
      (0 - no corrections, 1 - full correction)

    See Also
    --------
    ReplayBuffer.__init__
    """
    super(PrioritizedReplayBuffer, self).__init__(size)
    assert alpha >= 0
    self._alpha = alpha
    self._beta = beta

    it_capacity = 1
    while it_capacity < size:
      it_capacity *= 2

    self._it_sum = SumSegmentTree(it_capacity)
    self._it_min = MinSegmentTree(it_capacity)
    self._max_priority = 1.0

  def add(self, *args, **kwargs):
    idx = self._next_idx
    super().add(*args, **kwargs)
    self._it_sum[idx] = self._max_priority ** self._alpha
    self._it_min[idx] = self._max_priority ** self._alpha

  def _sample_proportional(self, batch_size):
    res = []
    p_total = self._it_sum.sum()
    every_range_len = p_total / batch_size
    for i in range(batch_size):
      mass = random.random() * every_range_len + i * every_range_len
      idx = self._it_sum.find_prefixsum_idx(mass)
      res.append(idx)
    return res

  def sample(self, batch_size, beta=None, weights=False, proportional=True):
    """Sample a batch of experiences.

    compared to ReplayBuffer.sample
    it also returns importance weights and idxes
    of sampled experiences.


    Parameters
    ----------
    batch_size: int
      How many transitions to sample.
    beta: float
      To what degree to use importance weights
      (0 - no corrections, 1 - full correction)
    weights: boolean
      Return importance weights if True

    Returns
    -------
    Tuple with N columns
    weights: np.array
      Array of shape (batch_size,) and dtype np.float32
      denoting importance weight of each sampled transition
    idxes: np.array
      Array of shape (batch_size,) and dtype np.int32
      idexes in buffer of sampled experiences
    """
    beta = self._beta if beta is None else beta
    assert beta > 0

    if proportional:
      idxes = self._sample_proportional(batch_size)
    else:
      idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
    results = self._encode_sample(idxes)

    if weights:
      weights = []
      itsum = self._it_sum.sum()
      p_min = self._it_min.min() / itsum
      max_weight = (p_min * len(self._storage)) ** (-beta)
      for idx in idxes:
        p_sample = self._it_sum[idx] / itsum
        weight = (p_sample * len(self._storage)) ** (-beta)
        weights.append(weight / max_weight)
        
      results.append(np.array(weights))
    #
    results.append(idxes)
    return results

  def update_priorities(self, idxes, priorities):
    """Update priorities of sampled transitions.

    sets priority of transition at index idxes[i] in buffer
    to priorities[i].

    Parameters
    ----------
    idxes: [int]
      List of idxes of sampled transitions
    priorities: [float]
      List of updated priorities corresponding to
      transitions at the sampled idxes denoted by
      variable `idxes`.
    """
    assert len(idxes) == len(priorities)
    for idx, priority in zip(idxes, priorities):
      assert priority > 0
      assert 0 <= idx < len(self._storage)
      self._it_sum[idx] = priority ** self._alpha
      self._it_min[idx] = priority ** self._alpha

      self._max_priority = max(self._max_priority, priority)