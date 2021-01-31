import numpy as np

class CNoisedNetwork:
  def __init__(self, network, noise):
    self._network = network
    self._noise = noise
    return
  
  def predict(self, X):
    res = self._network(np.array(X)).numpy()
    if 0 < self._noise:
      rndIndexes = np.where(np.random.random_sample(res.shape[0]) < self._noise)
      res[rndIndexes] = np.random.random_sample(res.shape)[rndIndexes]
    return res