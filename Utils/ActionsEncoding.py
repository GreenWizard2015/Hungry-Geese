import numpy as np
import itertools

def actions2index(actions, N=3):
  # ternary tree => array
  res = 0
  for action in actions:
    if action < 0: break
    res = (N * res) + action + 1
  return res

def index2actions(ind, N=3):
  res = []
  while 0 < ind:
    ind -= 1
    parent = ind // N
    res.insert(0, ind - (parent * N))
    ind = parent

  return res

def actions2coords(actions):
  shifts = [[0, -1], [-1, 0], [0, 1], [1, 0]]
  shiftsByDirection = [
    [3, 0, 1],
    [0, 1, 2],
    [1, 2, 3],
    [2, 3, 0],
  ]
  res = np.array((0, 0))
  prev = 1
  for act in actions:
    res += shifts[shiftsByDirection[prev][act]]
    prev = shiftsByDirection[prev][act]
    
  return res

def indexesForSeq(seqLen):
  minIndex = actions2index([0] * seqLen)
  maxIndex = actions2index([2] * seqLen)
  return minIndex, maxIndex

def extendTrajectory(actions, N):
  valid = [x for x in actions if 0 <= x]
  if N <= len(valid): return [valid]
  
  return [
    valid + list(p)
    for p in itertools.product(list(range(3)), repeat=N-len(valid))
  ]