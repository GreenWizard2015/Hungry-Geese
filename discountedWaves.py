import numpy as np

def discountedWaves(obstacles, start, N=10, discount=0.99):
  res = np.zeros_like(obstacles, np.float)
  visited = set()
  visited.add(start)
  w,h = obstacles.shape
  directions = np.array([[1, 0], [0, 1], [0, -1], [-1, 0]])
  
  res[start] = 1.
  waveN = 0
  prevWave = [start]
  while waveN < N:
    nextWave = []
    val = discount ** waveN
    for pt in prevWave:
      res[pt] = val
      for d in directions:
        x,y = npt = tuple(np.add(d, pt))
        valid = (0 <= x) and (0 <= y) and (x < w) and (y < h)
        if valid and (npt not in visited) and (obstacles[npt] <= 0):
          nextWave.append(npt)
          visited.add(npt)
    waveN += 1
    prevWave = nextWave
  return res
