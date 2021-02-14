import numpy as np

def discountedWaves(obstacles, start, N, discount=0.99):
  res = np.zeros_like(obstacles, np.float)
  w,h = obstacles.shape
  directions = np.array([[1, 0], [0, 1], [0, -1], [-1, 0]])
  
  res[start] = 1# - obstacles[start]
  waveN = 1
  coords = np.zeros((100 + 4 * N, 2), np.uint16)
  coords[0] = start
  coordN = 1

  while waveN <= N:
    cN = 0
    for pt in coords[:coordN]:
      for d in directions:
        x,y = npt = np.add(d, pt)
        valid = (0 <= x) and (0 <= y) and (x < w) and (y < h)
        if valid and (res[x,y] <= 0) and (obstacles[x,y] <= 0):
          res[x,y] = waveN
          cN += 1
          coords[-cN] = npt
    waveN += 1
    if cN <= 0: break
    coords[:cN] = coords[-cN:]
    coordN = cN
    
  res[start] = 1 - obstacles[start]
  res /= waveN
  return res
