import numpy as np
import cProfile

def discountedWaves_old(obstacles, start, N=10, discount=0.99):
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

def discountedWaves(obstacles, start, N=10, discount=0.99):
  N = 2
  res = np.zeros_like(obstacles, np.float)
  w,h = obstacles.shape
  directions = np.array([[1, 0], [0, 1], [0, -1], [-1, 0]])
  
  res[start] = 1.
  waveN = 0
  coords = np.zeros((100 + 4 * N, 2), np.uint16)
  coords[0] = start
  coordN = 1

  while waveN < N:
    nval = discount ** (1 + waveN)
    cN = 0
    for pt in coords[:coordN]:
      for d in directions:
        x,y = npt = np.add(d, pt)
        valid = (0 <= x) and (0 <= y) and (x < w) and (y < h)
        if valid and (res[x,y] <= 0) and (obstacles[x,y] <= 0):
          res[x,y] = nval
          cN += 1
          coords[-cN] = npt
    waveN += 1
    if cN <= 0: break
    coords[:cN] = coords[-cN:]
    coordN = cN
  return res
 
# np.set_printoptions(precision=2, linewidth=634)
# X = np.where(.9 < np.random.random_sample((128, 128)), 1, 0)
# # print(X)
# # print(discountedWaves_old(X, start=(5, 5), N=55))
# # print(discountedWaves(X, start=(5, 5), N=55))
# assert np.array_equal(
#   discountedWaves_old(X, start=(5, 5), N=55),
#   discountedWaves(X, start=(5, 5), N=55-1)
# )
#  
# p = cProfile.Profile()
# p.enable()
# for _ in range(128):
#   discountedWaves(X, start=(5, 5), N=55)
# p.disable()
# p.print_stats(sort=1)