import numpy as np
import os

def discountedWaves_centered(obstacles):
  res = np.zeros_like(obstacles, np.uint8)
  w,h = obstacles.shape
  directions = np.array([[1, 0], [0, 1], [0, -1], [-1, 0]])
  
  start = (w//2, h//2)
  res[start] = 1
  waveN = 1
  coords = np.zeros((100 + 4 * np.max(obstacles.shape), 2), np.uint8)
  coords[0] = start
  coordN = 1

  while True:
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
  res[0 == res] = waveN
  return res

def discountedWaves(obstacles, start, N):
  x,y = start
  chunk = obstacles[x-N:x+N+1, y-N:y+N+1]
  distMap = discountedWaves_centered(chunk)
  
  DMax = float(distMap.max())
  res = np.full_like(obstacles, DMax, np.float16)
  res[x-N:x+N+1, y-N:y+N+1] = distMap
  res /= DMax
  return res
