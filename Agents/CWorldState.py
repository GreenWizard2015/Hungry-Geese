import numpy as np
from kaggle_environments.envs.hungry_geese.hungry_geese import Action, row_col

PLAYERS_N = 4
BASIC_GRID_ROWS, BASIC_GRID_COLUMNS = BASIC_GRID_SHAPE = (7, 11)
BASIC_GRID_SIZE = np.prod(BASIC_GRID_SHAPE)
MAX_DIM = max(BASIC_GRID_SHAPE)

LAYER_FOOD, LAYER_OBSTACLES = np.arange(2)

INDEX_TO_COORD = np.array([
  row_col(x, BASIC_GRID_COLUMNS) for x in range(BASIC_GRID_SIZE)
])

ROTATIONS_FOR_ACTION = {
  'EAST': 1, 'WEST': 3,
  'SOUTH': 2, 'NORTH': 0, 
}
INDEX_TO_ACTION = ['WEST', 'NORTH', 'EAST', 'SOUTH']
ACTIONS_MAPPING_BY_DIRECTION = [
  # NORTH
  ['WEST', 'NORTH', 'EAST'],
  # EAST
  ['NORTH', 'EAST', 'SOUTH'],
  # SOUTH
  ['EAST', 'SOUTH', 'WEST'],
  # WEST
  ['SOUTH', 'WEST', 'NORTH'],
]

DELTA_TO_ACTION = {
  (1, 0): Action.EAST, (-10, 0): Action.EAST,
  (-1, 0): Action.WEST, (10, 0): Action.WEST,
  (0, -1): Action.NORTH, (0, 6): Action.NORTH,
  (0, -6): Action.SOUTH, (0, 1): Action.SOUTH
}

###############
RAW_OBSERVATION_LAYERS = 1 + PLAYERS_N
# rotation, head index, world grid
EMPTY_RAW_OBSERVATION = np.zeros((BASIC_GRID_SIZE + 2, RAW_OBSERVATION_LAYERS), np.int)
###############
TAIL_VALUE = 0.1
def _encodeBody(L):
  res = list(np.arange(1., TAIL_VALUE, -(1. - TAIL_VALUE) / L))
  if 1 < L:
    assert TAIL_VALUE < min(res)
    res[-1] = TAIL_VALUE
  assert L == len(res)
  return np.array(res)

ENCODED_BODY = [[]] + [_encodeBody(L) for L in range(1, 64)]
###############
# map 1d cells to 2d
def _createWrappedCoords():
  ZERO_POINT_SHIFT = np.array([(3 * MAX_DIM - p) // 2 for p in BASIC_GRID_SHAPE])
  res = np.zeros((3 * MAX_DIM, 3 * MAX_DIM, 2))
  for i, p in enumerate(ZERO_POINT_SHIFT):
    for x in range(3 * MAX_DIM):
      for y in range(3 * MAX_DIM):
        d = (x, y)[i]
        res[x, y, i] = ((d - p)) % BASIC_GRID_SHAPE[i]
  return res.astype(np.int32), ZERO_POINT_SHIFT

def _buildWrappedCoords():
  res = []
  lfrRes = []
  COORDS, ZERO_POINT_SHIFT = _createWrappedCoords()
  D = MAX_DIM // 2
  lfrPos = np.array([(D, D - 1, D), (D - 1, D, D + 1)])
  
  for p in INDEX_TO_COORD:
    X, Y = p + ZERO_POINT_SHIFT
    area = COORDS[X-D:X+D+1, Y-D:Y+D+1]
    for d in range(4):
      xy = np.rot90(area, k=d, axes=(0, 1))
      cX = xy[:, :, 0].reshape(-1)
      cY = xy[:, :, 1].reshape(-1)

      res.append(cX * BASIC_GRID_COLUMNS + cY)
      ####
      lfrXY = xy[lfrPos[0], lfrPos[1]]
      lfrRes.append((lfrXY[:, 0] * BASIC_GRID_COLUMNS) + lfrXY[:, 1])
    ######
  return np.array(res, np.uint8), np.array(lfrRes, np.uint8)

WRAP_COORDS, LFR_COORDS = _buildWrappedCoords()
###############
def gaussianBuilder(size, fwhm):
  x = np.arange(0, size * 2, 1, float)
  y = x[:,np.newaxis]
  centralGaussian = np.exp(-4*np.log(2) * ((x-size)**2 + (y-size)**2) / fwhm**2)

  def f(center):
    dx, dy = np.subtract([size, size], center)
    return centralGaussian[dy:dy+size, dx:dx+size]
  return f

FOOD_GAUSSIAN = gaussianBuilder(MAX_DIM, 5)
###############
LAYER_FOOD = 0
 
PLAYER_OBSERVATION_LAYERS = 3# food, players, ???
EMPTY_PLAYER_OBSERVATION = np.zeros((MAX_DIM, MAX_DIM, PLAYER_OBSERVATION_LAYERS), np.int)
###############
class CPlayerWorldState:
  def __init__(self, state, playerID=0):
    self._state = state
    self._playerID = playerID
    return
  
  def _head(self, i):
    p = self._state[:2, 1 + i]
    rot = int(p[0])
    pos = int(p[1] - 1)
    return rot, pos, rot + (pos * 4)
  
  def validMoves(self):
    world = self._state[2:, 1:]
    rot, _, head = self._head(self._playerID)
    obstacles = TAIL_VALUE < world[LFR_COORDS[head]].max(axis=1)
    actionsMask = 1. - obstacles.astype(np.float)
    return actionsMask, ACTIONS_MAPPING_BY_DIRECTION[rot]
  
  @property
  def raw(self):
    state = self._state.copy()
    if 0 < self._playerID:
      state[:, [1, 1 + self._playerID]] = state[:, [1 + self._playerID, 1]]
    return state
  
  @property
  def normalized(self):
    if 0 == self._playerID: return self
    return CPlayerWorldState(self.raw, playerID=0)
  
  def alive(self, i):
    _, pos, _ = self._head(i)
    return 0 <= pos

  def view(self, playerID=None):
    playerID = self._playerID if playerID is None else playerID
    if not self.alive(playerID): return EMPTY_PLAYER_OBSERVATION
    
    flatworld = self._state[2:]
    _, _, head = self._head(playerID)
    gridworld = flatworld[WRAP_COORDS[head]].view().reshape((MAX_DIM, MAX_DIM, -1))
    
    res = np.zeros(EMPTY_PLAYER_OBSERVATION.shape, np.float)
    res[:, :, LAYER_FOOD] = self._encodeFoodCentroids(gridworld[:, :, LAYER_FOOD])
    res[:, :, 1] = gridworld[:, :, 1:].sum(axis=-1)
    # cutoff head
    res[MAX_DIM//2, MAX_DIM//2, 1:] = 0
    return res
  
  def RGB(self):
    res = np.zeros((MAX_DIM, MAX_DIM, 3), np.float)
    if not self.alive(self._playerID): return res
    
    flatworld = self._state[2:]
    _, _, head = self._head(self._playerID)
    if 0 < self._playerID:
      flatworld[:, [1, 1 + self._playerID]] = flatworld[:, [1 + self._playerID, 1]]
      
    gridworld = flatworld[WRAP_COORDS[head]].view().reshape((MAX_DIM, MAX_DIM, -1))

    res[:, :, 0] = gridworld[:, :, LAYER_FOOD]
    res[:, :, 1] = gridworld[:, :, 1]
    res[:, :, 2] = gridworld[:, :, 2:].sum(axis=-1)
    return res
  
  def _encodeFoodCentroids(self, res):
    foodY, foodX = np.nonzero(res)
    for foodPos in zip(foodX, foodY):
      gaussian = FOOD_GAUSSIAN(foodPos)
      res = np.maximum(res, gaussian)
    return res
  
  def remapOffset(self, offset, playerID):
    offset = np.add(offset, (MAX_DIM // 2, MAX_DIM // 2))
    if playerID == self._playerID: return offset
    
    offsetIndex = (offset[0] * MAX_DIM) + offset[1]
    
    _, _, headSrc = self._head(playerID)
    remappedCell = WRAP_COORDS[headSrc][offsetIndex]
    
    _, _, headDest = self._head(self._playerID)
    coordsDest = WRAP_COORDS[headDest].view().reshape((MAX_DIM, MAX_DIM))
    return np.where(remappedCell == coordsDest)
  
  def remapGrid(self, grid, playerID):
    if playerID == self._playerID: return grid
    
    _, _, headSrc = self._head(playerID)
    remapped = WRAP_COORDS[headSrc]
    flatworld = np.empty((BASIC_GRID_SIZE,), np.float)
    flatworld[remapped] = grid.view().reshape((-1))
    
    _, _, headDest = self._head(self._playerID)
    gridworld = flatworld[WRAP_COORDS[headDest]].view().reshape((MAX_DIM, MAX_DIM, -1))
    return gridworld
  
  @property
  def ID(self):
    return self._playerID
  
####################
def GlobalState(food, geese, rotations):
  res = np.zeros_like(EMPTY_RAW_OBSERVATION, np.float)
  res[np.add(2, food), LAYER_FOOD] = 1

  for i, goose in enumerate(geese):
    if goose:
      res[0, i + 1] = rotations[i]
      res[1, i + 1] = goose[0] + 1
      res[np.add(2, goose), i + 1] = ENCODED_BODY[len(goose)]
  return res

class CGlobalWorldState:
  def __init__(self, state):
    self._state = state
    return
  
  def player(self, playerID):
    return CPlayerWorldState(self._state, playerID)
#################
class CWorldState:
  def __init__(self):
    self._prevHeads = [None] * 4
    return
  
  def update(self, observation):
    food = observation.food
    geese = observation.geese
    rotations = [ROTATIONS_FOR_ACTION[act] for act in self._updateHeads(geese)]
    self._state = CGlobalWorldState(GlobalState(food, geese, rotations))
    return
  
  def player(self, playerID):
    return self._state.player(playerID)
  
  def _updateHeads(self, geese):
    prevActions = []
    for i, goose in enumerate(geese):
      cur = 'NORTH'
      if 0 < len(goose):
        prev = self._prevHeads[i]
        self._prevHeads[i] = curPos = goose[0]
        if not(prev is None):
          oldGooseRow, oldGooseCol = INDEX_TO_COORD[prev]
          newGooseRow, newGooseCol = INDEX_TO_COORD[curPos]
          cur = DELTA_TO_ACTION[(
            (newGooseCol - oldGooseCol) % BASIC_GRID_COLUMNS,
            (newGooseRow - oldGooseRow) % BASIC_GRID_ROWS,
          )].name
      prevActions.append(cur)
    
    return prevActions