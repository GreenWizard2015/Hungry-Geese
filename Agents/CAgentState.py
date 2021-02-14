from kaggle_environments.envs.hungry_geese.hungry_geese import row_col
import numpy as np
from .discountedWaves import discountedWaves

PLAYERS_N = 4
LAYERS_PER_PLAYER = 4

BASIC_GRID_SHAPE = (7, 11)
MAX_DIM = max(BASIC_GRID_SHAPE)
ZERO_POINT_SHIFT = np.array([(3 * MAX_DIM - p) // 2 for p in BASIC_GRID_SHAPE])

LAYER_OBSTACLES = 0
LAYER_FOOD = 1
EMPTY_OBSERVATION = np.zeros((11, 11, 3))

def expandPoints(indexedPoints, shift=[0, 0]):
  points = np.array([ row_col(x, BASIC_GRID_SHAPE[1]) for x in indexedPoints ]) + shift
  X = points[:, 0]
  Y = points[:, 1]
  return(X, Y)

def GlobalObservations(observation, configuration):
  assert np.array_equal(BASIC_GRID_SHAPE, (configuration.rows, configuration.columns))
  ############
  # 0 - obstacles
  # 1 - food
  # 2+N*x+0 - head of player N
  # 2+N*x+1 - body of player N (1 - head, L - tail)
  # 2+N*x+2 - distances from head of player N
  # 2+N*x+3 - ???
  grid = np.zeros((2 + LAYERS_PER_PLAYER * PLAYERS_N, configuration.rows, configuration.columns), np.float)
  
  ptsX, ptsY = expandPoints(observation.food)
  grid[LAYER_FOOD, ptsX, ptsY] = 1
  # add geese
  for playerIndex, goose in enumerate(observation.geese):
    if len(goose) <= 0: continue
    
    bodyX, bodyY = expandPoints(goose)
    
    grid[LAYER_OBSTACLES, bodyX, bodyY] = 1 # obstacles
    ind = 2 + playerIndex * LAYERS_PER_PLAYER
    grid[ind + 0, bodyX[0], bodyY[0]] = len(goose) # head
    grid[ind + 1, bodyX, bodyY] = 1 + np.arange(len(bodyX)) # body
  # wrap map and make it square
  grid = np.pad(
    grid,
    pad_width=[(0, 0)] + [(x, x) for x in ZERO_POINT_SHIFT],
    mode='wrap'
  )
  # build distances map
#   for playerIndex, goose in enumerate(observation.geese):
#     if len(goose) <= 0: continue
#   
#     bodyX, bodyY = expandPoints(goose, shift=ZERO_POINT_SHIFT)
#     X, Y = bodyX[0], bodyY[0]
#     grid[2 + playerIndex * 4 + 2] = discountedWaves(
#       grid[LAYER_OBSTACLES], start=(X, Y), N=MAX_DIM
#     )
  ###
  return grid

RENDER_COLORS = {
  'enemy head future': (0, 128, 128),
  'enemy head': (0, 128, 255),
  'enemy body': (0, 64, 64),
  'food': (255, 255, 255),
  'player body': (128, 128, 128),
  'player head': (0, 255, 0)
}

def state2RGB(state, playerID):
  res = [np.zeros((*state.shape[1:], 3))]
  def masked(mask, color):
    res[0][0 < state[mask]] = RENDER_COLORS[color]
    return
  
  res[0][:, :, 0] = 64 * state[2+playerID*4+2]
  masked(LAYER_FOOD, 'food')

  heads = np.zeros(state.shape[1:])
  for i in range(4):
    if not (i == playerID): # enemies
      masked(2+i*4+1, 'enemy body')
      heads += state[2+i*4+0]
  
  headsReal = heads.copy()
  # left-right
  heads[:-1, :] += headsReal[1:, :]
  heads[1:, :] += headsReal[:-1, :]
  # up-down
  heads[:, :-1] += headsReal[:, 1:]
  heads[:, 1:] += headsReal[:, :-1]
  res[0][0 < heads] = RENDER_COLORS['enemy head future']
  res[0][0 < headsReal] = RENDER_COLORS['enemy head']
  ############
  masked(2+playerID*4+1, 'player body')
  c = state.shape[1] // 2
  ############
  res[0][c, c] = RENDER_COLORS['player head']
  # encode length at 0, 0
  res[0][0, 0, :] = 255. / state[2+playerID*4, c, c]
  return res[0] / 255.

def LocalObservations(observation, configuration, globalState):
  player = observation.index
  body = observation.geese[player]
  if len(body) <= 0:
    return None
  
  bodyX, bodyY = expandPoints(body, shift=ZERO_POINT_SHIFT)
  X, Y = bodyX[0], bodyY[0]
  d = MAX_DIM // 2
  
  state = globalState[:, X-d:X+d+1, Y-d:Y+d+1]

  state[2 + player * 4 + 2] = discountedWaves(
    state[LAYER_OBSTACLES], start=(d, d), N=MAX_DIM
  )
  return state

class CAgentState:
  INDEX_TO_ACTION = ['WEST', 'NORTH', 'EAST', 'SOUTH']
  
  ROTATIONS_FOR_ACTION = {
    'EAST': 1, 'WEST': 3,
    'SOUTH': 2, 'NORTH': 0, 
  }
  
  ###########################################
  def __init__(self):
    self._prevAction = 'NORTH'
    return
  
  def local(self, observation, configuration, gstate):
    grid = LocalObservations(observation, configuration, gstate)
    grid = np.rot90(
      grid,
      k=self.ROTATIONS_FOR_ACTION[self._prevAction],
      axes=(1, 2)
    )
    
    playerPos = (grid.shape[1] // 2, grid.shape[2] // 2)
    actionsMask = np.array([
      grid[0, playerPos[0], playerPos[1] - 1] < 1, # L
      grid[0, playerPos[0] - 1, playerPos[1]] < 1, # F
      grid[0, playerPos[0], playerPos[1] + 1] < 1, # R
    ]).astype(np.float)
  
    actN = len(self.INDEX_TO_ACTION)
    prevAct = next(i for i, v in enumerate(self.INDEX_TO_ACTION) if v == self._prevAction) + actN
    actionsMapping = [
      self.INDEX_TO_ACTION[(prevAct - 1) % actN],
      self.INDEX_TO_ACTION[(prevAct - 0) % actN],
      self.INDEX_TO_ACTION[(prevAct + 1) % actN],
    ]
    
    RGB = state2RGB(grid, observation.index)
    return RGB, actionsMask, actionsMapping

  def perform(self, action):
    self._prevAction = action
    return
  
  @property
  def EmptyObservations(self):
    return EMPTY_OBSERVATION
  
  @property
  def last_action(self):
    return self._prevAction