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
LAYER_HEAD, LAYER_BODY,  LAYER_TAIL, LAYER_MOVING = np.arange(LAYERS_PER_PLAYER)

LO_SHAPE = (2 + LAYERS_PER_PLAYER * 2, MAX_DIM, MAX_DIM)
LO_PLAYER = 2
LO_ENEMIES = 2 + LAYERS_PER_PLAYER
LO_EMPTY_OBSERVATION = np.zeros(LO_SHAPE)

RGB_EMPTY_OBSERVATION = np.zeros((11, 11, 3))

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
  # 2+N*x+2 - tails
  # 2+N*x+3 - distances from head of player N
  grid = np.zeros((2 + LAYERS_PER_PLAYER * PLAYERS_N, configuration.rows, configuration.columns), np.float)
  
  ptsX, ptsY = expandPoints(observation.food)
  grid[LAYER_FOOD, ptsX, ptsY] = 1
  # add geese
  for playerIndex, goose in enumerate(observation.geese):
    if len(goose) <= 0: continue
    
    bodyX, bodyY = expandPoints(goose)
    
    grid[LAYER_OBSTACLES, bodyX, bodyY] = 1 # obstacles
    ind = 2 + playerIndex * LAYERS_PER_PLAYER
    grid[ind + LAYER_HEAD, bodyX[0], bodyY[0]] = len(goose) # head
    
    ebody = np.arange(len(bodyX)) 
    ebody = np.maximum(1 + ebody, len(bodyX) - ebody) / len(bodyX)
    grid[ind + LAYER_BODY, bodyX, bodyY] = ebody # body
    
    grid[ind + LAYER_TAIL, bodyX[-1], bodyY[-1]] = len(goose) # tail
  # wrap map and make it square
  grid = np.pad(
    grid,
    pad_width=[(0, 0)] + [(x, x) for x in ZERO_POINT_SHIFT],
    mode='wrap'
  )
  # build distances map
  # disabled
#   for playerIndex, goose in enumerate(observation.geese):
#     if len(goose) <= 0: continue
#    
#     bodyX, bodyY = expandPoints(goose, shift=ZERO_POINT_SHIFT)
#     X, Y = bodyX[0], bodyY[0]
#     grid[2 + playerIndex * LAYERS_PER_PLAYER + LAYER_MOVING] = discountedWaves(
#       grid[LAYER_OBSTACLES], start=(X, Y), N=2+(MAX_DIM//2)
#     )
  ###
  return grid

RENDER_COLORS = {
  'food': (255, 255, 255),
  
  'enemy head future': (0, 128, 128),
  'enemy head': (0, 128, 255),
  'enemy tail': (0, 128, 128),
  'enemy body': (0, 64, 64),
  
  'player body': (128, 128, 128),
  'player head': (0, 255, 0),
  'player tail': (0, 200, 0),
}

def state2RGB(state):
  res = [np.zeros((*state.shape[1:], 3))]
  def masked(mask, color):
    res[0][0 < state[mask]] = RENDER_COLORS[color]
    return
  
  res[0][:, :, 0] = 64 * state[LO_PLAYER + LAYER_MOVING]
  res[0][:, :, 2] = 64 * state[LO_ENEMIES + LAYER_MOVING]
  masked(LAYER_FOOD, 'food')

  ############
  masked(LO_PLAYER + LAYER_BODY, 'player body')
  masked(LO_PLAYER + LAYER_TAIL, 'player tail')
  masked(LO_PLAYER + LAYER_HEAD, 'player head')
  ############
  ############
  masked(LO_ENEMIES + LAYER_BODY, 'enemy body')
  masked(LO_ENEMIES + LAYER_TAIL, 'enemy tail')
  masked(LO_ENEMIES + LAYER_HEAD, 'enemy head')
  ############
  
  heads = state[LO_ENEMIES + LAYER_HEAD].copy()
  headsReal = heads.copy()
  # left-right
  heads[:-1, :] += headsReal[1:, :]
  heads[1:, :] += headsReal[:-1, :]
  # up-down
  heads[:, :-1] += headsReal[:, 1:]
  heads[:, 1:] += headsReal[:, :-1]
  res[0][0 < heads] = RENDER_COLORS['enemy head future']
  res[0][0 < headsReal] = RENDER_COLORS['enemy head']
  
  return res[0] / 255.

def LocalObservations(observation, configuration, globalState):
  player = observation.index
  body = observation.geese[player]
  if len(body) <= 0:
    return None
  
  (X, ), (Y, ) = expandPoints(body[:1], shift=ZERO_POINT_SHIFT)
  d = MAX_DIM // 2
  
  state = globalState[:, X-d:X+d+1, Y-d:Y+d+1]
  playersStates = [
    state[2+i*LAYERS_PER_PLAYER:][:LAYERS_PER_PLAYER] for i in range(PLAYERS_N)
  ]
  
  res = np.zeros(LO_SHAPE)
  res[:2] = state[:2]
  res[2:2+LAYERS_PER_PLAYER] = playersStates[player]
  del playersStates[player]
  
  enemies = [x for x in playersStates if np.any(x[0])]
  res[LO_ENEMIES+3] = 1
  for enemy in enemies:
    res[LO_ENEMIES:][:3] += enemy[:3]
    # combine distances
    res[LO_ENEMIES+3] = np.minimum(enemy[3], res[LO_ENEMIES+3])

  return res

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
  
  def local(self, observation, configuration, gstate, details=False):
    state = LocalObservations(observation, configuration, gstate)
    state = np.rot90(
      state,
      k=self.ROTATIONS_FOR_ACTION[self._prevAction],
      axes=(1, 2)
    )
    
    playerPos = (state.shape[1] // 2, state.shape[2] // 2)
    actionsMask = np.array([
      state[LAYER_OBSTACLES, playerPos[0], playerPos[1] - 1] < 1, # L
      state[LAYER_OBSTACLES, playerPos[0] - 1, playerPos[1]] < 1, # F
      state[LAYER_OBSTACLES, playerPos[0], playerPos[1] + 1] < 1, # R
    ]).astype(np.float)
    # TODO: Check if obstacle is a tail, which move away
  
    actN = len(self.INDEX_TO_ACTION)
    prevAct = next(i for i, v in enumerate(self.INDEX_TO_ACTION) if v == self._prevAction) + actN
    actionsMapping = [
      self.INDEX_TO_ACTION[(prevAct - 1) % actN],
      self.INDEX_TO_ACTION[(prevAct - 0) % actN],
      self.INDEX_TO_ACTION[(prevAct + 1) % actN],
    ]
    
    results = [state, actionsMask, actionsMapping]
    if details:
      results.append(self._details(
        observation, configuration, gstate,
        playerPos, state, actionsMask, actionsMapping
      ))
    return results

  def _details(self,
    observation, configuration, gstate,
    playerPos, state, actionsMask, actionsMapping
  ):
    player = observation.index
    body = observation.geese[player]
    ########
    foodVectors = np.zeros((4, 2), np.float16)
    foods = list(zip(*np.nonzero(state[LAYER_FOOD])))
    for i, food in enumerate(foods[:foodVectors.shape[0]]):
      food = np.subtract(food, playerPos)
      foodVectors[i] = food / np.linalg.norm(food)
    ########
    X, Y = (playerPos[0] + 1, playerPos[1])
    d = 2
    prevObstacles = (0 < state[LO_ENEMIES+LAYER_BODY, X-d:X+d+1, Y-d:Y+d+1]).astype(np.uint8)
    return {
      'starve': 1 if 1 == len(body) else 0,
      'food vectors': foodVectors,
      'prev obstacles': prevObstacles
    }
    
  def perform(self, action):
    self._prevAction = action
    return
  
  @property
  def last_action(self):
    return self._prevAction
  
  @property
  def EmptyObservations(self):
    return LO_EMPTY_OBSERVATION