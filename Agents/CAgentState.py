from kaggle_environments.envs.hungry_geese.hungry_geese import row_col
import numpy as np

PLAYERS_N = 4
LAYERS_N = 4
LAYER_FOOD, LAYER_OBSTACLES, LAYER_PLAYER, LAYER_ENEMIES = np.arange(LAYERS_N)

BASIC_GRID_SHAPE = (7, 11)
MAX_DIM = max(BASIC_GRID_SHAPE)

EMPTY_OBSERVATION = np.zeros((11, 11, LAYERS_N), np.int)

INDEX_TO_COORD = np.array([
  row_col(x, BASIC_GRID_SHAPE[1]) for x in range(BASIC_GRID_SHAPE[0] * BASIC_GRID_SHAPE[1])
])

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
  COORDS, ZERO_POINT_SHIFT = _createWrappedCoords()
  D = MAX_DIM // 2
  for p in INDEX_TO_COORD:
    X, Y = p + ZERO_POINT_SHIFT
    area = COORDS[X-D:X+D+1, Y-D:Y+D+1]
    for d in range(4):
      xy = np.rot90(area, k=d, axes=(0, 1))
      cX = xy[:, :, 0].view().reshape(-1)
      cY = xy[:, :, 1].view().reshape(-1)

      res.append((cX, cY))
  return res

WRAP_COORDS = _buildWrappedCoords()

def _encodeBody(L):
  zzIndex = [1] * L
  for i in range(L):
    if 0 == (i & 0x1):
      zzIndex[i] = 1 + i
    else:
      zzIndex[i] = 1 + L - i
  return np.array(zzIndex).reshape(-1)
  
ENCODED_GEESE = [ _encodeBody(i) for i in range(128)]

def _encodeGoose(res, goose, isEnemy):
  res[goose, LAYER_OBSTACLES] = ENCODED_GEESE[len(goose)]
  res[goose, LAYER_ENEMIES if isEnemy else LAYER_PLAYER] = ENCODED_GEESE[len(goose)]
  return res
  
def _encodeCentered(res, rot, headIndex):
  COORDS = WRAP_COORDS[headIndex*4 + rot] 
  return res[COORDS[0], COORDS[1]].view().reshape((MAX_DIM, MAX_DIM, LAYERS_N))

def _encode(food, player, enemies, rot):
  res = np.zeros((BASIC_GRID_SHAPE[0] * BASIC_GRID_SHAPE[1], LAYERS_N))
  
  res[food, LAYER_FOOD] = 255
  for goose in enemies:
    res = _encodeGoose(res, goose, isEnemy=True)
  res = _encodeGoose(res, player, isEnemy=False)
  res = _encodeCentered(res.view().reshape((*BASIC_GRID_SHAPE, LAYERS_N)), rot, player[0])
  return res / 255.0

# def state2RGB(state):
#   res = np.zeros((*state.shape[:2], 3))
#   state = state[:, :, 0]
#   res[0 < state, 1] = (255 - state*255*5)[0 < state]
#   res[0 > state, 2] = (255 - state*-255*5)[0 > state]
#   res[1==state] = (255, 255, 255)
#   return res / 255.0
  
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
  
  def local(self, observation, configuration, details=False):
    playerID = observation.index
    FOOD_COORDS = observation.food
    PLAYERS_COORDS = [ observation.geese[i] for i in range(PLAYERS_N) ]
    ENEMIES_COORDS = [x for i, x in enumerate(PLAYERS_COORDS) if (0 < len(x)) and not (i == playerID)]
    PLAYER_COORDS = PLAYERS_COORDS[playerID]
    #################################
    state = _encode(
      FOOD_COORDS, PLAYER_COORDS, ENEMIES_COORDS,
      self.ROTATIONS_FOR_ACTION[self._prevAction]
    )
    
    playerPos = (state.shape[0] // 2, state.shape[1] // 2)
    LFR = np.abs([
      state[playerPos[0], playerPos[1] - 1, LAYER_OBSTACLES], # L
      state[playerPos[0] - 1, playerPos[1], LAYER_OBSTACLES], # F
      state[playerPos[0], playerPos[1] + 1, LAYER_OBSTACLES], # R
    ])
    
    actionsMask = 1.0 - (0 < LFR).astype(np.float)
    # TODO: Check if obstacle is a tail, which move away
  
    actN = len(self.INDEX_TO_ACTION)
    prevAct = next(i for i, v in enumerate(self.INDEX_TO_ACTION) if v == self._prevAction) + actN
    actionsMapping = [
      self.INDEX_TO_ACTION[(prevAct - 1) % actN],
      self.INDEX_TO_ACTION[(prevAct - 0) % actN],
      self.INDEX_TO_ACTION[(prevAct + 1) % actN],
    ]
    
    results = [state, actionsMask, actionsMapping]
    if True:#details:
      results.append(self._details(
        observation, configuration,
        FOOD_COORDS, PLAYER_COORDS, ENEMIES_COORDS,
        playerPos, state, actionsMask, actionsMapping
      ))
    return results

  def _foodVectors(self, state, playerPos):
    foodVectors = np.zeros((4, 2), np.float16)
    foods = list(zip(*np.nonzero(state[LAYER_FOOD])))
    for i, food in enumerate(foods[:foodVectors.shape[0]]):
      food = np.subtract(food, playerPos)
      foodVectors[i] = food / np.linalg.norm(food)
    return foodVectors
  
  def _details(self,
    observation, configuration,
    FOOD_COORDS, PLAYER_COORDS, ENEMIES_COORDS,
    playerPos, state, actionsMask, actionsMapping
  ):
    return {
      'starve': 1 if 1 == len(PLAYER_COORDS) else 0,
      # 'food vectors': self._foodVectors(state, playerPos),
    }
    
  def perform(self, action):
    self._prevAction = action
    return
  
  @property
  def last_action(self):
    return self._prevAction
  
  @property
  def EmptyObservations(self):
    return EMPTY_OBSERVATION