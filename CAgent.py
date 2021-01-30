from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, row_col
import numpy as np
from discountedWaves import discountedWaves
import math

INDEX_TO_ACTION = ['WEST', 'NORTH', 'EAST', 'SOUTH']

ROTATIONS_FOR_ACTION = {
  'EAST': 1, 'WEST': 3,
  'SOUTH': 2, 'NORTH': 0, 
}

EMPTY_OBSERVATION = None

class CAgent:
  def __init__(self, model=None, FOV=None):
    self._fieldOfView = FOV
    self._model = model
    return
  
  def reset(self):
    self._prevAction = 'NORTH'
    return
  
  def encodeObservations(self, obs_dict, config_dict):
    observation = Observation(obs_dict)
    configuration = Configuration(config_dict)
    ############
    # 0 - obstacles, 1 - food, 2..2+N - player 1..N
    grid = np.zeros((6, configuration.rows, configuration.columns), np.float)
    
    # let's add food to the map
    for food in observation.food:
      x,y = row_col(food, configuration.columns)
      grid[1, x, y] = 1
      
    # add geese
    for i, goose in enumerate(observation.geese):
      for n in goose:
        x,y = row_col(n, configuration.columns)
        grid[0, x, y] = 1 # global obstacles map
        #grid[2 + i, x, y] = 1 # player specific layer
    # wrap map and make it square
    maxDim = max(grid.shape[1:])
    zeroPtShift = np.array([(3 * maxDim - p) // 2 for p in grid.shape[1:]])
    grid = np.pad(
      grid,
      pad_width=[(0, 0)] + [(x, x) for x in zeroPtShift],
      mode='wrap'
    )
    # build distances map
    for i, goose in enumerate(observation.geese):
      if 0 < len(goose):
        head = zeroPtShift + row_col(goose[0], configuration.columns)
        grid[2 + i] = discountedWaves(grid[0], start=tuple(head))
    # current player at layer 2
    playerID = observation.index
    if 0 < playerID:
      grid[[2, 2 + playerID]] = grid[[2 + playerID, 2]]
    #######
    px, py = zeroPtShift + row_col(observation.geese[playerID][0], configuration.columns)
    d = maxDim // 2 if self._fieldOfView is None else self._fieldOfView
    
    grid = np.rot90(
      grid[:, px-d:px+d+1, py-d:py+d+1],
      k=ROTATIONS_FOR_ACTION[self._prevAction],
      axes=(1, 2)
    )
    
    playerPos = (grid.shape[1] // 2, grid.shape[2] // 2)
    actionsMask = np.array([
      grid[0, playerPos[0] - 1, playerPos[1]] < 1, # L
      grid[0, playerPos[0], playerPos[1] - 1] < 1, # F
      grid[0, playerPos[0] + 1, playerPos[1]] < 1, # R
    ]).astype(np.float)
  
    actN = len(INDEX_TO_ACTION)
    prevAct = next(i for i, v in enumerate(INDEX_TO_ACTION) if v == self._prevAction) + actN
    actionsMapping = [
      INDEX_TO_ACTION[(prevAct - 1) % actN],
      INDEX_TO_ACTION[(prevAct - 0) % actN],
      INDEX_TO_ACTION[(prevAct + 1) % actN],
    ]
    
    global EMPTY_OBSERVATION
    if EMPTY_OBSERVATION is None: # for training
      EMPTY_OBSERVATION = np.zeros_like(grid) 
    return (grid, actionsMask, actionsMapping)
  
  def _predict(self, states):
    return self._model.predict(np.array(states))
  
  # only for Kaggle
  def play(self, obs_dict, config_dict):
    grid = self.processObservations(obs_dict, config_dict)
    QValues = self._predict([grid])
    action, _ = self.choiceAction(QValues[0])
    return action
  
  def processObservations(self, obs_dict, config_dict, alive=True):
    if not alive: return EMPTY_OBSERVATION
    grid, self._actionsMask, self._actionsMapping = self.encodeObservations(obs_dict, config_dict)
    return grid
  
  def choiceAction(self, QValues):
    QValues[0 == self._actionsMask] = -math.inf
    actID = QValues.argmax(axis=-1)
    self._prevAction = self._actionsMapping[actID]
    return (self._prevAction, actID)