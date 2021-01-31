from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, row_col
import numpy as np
from discountedWaves import discountedWaves
import math

class CAgent:
  INDEX_TO_ACTION = ['WEST', 'NORTH', 'EAST', 'SOUTH']
  
  ROTATIONS_FOR_ACTION = {
    'EAST': 1, 'WEST': 3,
    'SOUTH': 2, 'NORTH': 0, 
  }
  
  EMPTY_OBSERVATION = None
  ZERO_POINT_SHIFT = None
  MAX_DIM = None
  
  def __init__(self, model=None, FOV=None):
    self._fieldOfView = FOV
    self._model = model
    return
  
  def reset(self):
    self._prevAction = 'NORTH'
    return
  
  def preprocessObservations(self, obs_dict, config_dict):
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
    CAgent.MAX_DIM = max(grid.shape[1:])
    CAgent.ZERO_POINT_SHIFT = np.array([(3 * CAgent.MAX_DIM - p) // 2 for p in grid.shape[1:]])
      
    grid = np.pad(
      grid,
      pad_width=[(0, 0)] + [(x, x) for x in CAgent.ZERO_POINT_SHIFT],
      mode='wrap'
    )
    # build distances map
    for i, goose in enumerate(observation.geese):
      if 0 < len(goose):
        head = CAgent.ZERO_POINT_SHIFT + row_col(goose[0], configuration.columns)
        grid[2 + i] = discountedWaves(grid[0], start=tuple(head))
    return grid
  
  def encodeObservations(self, obs_dict, config_dict, grid):
    observation = Observation(obs_dict)
    configuration = Configuration(config_dict)
    # current player at layer 2
    playerID = observation.index
    if 0 < playerID:
      grid[[2, 2 + playerID]] = grid[[2 + playerID, 2]]
    #######
    px, py = CAgent.ZERO_POINT_SHIFT + row_col(observation.geese[playerID][0], configuration.columns)
    d = CAgent.MAX_DIM // 2 if self._fieldOfView is None else self._fieldOfView
    
    grid = np.rot90(
      grid[:, px-d:px+d+1, py-d:py+d+1],
      k=self.ROTATIONS_FOR_ACTION[self._prevAction],
      axes=(1, 2)
    )
    
    playerPos = (grid.shape[1] // 2, grid.shape[2] // 2)
    actionsMask = np.array([
      grid[0, playerPos[0] - 1, playerPos[1]] < 1, # L
      grid[0, playerPos[0], playerPos[1] - 1] < 1, # F
      grid[0, playerPos[0] + 1, playerPos[1]] < 1, # R
    ]).astype(np.float)
  
    actN = len(self.INDEX_TO_ACTION)
    prevAct = next(i for i, v in enumerate(self.INDEX_TO_ACTION) if v == self._prevAction) + actN
    actionsMapping = [
      self.INDEX_TO_ACTION[(prevAct - 1) % actN],
      self.INDEX_TO_ACTION[(prevAct - 0) % actN],
      self.INDEX_TO_ACTION[(prevAct + 1) % actN],
    ]
    
    if self.EMPTY_OBSERVATION is None: # for training
      self.EMPTY_OBSERVATION = np.zeros_like(grid) 
    return (grid, actionsMask, actionsMapping)
  
  def _predict(self, states):
    return self._model.predict(np.array(states))
  
  # only for Kaggle
  def play(self, obs_dict, config_dict):
    grid = self.processObservations(
      obs_dict, config_dict,
      self.preprocessObservations(obs_dict, config_dict)
    )
    QValues = self._predict([grid])
    action, _ = self.choiceAction(QValues[0])
    return action
  
  def processObservations(self, obs_dict, config_dict, grid, alive=True):
    if not alive: return self.EMPTY_OBSERVATION
    grid, self._actionsMask, self._actionsMapping = self.encodeObservations(obs_dict, config_dict, grid)
    return grid
  
  def choiceAction(self, QValues):
    QValues[0 == self._actionsMask] = -math.inf
    actID = QValues.argmax(axis=-1)
    self._prevAction = self._actionsMapping[actID]
    return (self._prevAction, actID)