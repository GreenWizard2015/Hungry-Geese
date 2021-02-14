from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration
import numpy as np
import math
from .CAgentState import CAgentState, GlobalObservations

class CAgent:
  def __init__(self, model=None, kind='network'):
    self._model = model
    self.kind = kind
    return
  
  def reset(self):
    self._state = CAgentState()
    return
  
  def preprocessObservations(self, obs_dict, config_dict):
    observation = Observation(obs_dict)
    configuration = Configuration(config_dict)
    return GlobalObservations(observation, configuration)
  
  def encodeObservations(self, obs_dict, config_dict, gstate):
    observation = Observation(obs_dict)
    configuration = Configuration(config_dict)
    return self._state.local(observation, configuration, gstate)
  
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
    return action, grid
  
  def processObservations(self, obs_dict, config_dict, grid, alive=True):
    if not alive: return self._state.EmptyObservations
    
    grid, self._actionsMask, self._actionsMapping = self.encodeObservations(obs_dict, config_dict, grid)
    return grid
  
  def choiceAction(self, QValues):
    QValues[0 == self._actionsMask] = -math.inf
    actID = QValues.argmax(axis=-1)
    actName = self._actionsMapping[actID]
    self._state.perform(actName)
    return (actName, actID)