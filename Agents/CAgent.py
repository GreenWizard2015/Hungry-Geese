from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration
import numpy as np
import math
from .CAgentState import CAgentState

class CAgent:
  def __init__(self, model=None, kind='network'):
    self._model = model
    self.kind = kind
    return
  
  def reset(self):
    self._state = CAgentState()
    return
  
  def encodeObservations(self, obs_dict, config_dict, details=False):
    observation = Observation(obs_dict)
    configuration = Configuration(config_dict)
    return self._state.local(observation, configuration, details)
  
  def _predict(self, states):
    return self._model.predict(np.array(states))
  
  def play(self, obs_dict, config_dict):
    grid = self.processObservations(obs_dict, config_dict)
    QValues = self._predict([grid])[0]
    action, _ = self.choiceAction(QValues[0])
    return action, grid
  
  # only for Kaggle
  def __call__(self, obs_dict, config_dict):
    return self.play(obs_dict, config_dict)[0]
  
  def processObservations(self, obs_dict, config_dict, alive=True, details=False):
    if not alive: return self._state.EmptyObservations
    
    obs = self.encodeObservations(obs_dict, config_dict, details)
    state, self._actionsMask, self._actionsMapping = obs[:3]
    if details:
      return(state, obs[-1])
    return state
  
  def choiceAction(self, QValues):
    QValues[0 == self._actionsMask] = -math.inf
    actID = QValues.argmax(axis=-1)
    actName = self._actionsMapping[actID]
    self._state.perform(actName)
    return (actName, actID)