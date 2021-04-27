import math
import numpy as np
from .CAgentState import CAgentState
from Agents.CWorldState import CWorldState, CGlobalWorldState
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation

def restoreStates(states):
  restored = []
  for s in states:
    restored.append(CGlobalWorldState(s).player(0).view())
  return np.array(restored)

class CAgent:
  def __init__(self, world, model=None, kind='network'):
    self._model = model
    self.kind = kind
    world = CWorldState() if world is None else world # Kaggle
    self._world = world
    self._state = CAgentState(world)
    return

  def _predict(self, states):
    res = self._model(restoreStates(states))
    res = res[0] if isinstance(res, list) else res
    return res.numpy()
  
  def play(self, obs_dict, config_dict):
    grid = self.processObservations(obs_dict, config_dict)
    QValues = self._predict([grid])[0]
    action, _ = self.choiceAction(QValues)
    return action, grid
  
  # only for Kaggle
  def __call__(self, obs_dict, config_dict):
    self._world.update(Observation(obs_dict))
    return self.play(obs_dict, config_dict)[0]
  
  def processObservations(self, obs_dict, config_dict, alive=True):
    if not alive: return self._state.EmptyObservations
    
    state, self._actionsMask, self._actionsMapping = self._state.local(obs_dict, config_dict)
    return state
  
  def choiceAction(self, QValues):
    QValues[0 == self._actionsMask] = -math.inf
    actID = QValues.argmax(axis=-1)
    actName = self._actionsMapping[actID]
    return (actName, actID)