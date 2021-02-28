from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, \
GreedyAgent, Action
from .CAgentState import CAgentState

class CReplayAgent:
  def __init__(self):
    self._state = CAgentState()
    return
  
  def reset(self):
    self._state = CAgentState()
    return

  def encodeObservations(self, obs_dict, config_dict, details=False):
    observation = Observation(obs_dict)
    configuration = Configuration(config_dict)
    return self._state.local(observation, configuration, details)

  def processObservations(self, obs_dict, config_dict, alive=True, details=False):
    if not alive: return self._state.EmptyObservations
    
    obs = self.encodeObservations(obs_dict, config_dict, details)
    state, _, actionsMapping = obs[:3]
    
    self._actName = obs_dict['next action']
    self._actID = actionsMapping.index(self._actName)
    
    if details:
      return(state, obs[-1])
    return state
  
  def choiceAction(self, _):
    self._state.perform(self._actName)
    return (self._actName, self._actID)
  
  def play(self, obs_dict, config_dict):
    grid = self.processObservations(obs_dict, config_dict)
    action, _ = self.choiceAction(None)
    return action, grid
  
  @property
  def kind(self):
    return 'Replay'