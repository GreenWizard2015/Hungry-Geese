from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, \
GreedyAgent, Action
from .CAgentState import CAgentState

class CReplayAgent:
  def __init__(self, world):
    self._state = CAgentState(world)
    return

  def encodeObservations(self, obs_dict, config_dict):
    observation = Observation(obs_dict)
    configuration = Configuration(config_dict)
    return self._state.local(observation, configuration)

  def processObservations(self, obs_dict, config_dict, alive=True):
    if not alive: return self._state.EmptyObservations
    
    obs = self.encodeObservations(obs_dict, config_dict)
    state, _, actionsMapping = obs[:3]
    
    self._actName = obs_dict['next action']
    self._actID = actionsMapping.index(self._actName)
    return state
  
  def choiceAction(self, _):
    return (self._actName, self._actID)

  @property
  def kind(self):
    return 'Replay'