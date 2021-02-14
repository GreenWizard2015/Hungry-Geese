from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, \
GreedyAgent, Action
from .CAgentState import CAgentState, GlobalObservations

class CGreedyAgent:
  def __init__(self):
    return
  
  def reset(self):
    self._state = CAgentState()
    self._agent = None
    return
  
  def preprocessObservations(self, obs_dict, config_dict):
    observation = Observation(obs_dict)
    configuration = Configuration(config_dict)
    return GlobalObservations(observation, configuration)
  
  def encodeObservations(self, obs_dict, config_dict, gstate):
    observation = Observation(obs_dict)
    configuration = Configuration(config_dict)
    return self._state.local(observation, configuration, gstate)

  def processObservations(self, obs_dict, config_dict, grid, alive=True):
    if not alive: return self._state.EmptyObservations
    
    observation = Observation(obs_dict)
    configuration = Configuration(config_dict)
    
    if self._agent is None:
      self._agent = GreedyAgent(configuration)
      
    state, validAct, actionsMapping = self.encodeObservations(obs_dict, config_dict, grid)
    
    self._agent.last_action = Action[self._state.last_action] # sync
    self._actName = self._agent(observation)
    if self._actName not in actionsMapping:
      self._actName = next(
        (act for i, act in enumerate(actionsMapping) if 0 < validAct[i]),
        actionsMapping[1]
      )
    self._actID = actionsMapping.index(self._actName)
    return state
  
  def choiceAction(self, QValues):
    self._state.perform(self._actName)
    return (self._actName, self._actID)
  
  # only for Kaggle
  def play(self, obs_dict, config_dict):
    grid = self.processObservations(
      obs_dict, config_dict,
      self.preprocessObservations(obs_dict, config_dict)
    )
    action, _ = self.choiceAction(None)
    return action, grid
  
  @property
  def kind(self):
    return 'Greedy'