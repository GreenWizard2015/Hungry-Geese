from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, \
GreedyAgent, Action
from .CAgentState import CAgentState

class CGreedyAgent:
  def __init__(self, world):
    self._state = CAgentState(world)
    self._agent = None
    self._prevAction = 'NORTH'
    return

  def processObservations(self, obs_dict, config_dict, alive=True):
    if not alive: return self._state.EmptyObservations
    
    observation = Observation(obs_dict)
    configuration = Configuration(config_dict)
    
    if self._agent is None:
      self._agent = GreedyAgent(configuration)

    state, validAct, actionsMapping = self._state.local(observation, configuration)
    
    self._agent.last_action = Action[self._prevAction] # sync
    self._actName = self._agent(observation)
    if self._actName not in actionsMapping:
      self._actName = next(
        (act for i, act in enumerate(actionsMapping) if 0 < validAct[i]),
        actionsMapping[1]
      )
    self._actID = actionsMapping.index(self._actName)
    return state
  
  def choiceAction(self, QValues):
    self._prevAction = self._actName
    return (self._actName, self._actID)
  
  def play(self, obs_dict, config_dict):
    grid = self.processObservations(obs_dict, config_dict)
    action, _ = self.choiceAction(None)
    return action, grid
  
  @property
  def kind(self):
    return 'Greedy'