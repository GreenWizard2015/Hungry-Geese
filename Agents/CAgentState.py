from Agents.CWorldState import EMPTY_PLAYER_OBSERVATION, EMPTY_RAW_OBSERVATION
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation

EMPTY_OBSERVATION = EMPTY_PLAYER_OBSERVATION

class CAgentState:
  def __init__(self, world):
    self._world = world
    return
  
  def local(self, observation, configuration):
    observation = observation if isinstance(observation, Observation) else Observation(observation)
    player = self._world.player(observation.index)
    
    actionsMask, actionsMapping = player.validMoves()
    return player.raw, actionsMask, actionsMapping
  
  @property
  def EmptyObservations(self):
    return EMPTY_RAW_OBSERVATION