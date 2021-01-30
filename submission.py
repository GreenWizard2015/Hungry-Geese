from CAgent import CAgent

GLOBAL_AGENT = CAgent(model=...)

def agent(obs_dict, config_dict):
  global GLOBAL_AGENT
  return GLOBAL_AGENT.play(obs_dict, config_dict)