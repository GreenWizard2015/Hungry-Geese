from CAgent import CAgent
import numpy as np

def collectReplays(env, model):
  agents = [CAgent() for _ in range(env.agents)]
  replays = [[] for _ in agents]
  for a in agents: a.reset()
  
  def encodedState(state):
    res = []
    for obs, agent in zip(state, agents):
      res.append(agent.processObservations(obs['observation'], env.configuration, obs['alive']))
    return res
  
  prevStates = encodedState(env.reset())
  while not env.done:
    predictions = model.predict(np.array(prevStates))
    actions = [agent.choiceAction(pred) for agent, pred in zip(agents, predictions)]
    actionsNames, actionsID = zip(*actions)
    
    observations = env.step(actionsNames)
    states = encodedState(observations)
    for i, obs in enumerate(observations):
      if obs['was alive']:
        replays[i].append((
          prevStates[i], obs['step reward'], actionsID[i], states[i], obs['alive']
        ))
    ##
    prevStates = states
  ##
  return replays
  