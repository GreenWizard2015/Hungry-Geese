import pylab as plt
import Agents
import numpy as np
from CHGEnvironment import CHGEnvironment

def collectReplays(model, agentsN, envN, agentsParams={}, envParams={}):
  environments = [CHGEnvironment({**envParams, 'agents': agentsN}) for _ in range(envN)]
  def makeAgent(index):
    if 2 <= index:
      return Agents.CGreedyAgent()
    return Agents.CAgent(*agentsParams)
  
  EA = [
    (env, [makeAgent(i) for i in range(agentsN)]) for env in environments
  ]
  allAgents = []
  for _, agents in EA: allAgents.extend(agents)
    
  replays = [[] for _ in allAgents]
  scores = [0 for _ in allAgents]
  for e in environments: e.reset()
  for a in allAgents: a.reset()
  
  def encodedStates():
    res = []
    observations = []
    for env, agents in EA:
      grid = None
      for obs, agent in zip(env.state, agents):
        observations.append(obs)
        if obs['alive']:
          if grid is None:
            grid = agent.preprocessObservations(obs['observation'], env.configuration)
            
          res.append(
            agent.processObservations(obs['observation'], env.configuration, grid.copy(), True)
          )
        else:
          res.append(
            agent.processObservations(obs['observation'], env.configuration, None, False)
          )
    return res, observations
  
  deathReasons = ['' for _ in allAgents]
  prevStates, _ = encodedStates()
  while not all(env.done for env in environments):
    predictions = model.predict(np.array(prevStates))
    actions = [agent.choiceAction(pred) for agent, pred in zip(allAgents, predictions)]
    actionsNames, actionsID = zip(*actions)
    
    for i, env in enumerate(environments):
      env.step(actionsNames[(i * agentsN):((i + 1) * agentsN)])
      
    states, observations = encodedStates()
    for i, obs in enumerate(observations):
      if obs['was alive']:
        scores[i] = obs['score']
        replays[i].append((
          prevStates[i], actionsID[i], obs['step reward'], states[i], 1 - obs['was killed']
        ))
        deathReasons[i] = obs['death reason']
    ##
    prevStates = states
  ##
  ranks = []
  kinds = []
  for env, agents in EA:
    ages = [x['age'] for x in env.state]
    byAge = list(sorted(range(len(ages)), key=lambda x: ages[x], reverse=True))
    for i, agent in enumerate(agents):
      ranks.append(1 + byAge[i])
      kinds.append(agent.kind)

  return replays, {
    'scores': scores,
    'ranks': ranks,
    'kinds': kinds,
    'death by': deathReasons,
  }
  
def trackScores(scores, metrics, levels=[.1, .3, .5, .75, .9], metricName='scores'):
  if metricName not in metrics:
    metrics[metricName] = {}
    
  def series(name):
    if name not in metrics[metricName]:
      metrics[metricName][name] = []
    return metrics[metricName][name]
  ########
  N = len(scores)
  orderedScores = list(sorted(scores, reverse=True))
  totalScores = sum(scores) / N
  series('avg.').append(totalScores)
  
  for level in levels:
    series('top %.0f%%' % (level * 100)).append(orderedScores[int(N * level)])
  return

def plotData2file(data, filename, metricName):
  plt.clf()

  fig = plt.figure()
  axe = fig.subplots()
  for name, dataset in data[metricName].items():
    axe.plot(dataset, label=name)
  axe.title.set_text(metricName)
  axe.legend(loc='upper left')

  fig.tight_layout()
  fig.savefig(filename)
  plt.close(fig)
  return

def profileAndHalt(f):
  import cProfile
  p = cProfile.Profile()
  p.enable(subcalls=True, builtins=True)
  f()
  p.disable()
  p.print_stats(sort=2)
  exit()