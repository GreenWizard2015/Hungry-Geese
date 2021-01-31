import pylab as plt
from CAgent import CAgent
import numpy as np
from CHGEnvironment import CHGEnvironment

def collectReplays(model, agentsN, envN, agentsParams={}, envParams={}):
  environments = [CHGEnvironment({**envParams, 'agents': agentsN}) for _ in range(envN)]
  EA = [
    (env, [CAgent(*agentsParams) for _ in range(agentsN)]) for env in environments
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
        scores[i] += obs['step reward']
        replays[i].append((
          prevStates[i], actionsID[i], obs['step reward'], states[i], obs['alive']
        ))
    ##
    prevStates = states
  ##
  return replays, scores
  
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

def plotData2file(data, filename, maxCols=3):
  plt.clf()
  N = len(data)
  rows = (N + maxCols - 1) // maxCols
  cols = min((N, maxCols))
  
  figSize = plt.rcParams['figure.figsize']
  fig = plt.figure(figsize=(figSize[0] * cols, figSize[1] * rows))
  
  axes = fig.subplots(ncols=cols, nrows=rows)
  axes = axes.reshape((-1,)) if 1 < len(data) else [axes]
  for (chartname, series), axe in zip(data.items(), axes):
    for name, dataset in series.items():
      axe.plot(dataset, label=name)
    axe.title.set_text(chartname)
    axe.legend()
    
  fig.savefig(filename)
  plt.close(fig)
  return