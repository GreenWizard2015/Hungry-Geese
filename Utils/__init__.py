import pylab as plt
import Agents
import numpy as np
from CHGEnvironment import CHGEnvironment

class CDummyNetwork:
  def predict(self, X):
    return X

DummyNetwork = CDummyNetwork()

def collectReplays(models, agentsKinds, envN, envParams={}):
  agentsN = len(agentsKinds)
  environments = [CHGEnvironment({**envParams, 'agents': agentsN}) for _ in range(envN)]
  def makeAgent(index):
    spawnMe = agentsKinds[index]
    return spawnMe()
  
  EA = [
    (env, [makeAgent(i) for i in range(agentsN)]) for env in environments
  ]
  allAgents = []
  for _, agents in EA: allAgents.extend(agents)
    
  replays = [[] for _ in allAgents]
  scores = [0 for _ in allAgents]
  isAlive = [True for _ in allAgents]
  for e in environments: e.reset()
  for a in allAgents: a.reset()
  
  def encodedStates():
    res = []
    observations = []
    Details = []
    for env, agents in EA:
      grid = None

      for obs, agent in zip(env.state, agents):
        state = None
        details = None
        if obs['alive']:
          if grid is None:
            grid = agent.preprocessObservations(obs['observation'], env.configuration)
          
          state, details = agent.processObservations(
            obs['observation'], env.configuration, grid.copy(), True, details=True
          )
        else:
          state = agent.processObservations(obs['observation'], env.configuration, None, False)
          
        res.append(state)
        observations.append(obs)
        Details.append(details)
    return res, observations, Details
  
  deathReasons = ['' for _ in allAgents]
  prevStates, _, prevDetails = encodedStates()
  actionsNames = [None] * len(allAgents)
  actionsID = [None] * len(allAgents)
  while not all(env.done for env in environments):
    #######
    for i, model in enumerate(models):
      agentsIDs = [(j * agentsN) + i for j in range(envN) if isAlive[(j * agentsN) + i]]
      if agentsIDs:
        predictions = model.predict([prevStates[j] for j in agentsIDs])
        actions = [allAgents[j].choiceAction(pred) for j, pred in zip(agentsIDs, predictions)]
        for (actName, actID), agentID in zip(actions, agentsIDs):
          actionsNames[agentID] = actName
          actionsID[agentID] = actID
    ######
    for i, env in enumerate(environments):
      env.step(actionsNames[(i * agentsN):((i + 1) * agentsN)])
      
    states, observations, Details = encodedStates()
    for i, (obs, details) in enumerate(zip(observations, prevDetails)):
      if obs['was alive']:
        replays[i].append((
          prevStates[i], actionsID[i], obs['step reward'], states[i], 1 - obs['was killed'], details
        ))
        
        scores[i] = obs['score']
        deathReasons[i] = obs['death reason']
    ##
    prevStates = states
    prevDetails = Details
  ##
  ranks = []
  kinds = []
  for env, agents in EA:
    agentsScores = [x['score'] for x in env.state]
    ranked = np.argsort(np.argsort(-np.array(agentsScores)))
    
    for rank, agent in zip(ranked, agents):
      ranks.append(1 + rank)
      kinds.append(agent.kind)

  return replays, {
    'scores': scores,
    'ranks': ranks,
    'kinds': kinds,
    'death by': deathReasons,
    'raw replays': [e.replay() for e in environments]
  }
  
def expandReplays(replay, envParams={}):
  return collectReplays(
    [DummyNetwork, DummyNetwork, DummyNetwork, DummyNetwork],
    [Agents.CReplayAgent, Agents.CReplayAgent, Agents.CReplayAgent, Agents.CReplayAgent],
    envN=1,
    envParams={**envParams, 'replay': replay}
  )

  
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

def plotSeries2file(data, filename, title):
  plt.clf()

  fig = plt.figure()
  axe = fig.subplots()
  for name, dataset in data.items():
    axe.plot(dataset, label=name)
  axe.title.set_text(title)
  axe.legend(loc='upper left')

  fig.tight_layout()
  fig.savefig(filename)
  plt.close(fig)
  return

def plotData2file(data, filename, metricName):
  plotSeries2file(data[metricName], filename, metricName)
  return

def profileAndHalt(f):
  import cProfile
  p = cProfile.Profile()
  p.enable(subcalls=True, builtins=True)
  f()
  p.disable()
  p.print_stats(sort=2)
  exit()