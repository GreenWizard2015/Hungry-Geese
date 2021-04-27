import pylab as plt
import Agents
import numpy as np
from CHGEnvironment import CHGEnvironment
from Agents.CWorldState import CWorldState, CGlobalWorldState
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation

class CDummyNetwork:
  def predict(self, X):
    return X

DummyNetwork = CDummyNetwork()

def collectExperience(agents, memory, params):
  trajectories, stats = replays = collectReplays(
    models=[x[0] for x in agents],
    agentsKinds=[x[1] for x in agents],
    envN=params['episodes'],
    envParams=params.get('env', {})
  )
  if memory: memory.store(replays)
  
  scores = stats['scores']
  ranks = stats['ranks']
  kinds = stats['kinds']
  kindsSet = set(kinds)
  
  deathBy = stats['death by']
  deathReasons = set(deathBy)
    
  for kind in kindsSet:
    print(kind, {
      reason: sum(
        1 for i, x in enumerate(deathBy) if (x==reason) and (kind==kinds[i])
      ) for reason in deathReasons
    })
  
  RLRewards = []
  Ages = []
  for traj in trajectories:
    Ages.append(len(traj))
    RLRewards.append(sum(x[2] for x in traj))

  winRates = {}
  stats = {}
  for kind in kinds:
    replaysID = [i for i, k in enumerate(kinds) if k == kind]
    stats.update({
      'Age_%s' % kind: [Ages[i] for i in replaysID],
      'Score_%s' % kind: [scores[i] for i in replaysID],
      'RLRewards_%s' % kind: [RLRewards[i] for i in replaysID],
    })
  
    winN = sum(1 for i in replaysID if ranks[i] == 1)
    winRates[kind] = winN / float(len(replaysID))
    
  print('Win rates: ', winRates)
  return stats, winRates

def collectReplays(models, agentsKinds, envN, envParams={}):
  agentsN = len(agentsKinds)
  worlds = [CWorldState() for _ in range(envN)]
  environments = [CHGEnvironment({**envParams, 'agents': agentsN}) for _ in range(envN)]
  def makeAgent(index, world):
    spawnMe = agentsKinds[index]
    return spawnMe(world)
  
  EA = [
    (env, [makeAgent(i, world) for i in range(agentsN)]) for env, world in zip(environments, worlds)
  ]
  allAgents = []
  for _, agents in EA: allAgents.extend(agents)
    
  replays = [[] for _ in allAgents]
  scores = [0 for _ in allAgents]
  isAlive = [True for _ in allAgents]
  for e in environments: e.reset()
  
  def encodedStates():
    for env, world in zip(environments, worlds):
      if not env.done:
        world.update(Observation(env.state[0]['observation']))
      
    states = []
    observations = []
    for env, agents in EA:
      for obs, agent in zip(env.state, agents):
        state = agent.processObservations(obs['observation'], env.configuration, obs['alive'])
        states.append(state)
        observations.append(obs)
    return states, observations
  
  deathReasons = ['' for _ in allAgents]
  prevStates, _ = encodedStates()
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
      
    states, observations = encodedStates()
    for i, obs in enumerate(observations):
      if obs['was alive']:
        replays[i].append((
          prevStates[i], actionsID[i], obs['step reward'], states[i], 1 - obs['was killed']
        ))
        
        scores[i] = obs['score']
        deathReasons[i] = obs['death reason']
        isAlive[i] = obs['alive']
    ##
    prevStates = states
  ##
  kinds = [agent.kind for agent in allAgents]
  ranks = []
  rankRewards = envParams['rank reward']
  for i, env in enumerate(environments):
    rnk = env.ranks()
    ranks.extend(rnk)
    for j, rank in enumerate(rnk): # RL bonus
      S, A, R, NS, D = replays[(i * agentsN) + j][-1]
      replays[(i * agentsN) + j][-1] = (S, A, R + rankRewards[rank], NS, D)

  return replays, {
    'scores': scores,
    'ranks': ranks,
    'kinds': kinds,
    'death by': deathReasons,
    'raw replays': [e.replay() for e in environments],
    'games': [
      list(zip(
        replays[(i * agentsN):((i + 1) * agentsN)],
        ranks[(i * agentsN):((i + 1) * agentsN)],
      )) for i, env in enumerate(environments)
    ],
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
  
def restoreStates(states):
  restored = []
  for s in states:
    restored.append(CGlobalWorldState(s).player(0).view())
  return np.array(restored)