from kaggle_environments.envs.hungry_geese.hungry_geese import Action, row_col
import numpy as np
import random

class CGoose:
  def __init__(self, index, head, configs):
    self.index = index
    self.alive = True
    self.body = [head]
    self._prevAction = None
    self._nextAction = None
    self._score = 0
    self._configs = configs
    self._deathReason = ''
    self._age = 0
    self.nextStep()
    return
  
  def nextStep(self):
    if self.alive:
      self._age += 1
      
    self._stepReward = 0
    self._wasAlive = self.alive
    self._wasKilled = False
    return

  def die(self, reason, killed=False):
    if self.alive:
      self._deathReason = reason
      if killed:
        self._stepReward += self._configs.get('death reward', -1)
      self._wasKilled = killed

    self.alive = False
    self.body = []
    return
    
  def checkMove(self, action, newPos):
    if not (self._prevAction is None): # not first move
      if action == self._prevAction.opposite():
        self.die('Illegal move.', killed=True)
        print('Illegal move.')
        return False
    
    if newPos in self.body[:-1]: # self collision
      self.die('Self collision.', killed=True)
      return False
    
    self._prevAction = action
    return True
  
  def moveTo(self, head, grow=False):
    self.body.insert(0, head)
    if not grow:
      self._stepReward += self._configs.get('move reward', lambda _: 0)(self.body)
      return self.body.pop()
    
    self._stepReward += self._configs.get('grow reward', lambda _: 1)(self.body)
    return
  
  def starve(self):
    self.body.pop()
    
    if len(self.body) < 1:
      self._stepReward += self._configs.get('starve reward', 0)
      self.die('Starved.')
    return
  
  def killOpponent(self):
    if self.alive:
      self._stepReward += self._configs.get('kill reward', 0)
    return
  
  def opponentDeath(self):
    if self.alive:
      self._stepReward += self._configs.get('opponent death reward', 0)
    return
  
  def killed(self):
    if self._wasKilled and self._wasAlive:
      self._stepReward += self._configs.get('killed reward', 0)
      self._deathReason = 'Killed by enemy' 
    return

  def survived(self):
    if self.alive:
      self._stepReward += self._configs.get('survived reward', 0)
      self._deathReason = 'Alive'
    return

  def state(self, food, geese, step):
    return {
      'age': self._age,
      'action': self._prevAction,
      'reward': self._score,
      'score': self._score,
      'step reward': self._stepReward,
      'info': {}, # dummy
      'observation': {
        'remainingOverageTime': 60, # dummy
        'step': step,
        'geese': geese,
        'food': food,
        'index': self.index,
        'next action': self._nextAction,
      },
      'status': 'ACTIVE' if self.alive else 'DONE',
      'death reason': self._deathReason,
      'was alive': self._wasAlive,
      'alive': self.alive,
      'was killed': self._wasKilled,
    }
  
  def score(self, v=None):
    self._score = self._score if v is None else v
    return self._score
  
  def nextAction(self, action):
    self._nextAction = action
    return
  
  def rank(self, rank, survived=False):
    if self._wasAlive and (survived or not self.alive):
      self._stepReward += self._configs['rank reward'][rank]
    return
   
class CHGEnvironment:
  def __init__(self, params=None):
    replay = params.get('replay', None)
    self.isReplay = not (replay is None)
    if self.isReplay:
      self._replay = replay
      self._replayInd = 0
      
    self._params = params
    self.agents = self._agents = params.get('agents', 4)
    self._rows = params.get('rows', 7)
    self._columns = params.get('columns', 11)
    self._hungerRate = params.get('hunger rate', 40)
    self._minFood = params.get('min food', 2)
    self._minPlayers = params.get('min players', 2)
    self._seed = params.get('seed', None)
    self._episodeSteps = params.get('episode steps', 200)
    
    self.configuration = {
      "columns": self._columns,
      "rows": self._rows,
      "hunger_rate": self._hungerRate,
      "min_food": self._minFood
    }
    return
  
  def reset(self):
    if self.isReplay:
      self._replayInd = 0
    else:
      self._replay = []

    self._random = random.Random(self._seed)
    self._step = 0
    self._food = set()
    self._geese = []
    
    self._spawnFood()
    self._geese = [CGoose(i, self._sampleCell(), self._params) for i in range(self._agents)]
    return self.state
  
  def _save(self, data):
    if not self.isReplay:
      self._replay.append(data)
    return
  
  def _next(self, kind, moveNext=True):
    k, res = self._replay[self._replayInd]
    assert k == kind
    if moveNext:
      self._replayInd += 1
    return res

  def step(self, actions=None):
    for goose in self._geese:
      goose.nextStep()

    if self.done: return actions
    
    if self.isReplay:
      actions = self._next('step')
    else:
      self._save(('step', [str(x if goose.alive else None) for goose, x in zip(self._geese, actions)]))
    
    self._step += 1
    
    hungerStrike = (self._step % self._hungerRate == 0)
    for goose in self._geese:
      # If hunger strikes remove from the tail.
      if hungerStrike and goose.alive:
        goose.starve()
      
    for action, goose in zip(actions, self._geese):
      self._perform(goose, action)
    ######
    # check colliding NEW POSITION of heads
    self._resolveCollisions()
    ########
    self._spawnFood()
    ######
    done = self.done
    if done:
      for goose in self._geese:
        if goose.alive:
          # Boost the survivor's reward to maximum
          goose.score(2 * self._episodeSteps + len(goose.body))
          goose.survived()
    
    for goose, rank in zip(self._geese, self._ranks()):
      goose.rank(1 + rank, survived=done)
    return actions
  
  def _gooseCollide(self, ind):
    goose = self._geese[ind]
    if not goose.alive:
      return None
    
    head = goose.body[0]
    for i, enemy in enumerate(self._geese):
      if enemy.alive and not(i == ind):
        if head in enemy.body:
          return enemy
    return None
  
  def _ranks(self):
    scores = [x.score() for x in self._geese]
    return np.argsort(np.argsort(-np.array(scores)))
    
  def _resolveCollisions(self):
    collideWith = [self._gooseCollide(ind) for ind, _ in enumerate(self._geese)]
    for i, killer in enumerate(collideWith):
      if not (killer is None):
        self._geese[i].die('Collide with goose', killed=True)
        
    for killer in collideWith:
      if not (killer is None):
        if killer.alive:
          killer.killOpponent()

    for i, killer in enumerate(collideWith):
      if not (killer is None):
        if killer.alive:
          self._geese[i].killed()
    
    if any(not(x is None) for x in collideWith):
      for goose in self._geese:
        goose.opponentDeath()
    return
  
  @property
  def state(self):
    food = list(self._food)
    geese = []
    for goose in self._geese:
      geese.append(goose.body if goose.alive else [])
      
    if self.isReplay and (self._replayInd < len(self._replay)):
      for goose, act in zip(self._geese, self._next('step', moveNext=False)):
        goose.nextAction(act)
    return [goose.state(food, geese, self._step) for goose in self._geese]
  
  @property
  def done(self):
    alive = sum(1 for goose in self._geese if goose.alive)
    if alive < self._minPlayers: return True
    if self._episodeSteps < self._step: return True
    return False
  
  def _spawnFood(self):
    needed_food = self._minFood - len(self._food)
    for _ in range(needed_food):
      self._food.add(self._sampleCell())
    return needed_food

  def _sampleCell(self):
    if self.isReplay:
      return self._next('cell')
    
    freeCells = list(range(self._columns * self._rows))
    for x in self._food:
      freeCells.remove(x)

    for goose in self._geese:
      for x in goose.body:
        if x in freeCells: freeCells.remove(x)

    res = self._random.choice(freeCells)
    self._save(('cell', res))
    return res

  def _translate(self, position: int, direction: Action) -> int:
    rows, columns = self._rows, self._columns
    row, column = row_col(position, columns)
    row_offset, column_offset = direction.to_row_col()
    row = (row + row_offset) % rows
    column = (column + column_offset) % columns
    return row * columns + column
  
  def _perform(self, goose, action):
    if not goose.alive: return
    goose.score(self._step + len(goose.body))  # standard score
    
    action = Action[action]
    newHead = self._translate(goose.body[0], action)
    if not goose.checkMove(action, newHead): return
    
    goose.moveTo(newHead, grow=self._eatFood(newHead))
    return
  
  def _eatFood(self, pos):
    if pos in self._food:
      self._food.remove(pos)
      return True
    return False
  
  def replay(self):
    return self._replay