from kaggle_environments.envs.hungry_geese.hungry_geese import Action, row_col
import random

class CGoose:
  def __init__(self, index, head, configs):
    self.index = index
    self.alive = True
    self.body = [head]
    self._prevAction = None
    self._score = 0
    self._configs = configs
    self._deathReason = ''
    self.reset()
    return
  
  def reset(self):
    self._stepReward = 0
    self._wasAlive = self.alive
    return
  
  def killOpponent(self):
    self._stepReward += self._configs.get('kill reward', 0)
    return
  
  def die(self, reason):
    if self.alive:
      self._stepReward += self._configs.get('death reward', -1)
      self._deathReason = reason
    self.alive = False
    return
    
  def checkMove(self, action, newPos):
    if (self._prevAction is None) or not (action == self._prevAction.opposite()):
      if not (newPos in self.body): # self collision 
        self._prevAction = action
        return True
    
    self.die('Illegal move (self collision).')
    print('Illegal move (self collision).')
    return False
  
  def moveTo(self, head, grow=False):
    self.body.insert(0, head)
    if not grow:
      self._stepReward += self._configs.get('move reward', 0)
      return self.body.pop()
    
    self._stepReward += self._configs.get('grow reward', 1)
    return
  
  def starve(self):
    self._stepReward += self._configs.get('starve reward', 0)
    if len(self.body) <= 1:
      self.alive = False# self.die('Starved.')
    return self.body.pop()

  def state(self, food, geese, step):
    return {
      'action': str(self._prevAction),
      'reward': self._score,
      'step reward': self._stepReward,
      'info': {}, # dummy
      'observation': {
        'remainingOverageTime': 60, # dummy
        'step': step,
        'geese': geese,
        'food': food,
        'index': self.index
      },
      'status': 'ACTIVE' if self.alive else 'DONE',
      'death reason': self._deathReason,
      'was alive': self._wasAlive,
      'alive': self.alive
    }
  
  def score(self, v):
    self._score = v
    return v
     
class CHGEnvironment:
  def __init__(self, params):
    self._params = params
    self.agents = self._agents = params.get('agents', 4)
    self._rows = params.get('rows', 7)
    self._columns = params.get('columns', 11)
    self._hungerRate = params.get('hunger rate', 40)
    self._minFood = params.get('min food', 2)
    self._minPlayers = params.get('min players', 1)
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
    self._random = random.Random(self._seed)
    self._step = 0
    self._freeCells = set(range(self._columns * self._rows))
    
    self._food = set()
    self._spawnFood()
    
    self._geese = [CGoose(i, self._sampleCell(), self._params) for i in range(self._agents)]
    return self.state
  
  def step(self, actions):
    self._step += 1
    
    for action, goose in zip(actions, self._geese):
      goose.reset()
      if goose.alive:
        self._perform(goose, Action[action])
    ######
    self._spawnFood()
    
    if self.done:
      for goose in self._geese:
        if goose.alive:
          # Boost the survivor's reward to maximum
          goose.score(2 * self._episodeSteps + len(goose.body))
    
    return self.state
  
  @property
  def state(self):
    food = list(self._food)
    geese = []
    for goose in self._geese:
      geese.append(goose.body if goose.alive else [])
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
    cell = self._random.choice(tuple(self._freeCells)) 
    self._freeCells.remove(cell)
    return cell

  def _translate(self, position: int, direction: Action) -> int:
    rows, columns = self._rows, self._columns
    row, column = row_col(position, columns)
    row_offset, column_offset = direction.to_row_col()
    row = (row + row_offset) % rows
    column = (column + column_offset) % columns
    return row * columns + column
  
  def _perform(self, goose, action):
    goose.score(self._step + len(goose.body))  # standard score
    
    newHead = self._translate(goose.body[0], action)
    if goose.checkMove(action, newHead):
      if newHead not in self._freeCells: # collide
        if newHead in self._food: # with food
          #self._freeCells.remove(newHead)
          goose.moveTo(newHead, grow=True)
        else: # with goose?
          for g in self._geese:
            if newHead in g.body:
              g.killOpponent()
              goose.die('Collide with goose (%d %d)' % (goose.index, g.index))
      else: # just move
        self._freeCells.remove(newHead)
        self._freeCells.add(goose.moveTo(newHead))
    ####
    # If hunger strikes remove from the tail.
    if (self._step % self._hungerRate == 0) and goose.alive:
      self._freeCells.add(goose.starve())
      
    if not goose.alive: # remove body if die
      for p in goose.body:
        self._freeCells.add(p)
    return