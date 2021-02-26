import json
import hashlib
import os
import Utils
import random
import glob

class CHGReplaysStorage:
  def __init__(self, params):
    self._folder = params['folder']
    self._replaysPerChunk = params['replays per chunk']
    self._envParams = params['env']
    
    self._newChunk = []
    self._sampledChunk = []
    return
  
  def store(self, replay):
    self._newChunk.append(replay)
    if self._replaysPerChunk < len(self._newChunk):
      self.flush()
    return
  
  def flush(self):
    os.makedirs(self._folder, exist_ok=True)
    
    asStr = json.dumps(self._newChunk)
    filename = os.path.join(self._folder, '%s.json' % hashlib.sha224(asStr.encode()).hexdigest())
    with open(filename, 'w') as f:
      f.write(asStr)
    
    self._newChunk = []
    return
  
  def _loadSamples(self):
    if 0 < len(self._sampledChunk): return
    
    chunks = glob.glob(os.path.join(self._folder, '*.json'))
    sel = random.choice(chunks)
    with open(sel, 'r') as f:
      self._sampledChunk = json.load(f)
    return
  
  def sampleReplay(self):
    self._loadSamples()
    if 0 < len(self._sampledChunk):
      ind = random.randrange(0, len(self._sampledChunk))
      replay = self._sampledChunk[ind]
      del self._sampledChunk[ind]
      return Utils.expandReplays(replay, self._envParams)
    return None
    