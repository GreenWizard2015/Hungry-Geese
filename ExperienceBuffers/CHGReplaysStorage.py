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
    self._disabled = params.get('disabled', False)
    
    self._newChunk = []
    self._sampledChunk = []
    return
  
  def store(self, replay):
    if self._disabled: return
    self._newChunk.append(replay)
    if self._replaysPerChunk < len(self._newChunk):
      self.flush()
    return
  
  def flush(self):
    if self._disabled: return
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
    if not chunks: return
    sel = random.choice(chunks)
    with open(sel, 'r') as f:
      self._sampledChunk = json.load(f)
    return
  
  def sampleReplay(self):
    if self._disabled: return None
    
    self._loadSamples()
    if 0 < len(self._sampledChunk):
      ind = random.randrange(0, len(self._sampledChunk))
      replay = self._sampledChunk[ind]
      del self._sampledChunk[ind]
      return Utils.expandReplays(replay, self._envParams)
    return None
    