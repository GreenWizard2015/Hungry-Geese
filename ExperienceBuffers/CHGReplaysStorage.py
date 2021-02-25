import json
import hashlib
import os
class CHGReplaysStorage:
  def __init__(self, params):
    self._folder = params['folder']
    self._replaysPerChunk = params['replays per chunk']
    
    self._newChunk = []
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