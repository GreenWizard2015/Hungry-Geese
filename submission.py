from datetime import datetime
import tarfile
import os
from io import BytesIO

SUBMISSION_FILE = 'submission-%s.tar.gz' % (datetime.now().strftime('%Y-%m-%d'))

INCLUDES = [
  'ConvDQNModel.py',
  'CREDQEnsemble.py',
  'weights/agent.h5',
  'Agents/CAgent.py',
  'Agents/CAgentState.py',
  'Agents/CWorldState.py',
  'Utils/TFUtils.py'
]

AGENT_CODE = """
import os
import sys

FOLDER = '/kaggle_simulations/agent/'
sys.path.append(FOLDER)
#################
from CREDQEnsemble import CREDQEnsemble
import ConvDQNModel
from Agents.CAgent import CAgent
from Agents.CAgentState import EMPTY_OBSERVATION

network = CREDQEnsemble(submodel=ConvDQNModel.createModel, NModels=3)
network.load(os.path.join(FOLDER, 'weights/agent.h5'))

MY_AGENT = CAgent(None, model=network)
def agent(obs_dict, config_dict):
  return MY_AGENT(obs_dict, config_dict)
"""

FOLDER = os.path.dirname(__file__)
print(SUBMISSION_FILE)
with tarfile.open(SUBMISSION_FILE, "w:gz") as tar:
  for file in INCLUDES:
    tar.add(os.path.join(FOLDER, file), arcname=file)
  
  fi = tarfile.TarInfo('main.py')
  fi.size = len(AGENT_CODE)
  s = BytesIO()
  s.write(AGENT_CODE.encode('utf-8'))
  s.seek(0)
  tar.addfile(fi, s)