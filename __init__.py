# system
import os 
import gc
from contextlib import contextmanager
from IPython.display import clear_output
from typing import Dict, List, Optional, Tuple

# data process
import time
import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# causis api
from causis_api.const import get_version
from causis_api.const import login
login.username = 'shuai.song'
login.password = 'Tsinghua2022'
login.version = get_version()
from causis_api.data import *

# models
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, KFold
import lightgbm as lgb

