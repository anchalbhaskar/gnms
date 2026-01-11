import numpy as np 
if not hasattr(np, "float_"):
    np.float_ = np.float64

import os 
notebook_dir = os.getcwd()

print('success')

import sys 
loc = os.path.abspath(os.path.join(notebook_dir,'..','..','src'))
sys.path.append(loc)

print('success')

#other basic imports 
import time 
import torch 

print('success')

from gnm import *

print('success')

from gnm import defaults, utils, evaluation, fitting, generative_rules, weight_criteria

print('success')