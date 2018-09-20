
# coding: utf-8

 


import pandas as pd
import numpy as np
import pymc3 as pm
import theano.tensor as T
from theano import tensor as tt 
from pymc3 import gelman_rubin
import theano
import matplotlib.pyplot as plt
from BSSP.MyBounded import MyBounded


from copy import deepcopy
import seaborn as sns
 


# $$Ti_{t+1} = A* Ti_{t} + B * [Ta_{t+1}, Ph_{t+1}, Ps_{t+1}] + N(0, Q) $$
# 
# $$y_{t+1} = C * Ti_{t+1} + N(0, R)$$

 


data = pd.read_csv('./data/inputPRBS1.csv', delimiter = ',')


 


print(data.head())

