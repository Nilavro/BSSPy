
# coding: utf-8

 


import pandas as pd
import numpy as np
import pymc3 as pm

from theano import tensor as tt 
from pymc3 import gelman_rubin
import theano
import matplotlib.pyplot as plt
from BSSP.StateSpaceModel import StateSpaceModel 
from BSSP.MyBounded import MyBounded
from BSSP.MyBound import MyBound

from copy import deepcopy
import seaborn as sns
 


# $$Ti_{t+1} = A* Ti_{t} + B * [Ta_{t+1}, Ph_{t+1}, Ps_{t+1}] + N(0, Q) $$
# 
# $$y_{t+1} = C * Ti_{t+1} + N(0, R)$$

 


data = pd.read_csv('./data/inputPRBS1.csv', delimiter = ',')


 


print(data.head())


 


# data = data.groupby(np.arange(len(data))).mean()


y = data['yTi'].values 
 
u = data[['Ta', 'Ps', 'Ph']].values

print(y.shape )
print(u.shape)

 




dT = 1
with pm.Model() as model:
	
	boundedGamma = MyBound(pm.Gamma, lower= 0, upper=100 )

	Ri  = boundedGamma('Ri', alpha = 10e-5, beta = 10e-5, transform='interval')
	Ci =  boundedGamma('Ci', alpha = 10e-5, beta = 10e-5, transform='interval')
	Aw =  boundedGamma('Aw', alpha = 10e-5, beta = 10e-5, transform='interval') 
	
	A = tt.as_tensor([1 - dT * tt.inv(tt.mul(Ri,Ci))]) 

	
	B = tt.as_tensor([tt.inv(tt.mul(Ri,Ci))*dT,  tt.mul(Aw,tt.inv(Ci))*dT,  tt.inv(Ci) *dT]) 
	
	x_init = pm.Normal('x_init', mu = 20, tau = .2)
	Tau = pm.Gamma('tau', alpha=1e-5, beta=1e-5) 
	X = StateSpaceModel('x', A=A, B=B, u=u,  tau = Tau, x0 = x_init,  shape=(y.shape[0],1))
	
	C =   tt.eye(1)
	Tau_o = pm.Gamma('tau_o', alpha=1e-5, beta=1e-5)
	
	Y = pm.Normal('y', mu=tt.dot(C, X.T).T, tau=Tau_o, observed=y.reshape(-1,1) ) 


 

with model:
	inference = pm.ADVI()
	tracker = pm.callbacks.Tracker(
	mean= inference.approx.mean.eval,  # callable that returns mean
	std= inference.approx.std.eval  # callable that returns std
	)
	approx = pm.fit(n= 160000, method=inference, callbacks=[tracker])

 

traceTi = approx.sample(5000)   
 
 
 


print ( "R  \mu : {} \sigma : {}".format(traceTi['Ri'].mean(axis = 0)  ,np.std(traceTi['Ri'], axis = 0) ))

print ( "C  \mu : {} \sigma : {}".format(traceTi['Ci'].mean(axis = 0)  ,np.std(traceTi['Ci'], axis = 0) ))
 
print ( "Aw  \mu : {} \sigma : {}".format(traceTi['Aw'].mean(axis = 0)  ,np.std(traceTi['Aw'], axis = 0) ))

 