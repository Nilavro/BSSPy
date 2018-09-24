
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
 

  
 


data = pd.read_csv('./data/inputPRBS1.csv', delimiter = ',')


 


print(data.head())


 


# data = data.groupby(np.arange(len(data))).mean()


y = data['yTi'].values 
 
u = data[['Ta', 'Ps', 'Ph']].values

print(y.shape )
print(u.shape)

 

N_states = 3

# Ti Te Th
dT = 1
with pm.Model() as model:
    
    boundedGamma = MyBound(pm.Gamma, lower= 0, upper=100)    
    Rie = boundedGamma('Rie', alpha = 10e-5, beta = 10e-5, transform='interval')
    Rea = boundedGamma('Rea', alpha = 10e-5, beta = 10e-5, transform='interval')      
    Rih = boundedGamma('Rih', alpha = 10e-5, beta = 10e-5, transform='interval')
    Ci =  boundedGamma('Ci', alpha = 10e-5, beta = 10e-5, transform='interval')
    Ch =  boundedGamma('Ch', alpha = 10e-5, beta = 10e-5, transform='interval')
    Ce =  boundedGamma('Ce', alpha = 10e-5, beta = 10e-5, transform='interval')
    Aw =  boundedGamma('Aw', alpha = 10e-5, beta = 10e-5, transform='interval')
    Ae =  boundedGamma('Ae', alpha = 10e-5, beta = 10e-5, transform='interval')
    
    flat_A = tt.as_tensor([1 -dT* tt.inv(tt.mul(Rie,Ci)) -dT* tt.inv(tt.mul(Rih,Ci)), dT*  tt.inv(tt.mul(Rie,Ci)),   tt.inv(tt.mul(Rih,Ci)), 
                          dT* tt.inv(tt.mul(Rie,Ce)) ,  1 - dT*tt.inv(tt.mul(Rie,Ce)) -dT* tt.inv(tt.mul(Rea,Ce)), 0 , 
                          dT * tt.inv(tt.mul(Rih,Ch)) , 0 , 1 - dT* tt.inv(tt.mul(Rih,Ch))])
    A = flat_A.reshape((N_states, N_states))  

   
    flat_B = tt.as_tensor([0, dT* tt.mul(Aw,tt.inv(Ci)), 0,
                           dT* tt.inv(tt.mul(Rea,Ce)), dT*tt.mul(Ae,tt.inv(Ci)), 0 , 
                           0 , 0 , tt.inv(Ch)])
    
    B = flat_B.reshape((N_states, 3)) 
    
    Tau_init1 = pm.Gamma('Tau_init1', alpha=1e-5, beta=1e-5) 
    Tau_init2 = pm.Gamma('Tau_init2', alpha=1e-5, beta=1e-5) 
    Tau_init3 = pm.Gamma('Tau_init3', alpha=1e-5, beta=1e-5) 

    x_i_init = pm.Normal('x_i_init', mu = 15, tau = Tau_init1)
    x_e_init = pm.Normal('x_e_init', mu = 5, tau = Tau_init2)
    x_h_init = pm.Normal('x_h_init', mu = 20, tau = Tau_init3)
    x_init = tt.as_tensor([x_i_init, x_e_init, x_h_init])
    
    Tau = pm.Gamma('Tau', alpha=1e-5, beta=1e-5)
    
     
     

    
    X = StateSpaceModel('x', A=A, B=B, u=u,   tau=Tau, x0 = x_init,  shape=(y.shape[0], N_states))
    
    C =   tt.as_tensor([1, 0, 0])
    
    
  
 
    Tau_o = pm.Gamma('tau_o', alpha=1e-5, beta=1e-5)
    
    Y = pm.Normal('y', mu=T.dot(C, X.T).T, tau=Tau_o, observed = y  )



 

with model:
	inference = pm.ADVI()
	tracker = pm.callbacks.Tracker(
	mean= inference.approx.mean.eval,  # callable that returns mean
	std= inference.approx.std.eval  # callable that returns std
	)
	approx = pm.fit(n= 160000, method=inference, callbacks=[tracker])

 

traceTi = approx.sample(5000)   
 
 
 


print ( "Rie  \mu : {} \sigma : {}".format(traceTi['Rie'].mean(axis = 0)  ,np.std(traceTi['Rie'], axis = 0) ))
print ( "Rih  \mu : {} \sigma : {}".format(traceTi['Rih'].mean(axis = 0)  ,np.std(traceTi['Rih'], axis = 0) ))
print ( "Rea  \mu : {} \sigma : {}".format(traceTi['Rea'].mean(axis = 0)  ,np.std(traceTi['Rea'], axis = 0) ))
print ( "Ci  \mu : {} \sigma : {}".format(traceTi['Ci'].mean(axis = 0)  ,np.std(traceTi['Ci'], axis = 0) ))
print ( "Ce  \mu : {} \sigma : {}".format(traceTi['Ce'].mean(axis = 0)  ,np.std(traceTi['Ce'], axis = 0) ))
print ( "Ch  \mu : {} \sigma : {}".format(traceTi['Ch'].mean(axis = 0)  ,np.std(traceTi['Ch'], axis = 0) ))
print ( "Aw  \mu : {} \sigma : {}".format(traceTi['Aw'].mean(axis = 0)  ,np.std(traceTi['Aw'], axis = 0) ))
print ( "Ae  \mu : {} \sigma : {}".format(traceTi['Ae'].mean(axis = 0)  ,np.std(traceTi['Ae'], axis = 0) ))

 
