import pymc3 as pm
from pymc3  import ADVI 
from pymc3.distributions import transforms
from pymc3.distributions.continuous import Continuous
from pymc3.distributions.dist_math import bound
import numpy as np
from BSSP.MyBounded import MyBounded

class MyBound(object):
    """Creates a new bounded distribution"""
    def __init__(self, distribution, lower=-np.inf, upper=np.inf):
        self.distribution = distribution
        self.lower = lower
        self.upper = upper
        
    def __call__(self, *args, **kwargs):
        first, args = args[0], args[1:]

        return MyBounded(first, self.distribution, self.lower, self.upper,
                       *args, **kwargs)

    def dist(self, *args, **kwargs):
        return MyBounded.dist(self.distribution, self.lower, self.upper,
                            *args, **kwargs)
