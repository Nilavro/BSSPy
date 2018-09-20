from pymc3 import Normal, Flat, Continuous
import theano.tensor as T


class StateSpaceModel(Continuous):
	"""
	A state space model with Gaussian noise.
	
	This models only the state variables so that the form of the observation
	noise can be specified separately.
	
	Parameters
	----------
	tau : tensor
		tau > 0, innovation precision
	sd : tensor 1e-5
		sd > 0, innovation standard deviation (alternative to specifying tau)
	A: tensor
		state update matrix
	B : tensor
		input matrix
	u : tensor
		(time x dim), inputs to the system
		init : distribution
		distribution for initial value (defaults to Flat())
	"""
	def __init__(self, tau=None, sd=None, A=None, B=None,  u=None, x0 = None,   init=Flat.dist(), *args, **kwargs):
		super(StateSpaceModel, self).__init__(*args, **kwargs)
		self.tau = tau
		self.sd = sd
		self.A = A
		self.B = B
		self.u = u
		self.init = init
		self.mean = x0 
		
	def logp(self, x):
		tau = self.tau
		sd = self.sd
		A = self.A
		B = self.B
		u = self.u
		init = self.init

		# x[0,:] = init
		x_im1 = x[:-1]
		x_i = x[1:]
		u_im1 = u[:-1] 
		innov_like = Normal.dist(mu=T.dot(A, x_im1.T)+ T.dot(B, u_im1.T), tau=tau, sd=sd).logp(x_i.T)
		return T.sum(init.logp(x[0])) + T.sum(innov_like)