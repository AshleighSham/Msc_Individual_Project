# Standard Libraries
import numpy as np
import scipy.io
reference = scipy.io.loadmat('reference.mat')

inp = {}

#ranges for E and G_f
pmin = np.array([30000, 0.02])
pmax = np.array([100000, 0.03])

#build matrix how?
inp['range'] = np.array([pmin, pmax])
print(inp['range'])

#number of MCMC iterations
inp['samples'] = 1000

##inital covariance
inp['icov'] = np.array([1e7, 1e-6]).T @ np.eye(len(inp['range'][0]))

#inital guess of parameters based on prior
inp['theta0'] = np.array([60000, 0.025])

#measurement/refernce observations
inp['sigma'] = 0.05
inp['measurement'] = reference

#starting point of Kalman MCMC
inp['Kalmans'] = 50

#assumed measurement noise for Kalman
inp['me'] = 1e-1

#results = MH_mcmc(inp)