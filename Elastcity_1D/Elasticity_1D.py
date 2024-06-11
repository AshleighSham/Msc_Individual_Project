# Standard Libraries
import numpy as np
#from MH import MH_mcmc
from EnKF import EnKF_mcmc
from DRAM import DRAM_algorithm
import utilities
import scipy.io
reference = scipy.io.loadmat(r"C:\Users\ashle\Documents\GitHub\Portfolio\ES98C\Elastcity_1D\reference.mat")
results = scipy.io.loadmat(r"C:\Users\ashle\Documents\GitHub\Portfolio\ES98C\Elastcity_1D\results.mat")

inp = {}
                       

# range of the parameters based on the prior density
inp['range']=np.array([[1], [20]])                              

# number of iteration in MCMC
inp['nsamples']=1000 

# initial covariance                          
inp['icov']=np.array([1])                                   

# initial guesss of the parameters based on the prior
inp['theta0']=np.array([3])  

# std                               
inp['sigma']=0.1

# measurement/ reference observation                                
inp['measurement']=reference['measurement']
# The starting point of the Kalman MCMC           
inp['Kalmans']=100                              

# assumed measurement error for Kalman MCMC
inp['me']=1e-1                                 

 
# The Metropolis-Hastings technique
#[results] = MH_mcmc(inp)
#print('The median of the posterior is:%d\n', np.median((results.MCMC)))

# The DRAM algorithm 
A = DRAM_algorithm(inp)
results = A.DRAM_go()
print("The median of the posterior is: %f" % np.median(results['MCMC'].T))

# The EnKF algorithm 
#B = EnKF_mcmc(inp)
#results = B.EnKF_go()
#print("The median of the posterior is:%d\n" np.median((results['MCMC'])))