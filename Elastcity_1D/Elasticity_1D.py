# Standard Libraries
import numpy as np
from MH import MH_mcmc
from EnKF import EnKF_mcmc
from DRAM import DRAM_algorithm
import utilities
import scipy.io
reference = scipy.io.loadmat(r"C:\Users\ashle\Documents\GitHub\Portfolio\ES98C\Elastcity_1D\reference.mat")
#results = scipy.io.loadmat(r"C:\Users\ashle\Documents\GitHub\Portfolio\ES98C\Elastcity_1D\results.mat")

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

option = 1

if option == 0:
    # The Metropolis-Hastings technique
    C = MH_mcmc(inp)
    results = C.MH_go()
    print(results['MCMC'])
    print('The median of the posterior is: %f' % np.median(results['MCMC'][0,100:]))

elif option == 1:
    # The DRAM algorithm 
    A = DRAM_algorithm(inp)
    results = A.DRAM_go()
    print(results['MCMC'])
    print("The median of the posterior is: %f" % np.median(results['MCMC'][0,100:]))

elif option == 2:
    #The EnKF algorithm 
    B = EnKF_mcmc(inp)
    results = B.EnKF_go()
    print(results['MCMC'])
    print("The median of the posterior is: %f" % np.median(results['MCMC'][0,50:]))