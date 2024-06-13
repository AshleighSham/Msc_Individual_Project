# Standard Libraries
import numpy as np
from MH import MH_mcmc
from EnKF import EnKF_mcmc
from DRAM import DRAM_algorithm
from AMH import AMH_mcmc
from MH_DR import MH_DR_mcmc
import scipy.io
import utilities
reference = scipy.io.loadmat(r"C:\Users\ashle\Documents\GitHub\Portfolio\ES98C\Elastcity_1D\reference.mat")
#results = scipy.io.loadmat(r"C:\Users\ashle\Documents\GitHub\Portfolio\ES98C\Elastcity_1D\results.mat")

inp = {}
                       
# range of the parameters based on the prior density
inp['range']=np.array([[5, 0.01], [40, 0.5]])                          

# number of iteration in MCMC
inp['nsamples']=500

# initial covariance                          
inp['icov']=np.array([[5, 0],[0, 0.005]])                                   

# initial guesss of the parameters based on the prior
inp['theta0']=np.array([[10],[0.2]])  

# std                               
inp['sigma']=0.1

# measurement/ reference observation                                
inp['measurement']=np.array([[ 0.        ],
 [ 0.        ],
 [ 0.66394251],
 [ 0.03942506],
 [ 1.33100886],
 [ 0.03123844],
 [ 0.        ],
 [ 0.        ],
 [ 0.66394251],
 [-0.03942506],
 [ 1.33100886],
 [-0.03123844]])

# The starting point of the Kalman MCMC           
inp['Kalmans']=100                          

# assumed measurement error for Kalman MCMC
inp['me']=1e-1                                 

option = 4

print()
print()
if option == 0:
    # The Metropolis-Hastings technique
    C = MH_mcmc(inp)
    results = C.MH_go()
    print('----------------------------------------------')
    print('Metropolis Hastings')
    print('----------------------------------------------')
    print('Acceptance Rate: %f' % results['accepted'])
    print('The median of the Youngs Modulus posterior is: %f' % np.median(results['MCMC'][0]))
    print('The median of the Poissons Ratio posterior is: %f' % np.median(results['MCMC'][1]))
    print()

elif option == 1:
    # The AMH algorithm 
    A = AMH_mcmc(inp)
    results = A.AMH_go()
    #print(results['MCMC'])
    print('----------------------------------------------')
    print('Adaptive Metropolis Hastings')
    print('----------------------------------------------')
    print('Acceptance Rate: %f' % results['accepted'])
    print('The median of the Youngs Modulus posterior is: %f' % np.median(results['MCMC'][0]))
    print('The median of the Poissons Ratio posterior is: %f' % np.median(results['MCMC'][1]))
    print()

elif option == 2:
    # The AMH algorithm 
    A = MH_DR_mcmc(inp)
    results = A.MH_DR_go()
    #print(results['MCMC'])
    print('----------------------------------------------')
    print('Metropolis Hastings Delayed Rejection')
    print('----------------------------------------------')
    print('Acceptance Rate: %f' % results['accepted'])
    print('The median of the Youngs Modulus posterior is: %f' % np.median(results['MCMC'][0]))
    print('The median of the Poissons Ratio posterior is: %f' % np.median(results['MCMC'][1]))
    print()

elif option == 3:
    # The DRAM algorithm 
    A = DRAM_algorithm(inp)
    results = A.DRAM_go()
    #print(results['MCMC'])
    print('----------------------------------------------')
    print('Delayed Rejection Adaptive Metropolis Hastings')
    print('----------------------------------------------')
    print('Acceptance Rate: %f' % results['accepted'])
    print('The median of the Youngs Modulus posterior is: %f' % np.median(results['MCMC'][0]))
    print('The median of the Poissons Ratio posterior is: %f' % np.median(results['MCMC'][1]))
    print()

elif option == 4:
    #The EnKF algorithm 
    B = EnKF_mcmc(inp)
    results = B.EnKF_go()
    #print(results['MCMC'])
    print('----------------------------------------------')
    print('Ensemble Kalman Filter')
    print('----------------------------------------------')
    print('Acceptance Rate: %f' % results['accepted'])
    print('The median of the Youngs Modulus posterior is: %f' % np.median(results['MCMC'][0]))
    print('The median of the Poissons Ratio posterior is: %f' % np.median(results['MCMC'][1]))
    print()