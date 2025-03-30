# Standard Libraries
import numpy as np
from MH import MH_mcmc
from EnKF import EnKF_mcmc
from DRAM import DRAM_algorithm
from AMH import AMH_mcmc
from MH_DR import MH_DR_mcmc
import scipy.io
reference = scipy.io.loadmat(r"C:\Users\ashle\Documents\GitHub\Msc_Individual_Project\Elastcity_1D\reference.mat")

inp = {}

# range of the parameters based on the prior density
inp['range'] = np.array([[1], [20]])

# number of iteration in MCMC
inp['nsamples'] = 5000

# initial covariance
inp['icov'] = np.array([[1]])

# initial guesss of the parameters based on the prior
inp['theta0'] = np.array([3])

# standard deviation
inp['sigma'] = 0.1

# measurement/ reference observation
inp['measurement'] = reference['measurement']

# The starting point of the Kalman MCMC
inp['Kalmans'] = 200

# assumed measurement error for Kalman MCMC
inp['me'] = 1e-1

option = 2

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
    print('The median of the posterior is: %f' % np.median(results['MCMC']))
    print()

elif option == 1:
    # The AMH algorithm
    A = AMH_mcmc(inp)
    results = A.AMH_go()

    print('----------------------------------------------')
    print('Adaptive Metropolis Hastings')
    print('----------------------------------------------')
    print('Acceptance Rate: %f' % results['accepted'])
    print("The median of the posterior is: %f" % np.median(results['MCMC']))
    print()

elif option == 2:
    # The AMH algorithm
    A = MH_DR_mcmc(inp)
    results = A.MH_DR_go()

    print('----------------------------------------------')
    print('Metropolis Hastings Delayed Rejection')
    print('----------------------------------------------')
    print('Acceptance Rate: %f' % results['accepted'])
    print("The median of the posterior is: %f" % np.median(results['MCMC']))
    print()

elif option == 3:
    # The DRAM algorithm
    A = DRAM_algorithm(inp)
    results = A.DRAM_go()

    print('----------------------------------------------')
    print('Delayed Rejection Adaptive Metropolis Hastings')
    print('----------------------------------------------')
    print('Acceptance Rate: %f' % results['accepted'])
    print("The median of the posterior is: %f" % np.median(results['MCMC']))
    print()

elif option == 4:
    # The EnKF algorithm
    B = EnKF_mcmc(inp)
    results = B.EnKF_go()

    print('----------------------------------------------')
    print('Ensemble Kalman Filter')
    print('----------------------------------------------')
    print('Acceptance Rate: %f' % results['accepted'])
    print("The median of the posterior is: %f" % np.median(results['MCMC']))
    print()
