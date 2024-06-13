from config import config
import numpy as np
import utilities
from MH import MH_mcmc
from EnKF import EnKF_mcmc
from DRAM import DRAM_algorithm
from AMH import AMH_mcmc
from MH_DR import MH_DR_mcmc

inp = {}

# range of the parameters based on the prior density
inp['range']=np.array([[config['Imposed limits']['Youngs Modulus'][0], config['Imposed limits']['Poissons Ratio'][0]], 
                       [config['Imposed limits']['Youngs Modulus'][1], config['Imposed limits']['Poissons Ratio'][1]]])                          

# number of iteration in MCMC
inp['nsamples']=config['Number of samples']

# initial covariance                          
inp['icov']=np.array([[config['Initial Variance']['Youngs Modulus'], 0],[0, config['Initial Variance']['Poissons Ratio']]])                                   

# initial guesss of the parameters based on the prior
inp['theta0']=np.array([[config['Initial Material Parameters']['Youngs Modulus']],[config['Initial Material Parameters']['Poissons Ratio']]])  

# std                               
inp['sigma']=config['Standard Deviation']

# The starting point of the Kalman MCMC           
inp['Kalmans']= config['Starting Kalman point']                        

# assumed measurement error for Kalman MCMC
inp['me']=config['Measurement error for Kalman'] 

#mesh set up
inp['mesh'] = [config['Mesh grid']['top'],
               config['Mesh grid']['bot'],
               config['Mesh grid']['left'],
               config['Mesh grid']['right']]

measurements=utilities.forward_model(np.array([[config['True Material Parameters']['Youngs Modulus']],[config['True Material Parameters']['Poissons Ratio']]]), inp['mesh'])
measurements += np.random.normal(0, config['Measurement Noise'], size = [np.size(measurements, 0), np.size(measurements, 1)])
inp['measurement'] = measurements

inp['Method'] = config['Methods']['Choosen Method']

print()
print()
print('True Youngs Modulus: %.3f' % config['True Material Parameters']['Youngs Modulus'])
print('True Poissons Ratio: %.3f' % config['True Material Parameters']['Poissons Ratio'])
print('Standard Deviation of Noise on Measurement Data: %.3f' %config['Measurement Noise'])
print()
if inp['Method'] == 0:
    # The Metropolis-Hastings technique
    C = MH_mcmc(inp)
    results = C.MH_go()
    if config['Print chain'] == 1:
        print(results['MCMC'])
    print('----------------------------------------------')
    print('Metropolis Hastings')
    print('----------------------------------------------')
    print('Acceptance Rate: %.3f' % results['accepted'])
    print('Number of Samples: %.0f' % config['Number of samples'])
    print('The median of the Youngs Modulus posterior is: %f' % np.median(results['MCMC'][0]))
    print('The median of the Poissons Ratio posterior is: %f' % np.median(results['MCMC'][1]))
    print()

elif inp['Method'] == 1:
    # The AMH algorithm 
    A = AMH_mcmc(inp)
    results = A.AMH_go()
    if config['Print chain'] == 1:
        print(results['MCMC'])
    print('----------------------------------------------')
    print('Adaptive Metropolis Hastings')
    print('----------------------------------------------')
    print('Acceptance Rate: %.3f' % results['accepted'])
    print('Number of Samples: %.0f' % config['Number of samples'])
    print('The median of the Youngs Modulus posterior is: %f' % np.median(results['MCMC'][0]))
    print('The median of the Poissons Ratio posterior is: %f' % np.median(results['MCMC'][1]))
    print()

elif inp['Method'] == 2:
    # The AMH algorithm 
    A = MH_DR_mcmc(inp)
    results = A.MH_DR_go()
    if config['Print chain'] == 1:
        print(results['MCMC'])
    print('----------------------------------------------')
    print('Metropolis Hastings Delayed Rejection')
    print('----------------------------------------------')
    print('Acceptance Rate: %.3f' % results['accepted'])
    print('Number of Samples: %.0f' % config['Number of samples'])
    print('The median of the Youngs Modulus posterior is: %f' % np.median(results['MCMC'][0]))
    print('The median of the Poissons Ratio posterior is: %f' % np.median(results['MCMC'][1]))
    print()

elif inp['Method'] == 3:
    # The DRAM algorithm 
    A = DRAM_algorithm(inp)
    results = A.DRAM_go()
    if config['Print chain'] == 1:
        print(results['MCMC'])
    print('----------------------------------------------')
    print('Delayed Rejection Adaptive Metropolis Hastings')
    print('----------------------------------------------')
    print('Acceptance Rate: %.3f' % results['accepted'])
    print('Number of Samples: %.0f' % config['Number of samples'])
    print('The median of the Youngs Modulus posterior is: %f' % np.median(results['MCMC'][0]))
    print('The median of the Poissons Ratio posterior is: %f' % np.median(results['MCMC'][1]))
    print()

elif inp['Method'] == 4:
    #The EnKF algorithm 
    B = EnKF_mcmc(inp)
    results = B.EnKF_go()
    if config['Print chain'] == 1:
        print(results['MCMC'])
    print('----------------------------------------------')
    print('Ensemble Kalman Filter')
    print('----------------------------------------------')
    print('Acceptance Rate: %.3f' % results['accepted'])
    print('Number of Samples: %.0f' % config['Number of samples'])
    print('The median of the Youngs Modulus posterior is: %f' % np.median(results['MCMC'][0]))
    print('The median of the Poissons Ratio posterior is: %f' % np.median(results['MCMC'][1]))
    print()