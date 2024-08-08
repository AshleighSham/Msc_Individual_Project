from config import config
import numpy as np
import utilities
from MH import MH_mcmc
from EnKF import EnKF_mcmc
from DRAM import DRAM_algorithm
from AMH import AMH_mcmc
from MH_DR import MH_DR_mcmc
from mesh import Mesh
from crank import Crank_mcmc
from baby import Baby_mcmc
import matplotlib.pyplot as plt
import seaborn as sns
import time
sns.set_context('talk')

inp = {}

chains = {0:[], 1:[]}
times = []


# range of the parameters based on the prior density
inp['range']=np.array([[config['Imposed limits']['Youngs Modulus'][0], config['Imposed limits']['Poissons Ratio'][0]], 
                       [config['Imposed limits']['Youngs Modulus'][1], config['Imposed limits']['Poissons Ratio'][1]]])    

inp['s'] = config['s']                      

# number of iteration in MCMC
inp['nsamples']=config['Number of samples']

# initial covariance                          
inp['icov']=np.array([[config['Initial Variance']['Youngs Modulus'], 0],[0, config['Initial Variance']['Poissons Ratio']]])                                   

# initial guesss of the parameters based on the prior
inp['theta0']=np.array([[config['Initial Material Parameters']['Youngs Modulus']],[config['Initial Material Parameters']['Poissons Ratio']]])  
#inp['theta0'] += inp['icov']@np.random.normal(size = (2, 1))
# std                               
inp['sigma']=config['Standard Deviation']

# The starting point of the Kalman MCMC           
inp['Kalmans']= config['Starting Kalman point']                        

# assumed measurement error for Kalman MCMC
inp['me']=config['Measurement error for Kalman'] 

#adaption setp size
inp['adapt'] = config['Adaption Step Size']

#mesh set up
inp['mesh'] = [config['Mesh grid']['quad'], 
               config['Mesh grid']['sf'],
               config['Mesh grid']['Nodal Coordinates'], 
               config['Mesh grid']['Element Node Numbers'],
               config['Mesh grid']['Number of elements'],
               config['Mesh grid']['Force Magnitude'],
               config['Mesh grid']['Force Nodes'],
               config['Mesh grid']['Fixed Nodes'],
               config['Mesh grid']['thickness']]

ini = [config['True Material Parameters']['Youngs Modulus'], config['True Material Parameters']['Poissons Ratio']]

measurements=utilities.forward_model(np.array([[config['True Material Parameters']['Youngs Modulus']],[config['True Material Parameters']['Poissons Ratio']]]), inp['mesh'])
measurements1 = measurements + np.random.normal(0, config['Measurement Noise'], size = [np.size(measurements, 0), np.size(measurements, 1)])
inp['measurement'] = measurements1

st = time.perf_counter()
A = MH_mcmc(inp)
resultsA = A.MH_go()
times.append(time.perf_counter() - st)
chains[0].append(resultsA['MCMC'][0])
chains[1].append(resultsA['MCMC'][1])
print('----------------------------------------------')
print('Metropolis Hastings')
print('----------------------------------------------')
print('Acceptance Rate: %.3f' % resultsA['accepted'])
print('Number of Samples: %.0f' % config['Number of samples'])
print('The median of the Youngs Modulus posterior is: %f, with uncertainty +/- %.5f' % (np.median(resultsA['MCMC'][0]), np.sqrt(np.var(resultsA['MCMC'][0]))))
print('The median of the Poissons Ratio posterior is: %f, with uncertainty +/- %.5f' % (np.median(resultsA['MCMC'][1]), np.sqrt(np.var(resultsA['MCMC'][1]))))
print()

st = time.perf_counter()
B = AMH_mcmc(inp)
resultsB = B.AMH_go()
times.append(time.perf_counter() - st)
chains[0].append(resultsB['MCMC'][0])
chains[1].append(resultsB['MCMC'][1])
print('----------------------------------------------')
print('AMH')
print('----------------------------------------------')
print('Acceptance Rate: %.3f' % resultsB['accepted'])
print('Number of Samples: %.0f' % config['Number of samples'])
print('The median of the Youngs Modulus posterior is: %f, with uncertainty +/- %.5f' % (np.median(resultsB['MCMC'][0]), np.sqrt(np.var(resultsB['MCMC'][0]))))
print('The median of the Poissons Ratio posterior is: %f, with uncertainty +/- %.5f' % (np.median(resultsB['MCMC'][1]), np.sqrt(np.var(resultsB['MCMC'][1]))))
print()

st = time.perf_counter()
C = MH_DR_mcmc(inp)
resultsC = C.MH_DR_go()
times.append(time.perf_counter() - st)
chains[0].append(resultsC['MCMC'][0])
chains[1].append(resultsC['MCMC'][1])
print('----------------------------------------------')
print('DR MH')
print('----------------------------------------------')
print('Acceptance Rate: %.3f' % resultsC['accepted'])
print('Number of Samples: %.0f' % config['Number of samples'])
print('The median of the Youngs Modulus posterior is: %f, with uncertainty +/- %.5f' % (np.median(resultsC['MCMC'][0]), np.sqrt(np.var(resultsC['MCMC'][0]))))
print('The median of the Poissons Ratio posterior is: %f, with uncertainty +/- %.5f' % (np.median(resultsC['MCMC'][1]), np.sqrt(np.var(resultsC['MCMC'][1]))))
print()

st = time.perf_counter()
D = DRAM_algorithm(inp)
resultsD = D.DRAM_go()
times.append(time.perf_counter() - st)
chains[0].append(resultsD['MCMC'][0])
chains[1].append(resultsD['MCMC'][1])
print('----------------------------------------------')
print('DRAM')
print('----------------------------------------------')
print('Acceptance Rate: %.3f' % resultsD['accepted'])
print('Number of Samples: %.0f' % config['Number of samples'])
print('The median of the Youngs Modulus posterior is: %f, with uncertainty +/- %.5f' % (np.median(resultsD['MCMC'][0]), np.sqrt(np.var(resultsD['MCMC'][0]))))
print('The median of the Poissons Ratio posterior is: %f, with uncertainty +/- %.5f' % (np.median(resultsD['MCMC'][1]), np.sqrt(np.var(resultsD['MCMC'][1]))))
print()

st = time.perf_counter()
F = EnKF_mcmc(inp)
resultsF = F.EnKF_go()
times.append(time.perf_counter() - st)
chains[0].append(resultsF['MCMC'][0])
chains[1].append(resultsF['MCMC'][1])
print('----------------------------------------------')
print('EnKF')
print('----------------------------------------------')
print('Acceptance Rate: %.3f' % resultsF['accepted'])
print('Number of Samples: %.0f' % config['Number of samples'])
print('The median of the Youngs Modulus posterior is: %f, with uncertainty +/- %.5f' % (np.median(resultsF['MCMC'][0]), np.sqrt(np.var(resultsF['MCMC'][0]))))
print('The median of the Poissons Ratio posterior is: %f, with uncertainty +/- %.5f' % (np.median(resultsF['MCMC'][1]), np.sqrt(np.var(resultsF['MCMC'][1]))))
print()

# st = time.perf_counter()
# inp['nsamples'] = config['Number of samples']
# G = Baby_mcmc(inp)

# resultsG = G.Baby_go()
# times.append(time.perf_counter() - st)
# chains[0].append(resultsG['MCMC'][0])
# chains[1].append(resultsG['MCMC'][1])
# print('----------------------------------------------')
# print('Baby')
# print('----------------------------------------------')
# print('Acceptance Rate: %.3f' % resultsF['accepted'])
# print('Number of Samples: %.0f' % config['Number of samples'])
# print('The median of the Youngs Modulus posterior is: %f, with uncertainty +/- %.5f' % (np.median(resultsG['MCMC'][0]), np.sqrt(np.var(resultsG['MCMC'][0]))))
# print('The median of the Poissons Ratio posterior is: %f, with uncertainty +/- %.5f' % (np.median(resultsG['MCMC'][1]), np.sqrt(np.var(resultsG['MCMC'][1]))))
# print()

#plt.show()

print(times)
np.save('2D_chains_2.npy', chains) 