from config import config
import numpy as np
import utilities
from MH import MH_mcmc
from EnKF import EnKF_mcmc
from DRAM import DRAM_algorithm
from AMH import AMH_mcmc
from MH_DR import MH_DR_mcmc
from EnKFbalnk import EnKF_mcmc2
from FMH import FMH_mcmc
from mesh import Mesh
from crank import Crank_mcmc
from baby import Baby_mcmc
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk')

inp = {}
hist = {0:[], 1:[], 2:[], 3:[]}
med = {0:[],1:[], 2:[], 3:[]}
#chains = {0:[], 1:[], 2:[], 3:[]}
# chains = np.load('4D_chains_ACTUAL_beam.npy', allow_pickle=True)
# chains = chains.tolist()

# range of the parameters based on the prior density
minis = np.array([np.array(config['Imposed Limits']['Youngs Modulus'])[:,0], np.array(config['Imposed Limits']['Poissons Ratio'])[:,0]]).flatten()
maxis = np.array([np.array(config['Imposed Limits']['Youngs Modulus'])[:,1], np.array(config['Imposed Limits']['Poissons Ratio'])[:,1]]).flatten()
inp['range']=np.array([minis, maxis])   
inp['s'] = config['s']                      

# number of iteration in MCMC
inp['nsamples']=config['Number of samples']

#freeze
inp['freeze'] = config['Freeze time']
inp['delay'] = config['Freeze delay']
inp['freeze loops'] = config['Freeze loops'] 

# initial covariance                          
inp['icov']=np.array([[config['Initial Variance']['Youngs Modulus'], 0],[0, config['Initial Variance']['Poissons Ratio']]])                                   
inp['icov'] = np.array([[1,0,0,0],[0,1,0,0],[0,0,1e-3,0],[0,0,0,1e-3]])

# initial guesss of the parameters based on the prior
itheta = [[config['Initial Material Parameters']['Youngs Modulus'][0]],[config['Initial Material Parameters']['Youngs Modulus'][1]], [config['Initial Material Parameters']['Poissons Ratio'][0]],[config['Initial Material Parameters']['Poissons Ratio'][1]]]
inp['theta0']=np.array(itheta)
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
               config['Mesh grid']['Element ID']]

ini = [config['True Material Parameters']['Youngs Modulus'][0], config['True Material Parameters']['Youngs Modulus'][1], config['True Material Parameters']['Poissons Ratio'][0],config['True Material Parameters']['Poissons Ratio'][1]]
measurements=utilities.forward_model(np.array(ini), inp['mesh'])
measurements1 = measurements + np.random.normal(0, config['Measurement Noise']*config['Mesh grid']['sf'], size = [np.size(measurements, 0), np.size(measurements, 1)])
inp['measurement'] = measurements1
Fig, Ax = plt.subplots(4,1)

# A = MH_mcmc(inp)
# resultsA = A.MH_go()
# #utilities.histogram_bulk(resultsA['MCMC'], 'MH', [[0, 30],[0,30],[0,0.5],[0, 0.5]],Fig, Ax, 'deepskyblue', 0.8, 3, hist)
# med[0].append(np.median(resultsA['MCMC'][0]))
# med[1].append(np.median(resultsA['MCMC'][1]))
# med[2].append(np.median(resultsA['MCMC'][2]))
# med[3].append(np.median(resultsA['MCMC'][3]))

# chains[0].append(resultsA['MCMC'][0])
# chains[1].append(resultsA['MCMC'][1])
# chains[2].append(resultsA['MCMC'][2])
# chains[3].append(resultsA['MCMC'][3])

# print('----------------------------------------------')
# print('Metropolis Hastings')
# print('----------------------------------------------')
# print('Acceptance Rate: %.3f' % resultsA['accepted'])
# print('Number of Samples: %.0f' % config['Number of samples'])
# print('The median of the E 1 posterior is: %f, with uncertainty +/- %.5f' % (np.median(resultsA['MCMC'][0]), np.sqrt(np.var(resultsA['MCMC'][0]))))
# print('The median of the E 2 posterior is: %f, with uncertainty +/- %.5f' % (np.median(resultsA['MCMC'][1]), np.sqrt(np.var(resultsA['MCMC'][1]))))
# print('The median of the V 1 posterior is: %f, with uncertainty +/- %.5f' % (np.median(resultsA['MCMC'][2]), np.sqrt(np.var(resultsA['MCMC'][2]))))
# print('The median of the V 2 posterior is: %f, with uncertainty +/- %.5f' % (np.median(resultsA['MCMC'][3]), np.sqrt(np.var(resultsA['MCMC'][3]))))
# print()

# inp['nsamples'] = config['Number of samples']
# B = AMH_mcmc(inp)
# resultsB = B.AMH_go()
# #utilities.histogram_bulk(resultsB['MCMC'], 'AMH',  [[0, 30],[0,30],[0,0.5],[0, 0.5]],Fig, Ax, 'mediumseagreen', 0.8, 3, hist)
# med[0].append(np.median(resultsB['MCMC'][0]))
# med[1].append(np.median(resultsB['MCMC'][1]))
# med[2].append(np.median(resultsB['MCMC'][2]))
# med[3].append(np.median(resultsB['MCMC'][3]))

# chains[0].append(resultsB['MCMC'][0])
# chains[1].append(resultsB['MCMC'][1])
# chains[2].append(resultsB['MCMC'][2])
# chains[3].append(resultsB['MCMC'][3])
# print('----------------------------------------------')
# print('AMH')
# print('----------------------------------------------')
# print('Acceptance Rate: %.3f' % resultsB['accepted'])
# print('Number of Samples: %.0f' % config['Number of samples'])
# print('The median of the E 1 posterior is: %f, with uncertainty +/- %.5f' % (np.median(resultsB['MCMC'][0]), np.sqrt(np.var(resultsB['MCMC'][0]))))
# print('The median of the E 2 posterior is: %f, with uncertainty +/- %.5f' % (np.median(resultsB['MCMC'][1]), np.sqrt(np.var(resultsB['MCMC'][1]))))
# print('The median of the V 1 posterior is: %f, with uncertainty +/- %.5f' % (np.median(resultsB['MCMC'][2]), np.sqrt(np.var(resultsB['MCMC'][2]))))
# print('The median of the V 2 posterior is: %f, with uncertainty +/- %.5f' % (np.median(resultsB['MCMC'][3]), np.sqrt(np.var(resultsB['MCMC'][3]))))
# print()

# inp['nsamples'] = config['Number of samples']
# C = MH_DR_mcmc(inp)
# resultsC = C.MH_DR_go()
# #utilities.histogram_bulk(resultsC['MCMC'], 'DR MH', [[0, 30],[0,30],[0,0.5],[0, 0.5]],Fig, Ax, 'orange', 0.8, 3, hist)
# med[0].append(np.median(resultsC['MCMC'][0]))
# med[1].append(np.median(resultsC['MCMC'][1]))
# med[2].append(np.median(resultsC['MCMC'][2]))
# med[3].append(np.median(resultsC['MCMC'][3]))

# chains[0].append(resultsC['MCMC'][0])
# chains[1].append(resultsC['MCMC'][1])
# chains[2].append(resultsC['MCMC'][2])
# chains[3].append(resultsC['MCMC'][3])
# print('----------------------------------------------')
# print('DR MH')
# print('----------------------------------------------')
# print('Acceptance Rate: %.3f' % resultsC['accepted'])
# print('Number of Samples: %.0f' % config['Number of samples'])
# print('The median of the E 1 posterior is: %f, with uncertainty +/- %.5f' % (np.median(resultsC['MCMC'][0]), np.sqrt(np.var(resultsC['MCMC'][0]))))
# print('The median of the E 2 posterior is: %f, with uncertainty +/- %.5f' % (np.median(resultsC['MCMC'][1]), np.sqrt(np.var(resultsC['MCMC'][1]))))
# print('The median of the V 1 posterior is: %f, with uncertainty +/- %.5f' % (np.median(resultsC['MCMC'][2]), np.sqrt(np.var(resultsC['MCMC'][2]))))
# print('The median of the V 2 posterior is: %f, with uncertainty +/- %.5f' % (np.median(resultsC['MCMC'][3]), np.sqrt(np.var(resultsC['MCMC'][3]))))
# print()

# inp['nsamples'] = config['Number of samples']
# D = DRAM_algorithm(inp)
# resultsD = D.DRAM_go()
# #utilities.histogram_bulk(resultsD['MCMC'], 'DRAM', [[0, 30],[0,30],[0,0.5],[0, 0.5]],Fig, Ax, 'hotpink', 0.8, 3, hist)
# med[0].append(np.median(resultsD['MCMC'][0]))
# med[1].append(np.median(resultsD['MCMC'][1]))
# med[2].append(np.median(resultsD['MCMC'][2]))
# med[3].append(np.median(resultsD['MCMC'][3]))

# chains[0].append(resultsD['MCMC'][0])
# chains[1].append(resultsD['MCMC'][1])
# chains[2].append(resultsD['MCMC'][2])
# chains[3].append(resultsD['MCMC'][3])
# print('----------------------------------------------')
# print('DRAM')
# print('----------------------------------------------')
# print('Acceptance Rate: %.3f' % resultsD['accepted'])
# print('Number of Samples: %.0f' % config['Number of samples'])
# print('The median of the E 1 posterior is: %f, with uncertainty +/- %.5f' % (np.median(resultsD['MCMC'][0]), np.sqrt(np.var(resultsD['MCMC'][0]))))
# print('The median of the E 2 posterior is: %f, with uncertainty +/- %.5f' % (np.median(resultsD['MCMC'][1]), np.sqrt(np.var(resultsD['MCMC'][1]))))
# print('The median of the V 1 posterior is: %f, with uncertainty +/- %.5f' % (np.median(resultsD['MCMC'][2]), np.sqrt(np.var(resultsD['MCMC'][2]))))
# print('The median of the V 2 posterior is: %f, with uncertainty +/- %.5f' % (np.median(resultsD['MCMC'][3]), np.sqrt(np.var(resultsD['MCMC'][3]))))
# print()

# inp['nsamples'] = config['Number of samples']
# E = Crank_mcmc(inp)
# resultsE = E.Crank_go()
# #utilities.histogram_bulk(resultsE['MCMC'], 'pCN',  [[0, 30],[0,30],[0,0.5],[0, 0.5]],Fig, Ax, 'mediumorchid', 0.8, 3, hist)
# med[0].append(np.median(resultsE['MCMC'][0]))
# med[1].append(np.median(resultsE['MCMC'][1]))
# med[2].append(np.median(resultsE['MCMC'][2]))
# med[3].append(np.median(resultsE['MCMC'][3]))

# chains[0].append(resultsE['MCMC'][0])
# chains[1].append(resultsE['MCMC'][1])
# chains[2].append(resultsE['MCMC'][2])
# chains[3].append(resultsE['MCMC'][3])
# print('----------------------------------------------')
# print('CRANK')
# print('----------------------------------------------')
# print('Acceptance Rate: %.3f' % resultsE['accepted'])
# print('Number of Samples: %.0f' % config['Number of samples'])
# print('The median of the E 1 posterior is: %f, with uncertainty +/- %.5f' % (np.median(resultsE['MCMC'][0]), np.sqrt(np.var(resultsE['MCMC'][0]))))
# print('The median of the E 2 posterior is: %f, with uncertainty +/- %.5f' % (np.median(resultsE['MCMC'][1]), np.sqrt(np.var(resultsE['MCMC'][1]))))
# print('The median of the V 1 posterior is: %f, with uncertainty +/- %.5f' % (np.median(resultsE['MCMC'][2]), np.sqrt(np.var(resultsE['MCMC'][2]))))
# print('The median of the V 2 posterior is: %f, with uncertainty +/- %.5f' % (np.median(resultsE['MCMC'][3]), np.sqrt(np.var(resultsE['MCMC'][3]))))
# print()

# inp['nsamples'] = config['Number of samples']
# F = EnKF_mcmc(inp)
# resultsF = F.EnKF_go()
# #utilities.histogram_bulk(resultsF['MCMC'], 'EnKF', [[0, 30],[0,30],[0,0.5],[0, 0.5]],Fig, Ax, 'mediumvioletred', 0.8, 3, hist)
# med[0].append(np.median(resultsF['MCMC'][0]))
# med[1].append(np.median(resultsF['MCMC'][1]))
# med[2].append(np.median(resultsF['MCMC'][2]))
# med[3].append(np.median(resultsF['MCMC'][3]))

# chains[0].append(resultsF['MCMC'][0])
# chains[1].append(resultsF['MCMC'][1])
# chains[2].append(resultsF['MCMC'][2])
# chains[3].append(resultsF['MCMC'][3])

# print('----------------------------------------------')
# print('EnKF')
# print('----------------------------------------------')
# print('Acceptance Rate: %.3f' % resultsF['accepted'])
# print('Number of Samples: %.0f' % config['Number of samples'])
# print('The median of the E 1 posterior is: %f, with uncertainty +/- %.5f' % (np.median(resultsF['MCMC'][0]), np.sqrt(np.var(resultsF['MCMC'][0]))))
# print('The median of the E 2 posterior is: %f, with uncertainty +/- %.5f' % (np.median(resultsF['MCMC'][1]), np.sqrt(np.var(resultsF['MCMC'][1]))))
# print('The median of the V 1 posterior is: %f, with uncertainty +/- %.5f' % (np.median(resultsF['MCMC'][2]), np.sqrt(np.var(resultsF['MCMC'][2]))))
# print('The median of the V 2 posterior is: %f, with uncertainty +/- %.5f' % (np.median(resultsF['MCMC'][3]), np.sqrt(np.var(resultsF['MCMC'][3]))))
# print()

# minis = np.array([np.array(config['Imposed Limits']['Youngs Modulus'])[:,0], np.array(config['Imposed Limits']['Poissons Ratio'])[:,0]]).flatten()
# maxis = np.array([np.array(config['Imposed Limits']['Youngs Modulus'])[:,1], np.array(config['Imposed Limits']['Poissons Ratio'])[:,1]]).flatten()
# inp['range']=np.array([minis, maxis])   
# inp['s'] = config['s']                      

# # number of iteration in MCMC
# inp['nsamples']=config['Number of samples']

# #freeze
# inp['freeze'] = config['Freeze time']
# inp['delay'] = config['Freeze delay']
# inp['freeze loops'] = config['Freeze loops'] 

# # initial covariance                          
# inp['icov']=np.array([[config['Initial Variance']['Youngs Modulus'], 0],[0, config['Initial Variance']['Poissons Ratio']]])                                   
# inp['icov'] = np.array([[1,0,0,0],[0,1,0,0],[0,0,1e-3,0],[0,0,0,1e-3]])

# # initial guesss of the parameters based on the prior
# itheta = [[config['Initial Material Parameters']['Youngs Modulus'][0]],[config['Initial Material Parameters']['Youngs Modulus'][1]], [config['Initial Material Parameters']['Poissons Ratio'][0]],[config['Initial Material Parameters']['Poissons Ratio'][1]]]
# inp['theta0']=np.array(itheta)
# # std                               
# inp['sigma']=config['Standard Deviation']

# # The starting point of the Kalman MCMC           
# inp['Kalmans']= config['Starting Kalman point']                        

# # assumed measurement error for Kalman MCMC
# inp['me']=config['Measurement error for Kalman'] 

# #adaption setp size
# inp['adapt'] = config['Adaption Step Size']

# #mesh set up
# inp['mesh'] = [config['Mesh grid']['quad'],
#                config['Mesh grid']['sf'], 
#                config['Mesh grid']['Nodal Coordinates'], 
#                config['Mesh grid']['Element Node Numbers'],
#                config['Mesh grid']['Number of elements'],
#                config['Mesh grid']['Force Magnitude'],
#                config['Mesh grid']['Force Nodes'],
#                config['Mesh grid']['Fixed Nodes'],
#                config['Mesh grid']['Element ID']]

G = Baby_mcmc(inp)
resultsG = G.Baby_go()
#utilities.histogram_bulk(resultsG['MCMC'], 'Baby', [[0, 30],[0,30],[0,0.5],[0, 0.5]],Fig, Ax, 'red', 0.8, 3, hist)
med[0].append(np.median(resultsG['MCMC'][0]))
med[1].append(np.median(resultsG['MCMC'][1]))
med[2].append(np.median(resultsG['MCMC'][2]))
med[3].append(np.median(resultsG['MCMC'][3]))

chains[0][-1] = resultsG['MCMC'][0]
chains[1][-1] = resultsG['MCMC'][1]
chains[2][-1] = resultsG['MCMC'][2]
chains[3][-1] = resultsG['MCMC'][3]
print('----------------------------------------------')
print('Baby')
print('----------------------------------------------')
print('Acceptance Rate: %.3f' % resultsG['accepted'])
print('Number of Samples: %.0f' % config['Number of samples'])
print('The median of the E 1 posterior is: %f, with uncertainty +/- %.5f' % (np.median(resultsG['MCMC'][0]), np.sqrt(np.var(resultsG['MCMC'][0]))))
print('The median of the E 2 posterior is: %f, with uncertainty +/- %.5f' % (np.median(resultsG['MCMC'][1]), np.sqrt(np.var(resultsG['MCMC'][1]))))
print('The median of the V 1 posterior is: %f, with uncertainty +/- %.5f' % (np.median(resultsG['MCMC'][2]), np.sqrt(np.var(resultsG['MCMC'][2]))))
print('The median of the V 2 posterior is: %f, with uncertainty +/- %.5f' % (np.median(resultsG['MCMC'][3]), np.sqrt(np.var(resultsG['MCMC'][3]))))
print()
#plt.show()

# data = {}
# data['graphs'] = hist
# data['values'] = med
# np.save('my_fileRVE.npy', data) 

# data = {}
# data['graphs'] = hist
# data['values'] = med
# np.save('my_fileBEAM.npy', data) 

# np.save('4D_chains_ACTUAL_beam.npy', chains)