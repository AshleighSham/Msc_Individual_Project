from config import config
import numpy as np
import utilities
from MH import MH_mcmc
from EnKF import EnKF_mcmc
from DRAM import DRAM_algorithm
from AMH import AMH_mcmc
from MH_DR import MH_DR_mcmc
from mesh import Mesh
import matplotlib.pyplot as plt

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
inp['mesh'] = [config['Mesh grid']['quad'], 
               config['Mesh grid']['Nodal Coordinates'], 
               config['Mesh grid']['Element Node Numbers'],
               config['Mesh grid']['Number of elements'],
               config['Mesh grid']['Force Magnitude'],
               config['Mesh grid']['Force Nodes'],
               config['Mesh grid']['Fixed Nodes']]

measurements=utilities.forward_model(np.array([[config['True Material Parameters']['Youngs Modulus']],[config['True Material Parameters']['Poissons Ratio']]]), inp['mesh'])
measurements += np.random.normal(0, config['Measurement Noise'], size = [np.size(measurements, 0), np.size(measurements, 1)])
inp['measurement'] = measurements

my_mesh = Mesh(inp['mesh'])
my_mesh.displacement(config['True Material Parameters']['Youngs Modulus'], config['True Material Parameters']['Poissons Ratio'])
my_mesh.deformation_plot(title = f'True Deformation, E: %.3f, v: %.3f' % (config['True Material Parameters']['Youngs Modulus'], config['True Material Parameters']['Poissons Ratio']))

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
    if config['Print Chain'] == 1:
        print(results['MCMC'])
    print('----------------------------------------------')
    print('Metropolis Hastings')
    print('----------------------------------------------')
    print('Acceptance Rate: %.3f' % results['accepted'])
    print('Number of Samples: %.0f' % config['Number of samples'])
    print('The median of the Youngs Modulus posterior is: %f' % np.median(results['MCMC'][0]))
    print('The median of the Poissons Ratio posterior is: %f' % np.median(results['MCMC'][1]))
    print()
    my_mesh = Mesh(inp['mesh'])
    my_mesh.displacement(np.median(results['MCMC'][0]), np.median(results['MCMC'][1]))
    my_mesh.deformation_plot(title = f'Estimated Deformation, E: %.3f, v: %.3f' % (np.median(results['MCMC'][0]), np.median(results['MCMC'][1])))


elif inp['Method'] == 1:
    # The AMH algorithm 
    A = AMH_mcmc(inp)
    results = A.AMH_go()
    if config['Print Chain'] == 1:
        print(results['MCMC'])
    print('----------------------------------------------')
    print('Adaptive Metropolis Hastings')
    print('----------------------------------------------')
    print('Acceptance Rate: %.3f' % results['accepted'])
    print('Number of Samples: %.0f' % config['Number of samples'])
    print('The median of the Youngs Modulus posterior is: %f' % np.median(results['MCMC'][0]))
    print('The median of the Poissons Ratio posterior is: %f' % np.median(results['MCMC'][1]))
    print()
    my_mesh = Mesh(inp['mesh'])
    my_mesh.displacement(np.median(results['MCMC'][0]), np.median(results['MCMC'][1]))
    my_mesh.deformation_plot(title = f'Estimated Deformation, E: %.3f, v: %.3f' % (np.median(results['MCMC'][0]), np.median(results['MCMC'][1])))

elif inp['Method'] == 2:
    # The MH DR algorithm 
    A = MH_DR_mcmc(inp)
    results = A.MH_DR_go()
    if config['Print Chain'] == 1:
        print(results['MCMC'])
    print('----------------------------------------------')
    print('Metropolis Hastings Delayed Rejection')
    print('----------------------------------------------')
    print('Acceptance Rate: %.3f' % results['accepted'])
    print('Number of Samples: %.0f' % config['Number of samples'])
    print('The median of the Youngs Modulus posterior is: %f' % np.median(results['MCMC'][0]))
    print('The median of the Poissons Ratio posterior is: %f' % np.median(results['MCMC'][1]))
    print()
    my_mesh = Mesh(inp['mesh'])
    my_mesh.displacement(np.median(results['MCMC'][0]), np.median(results['MCMC'][1]))
    my_mesh.deformation_plot(title = f'Estimated Deformation, E: %.3f, v: %.3f' % (np.median(results['MCMC'][0]), np.median(results['MCMC'][1])))

elif inp['Method'] == 3:
    # The DRAM algorithm 
    A = DRAM_algorithm(inp)
    results = A.DRAM_go()
    if config['Print Chain'] == 1:
        print(results['MCMC'])
    print('----------------------------------------------')
    print('Delayed Rejection Adaptive Metropolis Hastings')
    print('----------------------------------------------')
    print('Acceptance Rate: %.3f' % results['accepted'])
    print('Number of Samples: %.0f' % config['Number of samples'])
    print('The median of the Youngs Modulus posterior is: %f' % np.median(results['MCMC'][0]))
    print('The median of the Poissons Ratio posterior is: %f' % np.median(results['MCMC'][1]))
    print()
    my_mesh = Mesh(inp['mesh'])
    my_mesh.displacement(np.median(results['MCMC'][0]), np.median(results['MCMC'][1]))
    my_mesh.deformation_plot(title = f'Estimated Deformation, E: %.3f, v: %.3f' % (np.median(results['MCMC'][0]), np.median(results['MCMC'][1])))

elif inp['Method'] == 4:
    #The EnKF algorithm 
    B = EnKF_mcmc(inp)
    results = B.EnKF_go()
    if config['Print Chain'] == 1:
        print(results['MCMC'])
    print('----------------------------------------------')
    print('Ensemble Kalman Filter')
    print('----------------------------------------------')
    print('Acceptance Rate: %.3f' % results['accepted'])
    print('Number of Samples: %.0f' % config['Number of samples'])
    print('The median of the Youngs Modulus posterior is: %f' % np.median(results['MCMC'][0]))
    print('The median of the Poissons Ratio posterior is: %f' % np.median(results['MCMC'][1]))
    print()
    my_mesh = Mesh(inp['mesh'])
    my_mesh.displacement(np.median(results['MCMC'][0]), np.median(results['MCMC'][1]))
    my_mesh.deformation_plot(title = f'Estimated Deformation, E: %.3f, v: %.3f' % (np.median(results['MCMC'][0]), np.median(results['MCMC'][1])))

plt.show()