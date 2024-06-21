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

fig, ax1 = plt.subplots()
plt.subplots_adjust(bottom = 0.1)

inp = {}

# range of the parameters based on the prior density
minis, maxis = [], []
for i in range(config['Number of Materials']):
    minis.append(config['Imposed limits']['Youngs Modulus'][0])
    minis.append(config['Imposed limits']['Poissons Ratio'][0])
    maxis.append(config['Imposed limits']['Youngs Modulus'][1])
    maxis.append(config['Imposed limits']['Poissons Ratio'][1])
inp['range']=np.array([minis, maxis])                      

# number of iteration in MCMC
inp['nsamples']=config['Number of samples']

# initial covariance  
icov = []
for i in range(config['Number of Materials']):
    icov.append(config['Initial Variance']['Youngs Modulus'])
    icov.append(config['Initial Variance']['Poissons Ratio'])
inp['icov'] = np.eye(config['Number of Materials']*2)*np.array(icov)

# initial guesss of the parameters based on the prior
itheta = []
for i in range(config['Number of Materials']):
    itheta.append([config['Initial Material Parameters']['Youngs Modulus'][i]])
    itheta.append([config['Initial Material Parameters']['Poissons Ratio'][i]])
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

ini = []
for i in range(config['Number of Materials']):
    ini.append(config['True Material Parameters']['Youngs Modulus'][i])
    ini.append(config['True Material Parameters']['Poissons Ratio'][i])    
measurements=utilities.forward_model(np.array(ini), inp['mesh'])
measurements += np.random.normal(0, config['Measurement Noise'], size = [np.size(measurements, 0), np.size(measurements, 1)])
inp['measurement'] = measurements

lines = []
my_mesh = Mesh(inp['mesh'])
my_mesh.displacement(config['True Material Parameters']['Youngs Modulus'], config['True Material Parameters']['Poissons Ratio'])
my_mesh.deformation_plot(label = f'True Deformation, E1: %.3f, E2: %.3f, v1: %.3f, v2: %.3f' % (config['True Material Parameters']['Youngs Modulus'][0], config['True Material Parameters']['Youngs Modulus'][1], config['True Material Parameters']['Poissons Ratio'][0], config['True Material Parameters']['Poissons Ratio'][1]), 
                         colour= 'plum', ch = 1, ax = ax1, lines = lines, ls = 'solid')

inp['Method'] = config['Methods']['Choosen Method']

print()
print()
for i in range(config['Number of Materials']):
    print('True Youngs Modulus %.0f: %.3f' % (i, config['True Material Parameters']['Youngs Modulus'][i]))
    print('True Poissons Ratio %.0f: %.3f' % (i, config['True Material Parameters']['Poissons Ratio'][i]))
print('Standard Deviation of Noise on Measurement Data: %.3f' %config['Measurement Noise'])
print()
if inp['Method'] == 0:
    # The Metropolis-Hastings technique
    C = MH_mcmc(inp)
    results = C.MH_go()
    if config['Print Chain'] == 1:
        print(results['MCMC'])
    utilities.histogram(results['MCMC'], ['Youngs Modulus', 'Poissons Ratio','Youngs Modulus', 'Poissons Ratio'], ini)
    print('----------------------------------------------')
    print('Metropolis Hastings')
    print('----------------------------------------------')

elif inp['Method'] == 1:
    # The AMH algorithm 
    A = AMH_mcmc(inp)
    results = A.AMH_go()
    if config['Print Chain'] == 1:
        print(results['MCMC'])
    utilities.histogram(results['MCMC'], ['Youngs Modulus', 'Poissons Ratio','Youngs Modulus', 'Poissons Ratio'], ini)
    print('----------------------------------------------')
    print('Adaptive Metropolis Hastings')
    print('----------------------------------------------')

elif inp['Method'] == 2:
    # The MH DR algorithm 
    A = MH_DR_mcmc(inp)
    results = A.MH_DR_go()
    if config['Print Chain'] == 1:
        print(results['MCMC'])
    utilities.histogram(results['MCMC'], ['Youngs Modulus', 'Poissons Ratio','Youngs Modulus', 'Poissons Ratio'], ini)
    print('----------------------------------------------')
    print('Metropolis Hastings Delayed Rejection')
    print('----------------------------------------------')

elif inp['Method'] == 3:
    # The DRAM algorithm 
    A = DRAM_algorithm(inp)
    results = A.DRAM_go()
    if config['Print Chain'] == 1:
        print(results['MCMC'])
    utilities.histogram(results['MCMC'], ['Youngs Modulus', 'Poissons Ratio','Youngs Modulus', 'Poissons Ratio'], ini)
    print('----------------------------------------------')
    print('Delayed Rejection Adaptive Metropolis Hastings')
    print('----------------------------------------------')

elif inp['Method'] == 4:
    #The EnKF algorithm 
    B = EnKF_mcmc(inp)
    results = B.EnKF_go()
    if config['Print Chain'] == 1:
        print(results['MCMC'])
    utilities.histogram(results['MCMC'], ['Youngs Modulus', 'Poissons Ratio','Youngs Modulus', 'Poissons Ratio'], ini)
    print('----------------------------------------------')
    print('Ensemble Kalman Filter')
    print('----------------------------------------------')
   
print('Acceptance Rate: %.3f' % results['accepted'])
print('Number of Samples: %.0f' % config['Number of samples'])
print('The median of the Youngs Modulus 1 posterior is: %f, with uncertainty +/- %.5f' % (np.median(results['MCMC'][0]), np.sqrt(np.var(results['MCMC'][0]))))
print('The median of the Poissons Ratio 1 posterior is: %f, with uncertainty +/- %.5f' % (np.median(results['MCMC'][1]), np.sqrt(np.var(results['MCMC'][1]))))
print('The median of the Youngs Modulus 2 posterior is: %f, with uncertainty +/- %.5f' % (np.median(results['MCMC'][2]), np.sqrt(np.var(results['MCMC'][2]))))
print('The median of the Poissons Ratio 2 posterior is: %f, with uncertainty +/- %.5f' % (np.median(results['MCMC'][3]), np.sqrt(np.var(results['MCMC'][3]))))
print()
my_mesh = Mesh(inp['mesh'])
my_mesh.displacement([np.median(results['MCMC'][0]),np.median(results['MCMC'][2])], [np.median(results['MCMC'][1]), np.median(results['MCMC'][3])])
my_mesh.deformation_plot(label = f'Estimated Deformation, E1: %.3f, v1: %.3f, E2: %.3f, v2: %.3f ' % (np.median(results['MCMC'][0]), np.median(results['MCMC'][1]), np.median(results['MCMC'][2]),np.median(results['MCMC'][3])), ls =(0,(3,5)),colour = 'rebeccapurple', ch = 0.9, ax = ax1, lines = lines)

ax1.set_title('Deformation Plot', fontsize = 25)
fig.legend(loc = 'lower center', ncols=2)

plt.show()