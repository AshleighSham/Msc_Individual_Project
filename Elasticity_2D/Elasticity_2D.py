from config import config
import numpy as np
import utilities
from MH import MH_mcmc
from EnKF import EnKF_mcmc
from DRAM import DRAM_algorithm
from AMH import AMH_mcmc
from MH_DR import MH_DR_mcmc
from mesh import Mesh
from SEnKF import S_EnKF_mcmc
import pandas as pd
import time
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()
plt.subplots_adjust(bottom = 0.1)

inp = {}

# data = np.load('2D_chains.npy', allow_pickle=True)
# data = data.tolist()

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

# std                               
inp['sigma']=config["Standard Deviation"]

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

figd, axd = plt.subplots()

axd.scatter(range(len(measurements)),measurements, s =10)
axd.scatter(range(len(measurements1)), measurements1, s= 10)

# plt.show()


lines = []
my_mesh = Mesh(inp['mesh'])
true_displacement = my_mesh.displacement(config['True Material Parameters']['Youngs Modulus'], config['True Material Parameters']['Poissons Ratio'])

my_mesh.deformation_plot(label = f'True Deformation, E: %.3f, v: %.3f' % (config['True Material Parameters']['Youngs Modulus'], config['True Material Parameters']['Poissons Ratio']), colour= 'plum', ch = 1, ax = ax1, lines = lines, ls = 'solid')

fig2, ax2 = plt.subplots(2, 1)
my_mesh.contour_plot('True', fig2, ax2)

inp['Method'] = config['Methods']['Choosen Method']
#inp['theta0'] = np.array([np.random.choice(range(int(inp['range'][0][0]*1e6), int(inp['range'][1][0]*1e6)), 1)/1e6, np.random.choice(range(int(inp['range'][0][1]*1e6), int(inp['range'][1][1]*1e6)), 1)/1e6])

print()
print()
print('True Youngs Modulus: %.3f' % config['True Material Parameters']['Youngs Modulus'])
print('True Poissons Ratio: %.3f' % config['True Material Parameters']['Poissons Ratio'])
print('Standard Deviation of Noise on Measurement Data: %f' %(config['Measurement Noise']))
print()
st = time.perf_counter()
if inp['Method'] == 0:
    # The Metropolis-Hastings technique
    C = MH_mcmc(inp)
    results = C.MH_go()
    print(f'End Time: {time.perf_counter() - st}')
    if config['Print Chain'] == 1:
        print(results['MCMC'])
    #fig5, ax5 = plt.subplots(2, 1)
    #utilities.histogram(results['MCMC'], 100, ['Youngs Modulus', 'Youngs Modulus','Poissons Ratio', 'Poissons Ratio'], ini, inp['range'], fig5, ax5)
    print('----------------------------------------------')
    print('Metropolis Hastings')
    print('----------------------------------------------')

elif inp['Method'] == 1:
    # The AMH algorithm 
    A = AMH_mcmc(inp)
    results = A.AMH_go()
    print(f'End Time: {time.perf_counter() - st}')
    if config['Print Chain'] == 1:
        print(results['MCMC'])
    #fig5, ax5 = plt.subplots(2, 1)
    #utilities.histogram(results['MCMC'], 100, ['Youngs Modulus', 'Youngs Modulus','Poissons Ratio', 'Poissons Ratio'], ini, inp['range'], fig5, ax5)
    print('----------------------------------------------')
    print('Adaptive Metropolis Hastings')
    print('----------------------------------------------')

elif inp['Method'] == 2:
    # The MH DR algorithm 
    A = MH_DR_mcmc(inp)
    results = A.MH_DR_go()
    print(f'End Time: {time.perf_counter() - st}')
    if config['Print Chain'] == 1:
        print(results['MCMC'])
    #fig5, ax5 = plt.subplots(2, 1)
    #utilities.histogram(results['MCMC'], 100, ['Youngs Modulus', 'Youngs Modulus','Poissons Ratio', 'Poissons Ratio'], ini, inp['range'], fig5, ax5)
    print('----------------------------------------------')
    print('Metropolis Hastings Delayed Rejection')
    print('----------------------------------------------')

elif inp['Method'] == 3:
    # The DRAM algorithm 
    A = DRAM_algorithm(inp)
    results = A.DRAM_go()
    print(f'End Time: {time.perf_counter() - st}')
    if config['Print Chain'] == 1:
        print(results['MCMC'])
    #fig5, ax5 = plt.subplots(2, 1)
    #utilities.histogram(results['MCMC'], 100, ['Youngs Modulus', 'Youngs Modulus','Poissons Ratio', 'Poissons Ratio'], ini, inp['range'], fig5, ax5)
    print('----------------------------------------------')
    print('Delayed Rejection Adaptive Metropolis Hastings')
    print('----------------------------------------------')

elif inp['Method'] == 4:
    #The EnKF algorithm 
    B = EnKF_mcmc(inp)
    results = B.EnKF_go()
    print(f'End Time: {time.perf_counter() - st}')
    if config['Print Chain'] == 1:
        print(results['MCMC'])
    #fig5, ax5 = plt.subplots(2, 1)
    #utilities.histogram(results['MCMC'], 100, ['Youngs Modulus', 'Youngs Modulus','Poissons Ratio', 'Poissons Ratio'], ini, inp['range'], fig5, ax5)
    print('----------------------------------------------')
    print('Ensemble Kalman Filter')
    print('----------------------------------------------')
   
print('Acceptance Rate: %.3f' % results['accepted'])
print('Number of Samples: %.0f' % config['Number of samples'])
print('The median of the Youngs Modulus posterior is: %f, with uncertainty +/- %.5f' % (np.median(results['MCMC'][0]), np.sqrt(np.var(results['MCMC'][0]))))
print('The median of the Poissons Ratio posterior is: %f, with uncertainty +/- %.5f' % (np.median(results['MCMC'][1]), np.sqrt(np.var(results['MCMC'][1]))))
print()

outi = np.array([[np.median(results['MCMC'][0])],[np.median(results['MCMC'][1])]])
res=utilities.forward_model(outi, inp['mesh'])

print(f'RMSE: {np.linalg.norm(measurements/1000 - res/1000)/len(measurements)}')

fig, ax = plt.subplots(2,1)
ax[0].plot(range(len(results['MCMC'][0])), results['MCMC'][0])
ax[0].set_ylim([0,35])

ax[1].plot(range(len(results['MCMC'][1])), results['MCMC'][1])
ax[1].set_ylim([0,0.5])

plt.show()
# for i, q in zip(read_dictionary[0], range(7)):
#     ax[q][0].plot(range(len(i)), i, alpha = 0.8, c = colours[q], label = labels[q])
#     ax[q][0].axhline(10, alpha = 0.8, c = 'k', linestyle='dashed')
#     ax[q][0].set_ylim([0,35])
# my_mesh = Mesh(inp['mesh'])
# my_mesh.displacement(np.median(results['MCMC'][0]), np.median(results['MCMC'][1]))
# my_mesh.deformation_plot(label = f'Estimated Deformation, E: %.3f, v: %.3f' % (np.median(results['MCMC'][0]), np.median(results['MCMC'][1])), ls =(0,(3,5)),colour = 'rebeccapurple', ch = 0.9, ax = ax1, lines = lines)

# fig3, ax3 = plt.subplots(2, 1)
# my_mesh.contour_plot('Estimated', fig3, ax3)

# fig4, ax4 = plt.subplots(2, 1)
# my_mesh.error_plot(true_displacement, fig4, ax4)

# ax1.set_title('Deformation Plot', fontsize = 25)
# fig.legend(loc = 'lower center', ncols=2)

# plt.show()

# data = {'Initial':{0:[], 1:[]}, 'Median':{0:[], 1:[]}, 'Uncertainty':{0:[], 1:[]}}
# for i in range(2):
#     data['Initial'][i].append(inp['theta0'][i][0])

# data[0][-1] = results['MCMC'][0]
# data[1][-1] = results['MCMC'][1]

#np.save('2D_chains.npy', data, allow_pickle=True) 
