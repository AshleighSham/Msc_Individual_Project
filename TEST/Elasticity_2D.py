from config import config
import numpy as np
import utilities
from FMH import FMH_mcmc
from EnKF import EnKF_mcmc
from DRAM import DRAM_algorithm
from AMH import AMH_mcmc
from MH_DR import MH_DR_mcmc
from mesh import Mesh
from MH2 import MH2_mcmc
from EnKFbalnk import EnKF_mcmc2
from crank import Crank_mcmc
import seaborn as sns
sns.set_context('talk')
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots(figsize = (5,5))
plt.subplots_adjust(bottom = 0.1)

inp = {}

# range of the parameters based on the prior density
minis = np.array([np.array(config['Imposed Limits']['Youngs Modulus'])[:,0], np.array(config['Imposed Limits']['Poissons Ratio'])[:,0]]).flatten()
maxis = np.array([np.array(config['Imposed Limits']['Youngs Modulus'])[:,1], np.array(config['Imposed Limits']['Poissons Ratio'])[:,1]]).flatten()
inp['range']=np.array([minis, maxis])  

inp['Priority'] = config['Priority']

inp['s'] = config['s']

# number of iteration in MCMC
inp['nsamples']=config['Number of samples']

#freeze
inp['freeze'] = config['Freeze time']
inp['delay'] = config['Freeze delay']
inp['freeze loops'] = config['Freeze loops'] 

# initial covariance  
icov = [config['Initial Variance']['Youngs Modulus'], config['Initial Variance']['Youngs Modulus'], config['Initial Variance']['Poissons Ratio'], config['Initial Variance']['Poissons Ratio']]
inp['icov'] = np.eye(config['Number of Materials']*2)*np.array(icov)
inp['icov'] = np.array([[1,0,0,0],[0,1,0,0],[0,0,5e-2,0],[0,0,0,5e-2]])

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


#edges
edges_ind = []
if inp['mesh'][0] != 0:
    A = range(inp['mesh'][0][1])
    edges_ind = [a for a in A]
    for i in range(inp['mesh'][0][0]-1):
        edges_ind.append(A[-1] + 1 + inp['mesh'][0][1]*i)
        edges_ind.append(A[-1] + inp['mesh'][0][1]*i)
    for i in range(inp['mesh'][0][1]):
        edges_ind.append(inp['mesh'][0][1]*inp['mesh'][0][0]-1 - i)

meas_edge = []
for i in edges_ind:
    meas_edge.append(2*i)
    meas_edge.append(2*i + 1)


ini = [config['True Material Parameters']['Youngs Modulus'][0], config['True Material Parameters']['Youngs Modulus'][1], config['True Material Parameters']['Poissons Ratio'][0],config['True Material Parameters']['Poissons Ratio'][1]]
measurements=utilities.forward_model(np.array(ini), inp['mesh'])
measurements1 = measurements + np.random.normal(0, config['Measurement Noise']*config['Mesh grid']['sf'], size = [np.size(measurements, 0), np.size(measurements, 1)])
inp['measurement'] = measurements1

lines = []
my_mesh = Mesh(inp['mesh'])
true_displacement = my_mesh.displacement(config['True Material Parameters']['Youngs Modulus'], config['True Material Parameters']['Poissons Ratio'])
my_mesh.deformation_plot(label = f'True Deformation, E1: %.3f, E2: %.3f, v1: %.3f, v2: %.3f' % (config['True Material Parameters']['Youngs Modulus'][0], config['True Material Parameters']['Youngs Modulus'][1], config['True Material Parameters']['Poissons Ratio'][0], config['True Material Parameters']['Poissons Ratio'][1]), 
                         colour= 'lightskyblue', ch = 1, ax = ax1, lines = lines, ls = 'solid')


fig2, ax2 = plt.subplots(2, 1)
my_mesh.contour_plot('True', fig2, ax2)

inp['Method'] = config['Methods']['Choosen Method']

print()
print()
for i in range(config['Number of Materials']):
    print('True Youngs Modulus %.0f: %.3f' % (i, config['True Material Parameters']['Youngs Modulus'][i]))
    print('True Poissons Ratio %.0f: %.3f' % (i, config['True Material Parameters']['Poissons Ratio'][i]))
print('Standard Deviation of Noise on Measurement Data: %.10f' %(config['Measurement Noise']*config['Mesh grid']['sf']))
print()
if inp['Method'] == 0:
    # The Metropolis-Hastings technique
    C = FMH_mcmc(inp)
    results = C.FMH_go()
    if config['Print Chain'] == 1:
        print(results['MCMC'])
    inp['nsamples'] += 500
    D = EnKF_mcmc2(inp, results)
    results2 = D.EnKF_go()
    fig5, ax5 = plt.subplots(4, 1)
    utilities.histogram(results['MCMC'], ['Youngs Modulus', 'Youngs Modulus','Poissons Ratio', 'Poissons Ratio'], ini, inp['range'], fig5, ax5)
    print('----------------------------------------------')
    print('Metropolis Hastings')
    print('----------------------------------------------')

elif inp['Method'] == 1:
    # The AMH algorithm 
    A = AMH_mcmc(inp)
    results = A.AMH_go()
    if config['Print Chain'] == 1:
        print(results['MCMC'])
    fig5, ax5 = plt.subplots(4, 1)
    utilities.histogram(results['MCMC'], ['Youngs Modulus', 'Youngs Modulus','Poissons Ratio', 'Poissons Ratio'], ini, inp['range'], fig5, ax5)
    print('----------------------------------------------')
    print('Adaptive Metropolis Hastings')
    print('----------------------------------------------')

elif inp['Method'] == 2:
    # The MH DR algorithm 
    A = MH_DR_mcmc(inp)
    results = A.MH_DR_go()
    if config['Print Chain'] == 1:
        print(results['MCMC'])
    fig5, ax5 = plt.subplots(4, 1)
    utilities.histogram(results['MCMC'],['Youngs Modulus', 'Youngs Modulus','Poissons Ratio', 'Poissons Ratio'], ini, inp['range'], fig5, ax5)
    print('----------------------------------------------')
    print('Metropolis Hastings Delayed Rejection')
    print('----------------------------------------------')

elif inp['Method'] == 3:
    # The DRAM algorithm 
    A = DRAM_algorithm(inp)
    results = A.DRAM_go()
    if config['Print Chain'] == 1:
        print(results['MCMC'])
    fig5, ax5 = plt.subplots(4, 1)
    utilities.histogram(results['MCMC'], ['Youngs Modulus', 'Youngs Modulus','Poissons Ratio', 'Poissons Ratio'], ini, inp['range'], fig5, ax5)
    print('----------------------------------------------')
    print('Delayed Rejection Adaptive Metropolis Hastings')
    print('----------------------------------------------')

elif inp['Method'] == 4:
    #The Crank algorithm 
    B = Crank_mcmc(inp)
    results = B.Crank_go()
    if config['Print Chain'] == 1:
        print(results['MCMC'])
    fig5, ax5 = plt.subplots(4, 1)
    utilities.histogram(results['MCMC'], ['Youngs Modulus', 'Youngs Modulus','Poissons Ratio', 'Poissons Ratio'], ini, inp['range'], fig5, ax5)
    print('----------------------------------------------')
    print('Preconditioned Crank-Nicolson')
    print('----------------------------------------------')

elif inp['Method'] == 5:
    #The EnKF algorithm 
    B = EnKF_mcmc(inp)
    results = B.EnKF_go()
    if config['Print Chain'] == 1:
        print(results['MCMC'])
    fig5, ax5 = plt.subplots(4, 1)
    utilities.histogram(results['MCMC'], ['Youngs Modulus', 'Youngs Modulus','Poissons Ratio', 'Poissons Ratio'], ini, inp['range'], fig5, ax5)
    print('----------------------------------------------')
    print('Ensemble Kalman Filter')
    print('----------------------------------------------')
   
print('Acceptance Rate: %.3f' % results['accepted'])
print('Number of Samples: %.0f' % config['Number of samples'])
print('The median of the Youngs Modulus 1 posterior is: %f, with uncertainty +/- %.5f' % (np.median(results['MCMC'][0]), np.sqrt(np.var(results['MCMC'][0]))))
print('The median of the Youngs Modulus 2 posterior is: %f, with uncertainty +/- %.5f' % (np.median(results['MCMC'][1]), np.sqrt(np.var(results['MCMC'][1]))))
print('The median of the Poissons Ratio 1 posterior is: %f, with uncertainty +/- %.5f' % (np.median(results['MCMC'][2]), np.sqrt(np.var(results['MCMC'][2]))))
print('The median of the Poissons Ratio 2 posterior is: %f, with uncertainty +/- %.5f' % (np.median(results['MCMC'][3]), np.sqrt(np.var(results['MCMC'][3]))))
print()

# BURN = 1000

# print('Acceptance Rate: %.3f' % results['accepted'])
# print('Number of Samples: %.0f' % config['Number of samples'])
# print('The median of the Youngs Modulus 1 posterior is: %f, with uncertainty +/- %.5f' % (np.median(results['MCMC'][0][BURN:]), np.sqrt(np.var(results['MCMC'][0][BURN:]))))
# print('The median of the Youngs Modulus 2 posterior is: %f, with uncertainty +/- %.5f' % (np.median(results['MCMC'][1][BURN:]), np.sqrt(np.var(results['MCMC'][1][BURN:]))))
# print('The median of the Poissons Ratio 1 posterior is: %f, with uncertainty +/- %.5f' % (np.median(results['MCMC'][2][BURN:]), np.sqrt(np.var(results['MCMC'][2][BURN:]))))
# print('The median of the Poissons Ratio 2 posterior is: %f, with uncertainty +/- %.5f' % (np.median(results['MCMC'][3][BURN:]), np.sqrt(np.var(results['MCMC'][3][BURN:]))))
# print()

# fig6, ax6 = plt.subplots(4, 1)
# utilities.histogram(results['MCMC'][:,BURN:], ['Youngs Modulus', 'Youngs Modulus','Poissons Ratio', 'Poissons Ratio'], ini, inp['range'], fig6, ax6)
    
my_mesh = Mesh(inp['mesh'])
my_mesh.displacement([np.median(results['MCMC'][0]),np.median(results['MCMC'][1])], [np.median(results['MCMC'][2]), np.median(results['MCMC'][3])])
my_mesh.deformation_plot(label = f'Estimated Deformation, E1: %.3f, v1: %.3f, E2: %.3f, v2: %.3f ' % (np.median(results['MCMC'][0]), np.median(results['MCMC'][2]), np.median(results['MCMC'][1]),np.median(results['MCMC'][3])), ls =(0,(3,5)),colour = 'rebeccapurple', ch = 0.9, ax = ax1, lines = lines)
fig3, ax3 = plt.subplots(2, 1)
my_mesh.contour_plot('Estimated', fig3, ax3)

fig4, ax4 = plt.subplots(2, 1)
my_mesh.error_plot(true_displacement, fig4, ax4)

ax1.set_title('Deformation Plot', fontsize = 25)
fig.legend(loc = 'lower center', ncols=2)

plt.show()