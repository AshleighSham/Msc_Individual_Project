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

#data = np.load('RVE_EnKF.npy', allow_pickle=True)
#data = data.tolist()

fig, ax1 = plt.subplots()
plt.subplots_adjust(bottom = 0.1)

inp = {}

# range of the parameters based on the prior density
minis = np.array([np.array(config['Imposed Limits']['Youngs Modulus'])[:,0], np.array(config['Imposed Limits']['Poissons Ratio'])[:,0]]).flatten()
maxis = np.array([np.array(config['Imposed Limits']['Youngs Modulus'])[:,1], np.array(config['Imposed Limits']['Poissons Ratio'])[:,1]]).flatten()
inp['range']=np.array([minis, maxis])                      

# number of iteration in MCMC
inp['nsamples']=config['Number of samples']

# initial covariance  
icov = [config['Initial Variance']['Youngs Modulus'], config['Initial Variance']['Youngs Modulus'], config['Initial Variance']['Poissons Ratio'], config['Initial Variance']['Poissons Ratio']]
inp['icov'] = np.eye(config['Number of Materials']*2)*np.array(icov)
inp['icov'] = np.array([[1,0,0,0],[0,1,0,0],[0,0,1e-3,0],[0,0,0,1e-3]])

inp['s'] = config['s']

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

# fig2, ax2 = plt.subplots()

# ax2.plot(measurements, '.')
# ax2.plot(measurements1, '.')

# plt.show()

lines = []
my_mesh = Mesh(inp['mesh'])
true_displacement = my_mesh.displacement(config['True Material Parameters']['Youngs Modulus'], config['True Material Parameters']['Poissons Ratio'])
my_mesh.deformation_plot(label = f'True Deformation, E1: %.3f, E2: %.3f, v1: %.3f, v2: %.3f' % (config['True Material Parameters']['Youngs Modulus'][0], config['True Material Parameters']['Youngs Modulus'][1], config['True Material Parameters']['Poissons Ratio'][0], config['True Material Parameters']['Poissons Ratio'][1]), 
                         colour= 'plum', ch = 1, ax = ax1, lines = lines, ls = 'solid')

fig2, ax2 = plt.subplots(2, 1)
my_mesh.contour_plot('True', fig2, ax2)

inp['Method'] = config['Methods']['Choosen Method']

inp['theta0'] = np.array([np.random.choice(range(int(minis[0]*1e6), int(maxis[0]*1e6)), 1)/1e6, np.random.choice(range(int(minis[1]*1e6), int(maxis[1]*1e6)), 1)/1e6, np.random.choice(range(int(minis[2]*1e6), int(maxis[2]*1e6)), 1)/1e6, np.random.choice(range(int(minis[3]*1e6), int(maxis[3]*1e6)), 1)/1e6])

print()
print()
for i in range(config['Number of Materials']):
    print('True Youngs Modulus %.0f: %.3f' % (i, config['True Material Parameters']['Youngs Modulus'][i]))
    print('True Poissons Ratio %.0f: %.3f' % (i, config['True Material Parameters']['Poissons Ratio'][i]))
print('Standard Deviation of Noise on Measurement Data: %.10f' %(config['Measurement Noise']*config['Mesh grid']['sf']))
print()
if inp['Method'] == 0:
    # The Metropolis-Hastings technique
    C = MH_mcmc(inp)
    results = C.MH_go()
    if config['Print Chain'] == 1:
        print(results['MCMC'])
    utilities.histogram(results['MCMC'], ['Youngs Modulus', 'Youngs Modulus','Poissons Ratio', 'Poissons Ratio'], ini, inp['range'])
    print('----------------------------------------------')
    print('Metropolis Hastings')
    print('----------------------------------------------')

elif inp['Method'] == 1:
    # The AMH algorithm 
    A = AMH_mcmc(inp)
    results = A.AMH_go()
    if config['Print Chain'] == 1:
        print(results['MCMC'])
    utilities.histogram(results['MCMC'], ['Youngs Modulus', 'Youngs Modulus','Poissons Ratio', 'Poissons Ratio'], ini, inp['range'])
    print('----------------------------------------------')
    print('Adaptive Metropolis Hastings')
    print('----------------------------------------------')

elif inp['Method'] == 2:
    # The MH DR algorithm 
    A = MH_DR_mcmc(inp)
    results = A.MH_DR_go()
    if config['Print Chain'] == 1:
        print(results['MCMC'])
    utilities.histogram(results['MCMC'],['Youngs Modulus', 'Youngs Modulus','Poissons Ratio', 'Poissons Ratio'], ini, inp['range'])
    print('----------------------------------------------')
    print('Metropolis Hastings Delayed Rejection')
    print('----------------------------------------------')

elif inp['Method'] == 3:
    # The DRAM algorithm 
    A = DRAM_algorithm(inp)
    results = A.DRAM_go()
    if config['Print Chain'] == 1:
        print(results['MCMC'])
    utilities.histogram(results['MCMC'], ['Youngs Modulus', 'Youngs Modulus','Poissons Ratio', 'Poissons Ratio'], ini, inp['range'])
    print('----------------------------------------------')
    print('Delayed Rejection Adaptive Metropolis Hastings')
    print('----------------------------------------------')

elif inp['Method'] == 4:
    #The EnKF algorithm 
    B = Crank_mcmc(inp)
    results = B.Crank_go()
    if config['Print Chain'] == 1:
        print(results['MCMC'])
    utilities.histogram(results['MCMC'], ['Youngs Modulus', 'Youngs Modulus','Poissons Ratio', 'Poissons Ratio'], ini, inp['range'])
    print('----------------------------------------------')
    print('Ensemble Kalman Filter')
    print('----------------------------------------------')

elif inp['Method'] == 5:
    #The EnKF algorithm 
    B = EnKF_mcmc(inp)
    results = B.EnKF_go()
    if config['Print Chain'] == 1:
        print(results['MCMC'])
    utilities.histogram(results['MCMC'], ['Youngs Modulus', 'Youngs Modulus','Poissons Ratio', 'Poissons Ratio'], ini, inp['range'])
    print('----------------------------------------------')
    print('Ensemble Kalman Filter')
    print('----------------------------------------------')
   
elif inp['Method'] == 6:
    #The Baby algorithm 
    B = Baby_mcmc(inp)
    results = B.Baby_go()
    if config['Print Chain'] == 1:
        print(results['MCMC'])
    utilities.histogram(results['MCMC'], ['Youngs Modulus', 'Youngs Modulus','Poissons Ratio', 'Poissons Ratio'], ini, inp['range'])
    print('----------------------------------------------')
    print('Baby')
    print('----------------------------------------------')

print('Acceptance Rate: %.3f' % results['accepted'])
print('Number of Samples: %.0f' % config['Number of samples'])
print('The median of the Youngs Modulus 1 posterior is: %f, with uncertainty +/- %.5f' % (np.median(results['MCMC'][0]), np.sqrt(np.var(results['MCMC'][0]))))
print('The median of the Youngs Modulus 2 posterior is: %f, with uncertainty +/- %.5f' % (np.median(results['MCMC'][1]), np.sqrt(np.var(results['MCMC'][1]))))
print('The median of the Poissons Ratio 1 posterior is: %f, with uncertainty +/- %.5f' % (np.median(results['MCMC'][2]), np.sqrt(np.var(results['MCMC'][2]))))
print('The median of the Poissons Ratio 2 posterior is: %f, with uncertainty +/- %.5f' % (np.median(results['MCMC'][3]), np.sqrt(np.var(results['MCMC'][3]))))
print()
my_mesh = Mesh(inp['mesh'])
my_mesh.displacement([np.median(results['MCMC'][0]),np.median(results['MCMC'][1])], [np.median(results['MCMC'][2]), np.median(results['MCMC'][3])])
my_mesh.deformation_plot(label = f'Estimated Deformation, E1: %.3f, v1: %.3f, E2: %.3f, v2: %.3f ' % (np.median(results['MCMC'][0]), np.median(results['MCMC'][2]), np.median(results['MCMC'][1]),np.median(results['MCMC'][3])), ls =(0,(3,5)),colour = 'rebeccapurple', ch = 0.9, ax = ax1, lines = lines)
fig3, ax3 = plt.subplots(2, 1)
my_mesh.contour_plot('Estimated', fig3, ax3)

fig4, ax4 = plt.subplots(2, 1)
my_mesh.error_plot(true_displacement, fig4, ax4)

ax1.set_title('Deformation Plot', fontsize = 25)
fig.legend(loc = 'lower center', ncols=2)

#data = {'Initial':{0:[], 1:[], 2:[], 3:[]}, 'Median':{0:[], 1:[], 2:[], 3:[]}, 'Uncertainty':{0:[], 1:[], 2:[], 3:[]}}
# for i in range(4):
#     data['Initial'][i].append(inp['theta0'][i][0])

# data['Median'][0].append(np.median(results['MCMC'][0]))
# data['Median'][1].append(np.median(results['MCMC'][1]))
# data['Median'][2].append(np.median(results['MCMC'][2]))
# data['Median'][3].append(np.median(results['MCMC'][3]))

# data['Uncertainty'][0].append(np.sqrt(np.var(results['MCMC'][0])))
# data['Uncertainty'][1].append(np.sqrt(np.var(results['MCMC'][1])))
# data['Uncertainty'][2].append(np.sqrt(np.var(results['MCMC'][2])))
# data['Uncertainty'][3].append(np.sqrt(np.var(results['MCMC'][3])))

# np.save('RVE_EnKF.npy', data, allow_pickle=True) 