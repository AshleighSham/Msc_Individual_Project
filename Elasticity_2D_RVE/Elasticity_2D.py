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
import seaborn as sns
import time
import matplotlib.pyplot as plt
sns.set_context('talk')

# data = np.load('RVE_EnKF.npy', allow_pickle=True)
# data = data.tolist()
# print(len(data['Median'][0])+1)

inp = {}

# range of the parameters based on the prior density
minis = np.array([np.array(config['Imposed Limits']['Youngs Modulus'])[:, 0],
                  np.array(config['Imposed Limits']['Poissons Ratio'])[:, 0]]).flatten()
maxis = np.array([np.array(config['Imposed Limits']['Youngs Modulus'])[:, 1],
                  np.array(config['Imposed Limits']['Poissons Ratio'])[:, 1]]).flatten()
inp['range'] = np.array([minis, maxis])

# number of iteration in MCMC
inp['nsamples'] = config['Number of samples']

# initial covariance
icov = [config['Initial Variance']['Youngs Modulus'], config['Initial Variance']['Youngs Modulus'],
        config['Initial Variance']['Poissons Ratio'], config['Initial Variance']['Poissons Ratio']]
inp['icov'] = np.eye(config['Number of Materials']*2)*np.array(icov)
inp['icov'] = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1e-3, 0], [0, 0, 0, 1e-3]])

inp['s'] = config['s']

# initial guesss of the parameters based on the prior
itheta = [[config['Initial Material Parameters']['Youngs Modulus'][0]],
          [config['Initial Material Parameters']['Youngs Modulus'][1]],
          [config['Initial Material Parameters']['Poissons Ratio'][0]],
          [config['Initial Material Parameters']['Poissons Ratio'][1]]]

inp['theta0'] = np.array(itheta)

# std                               
inp['sigma'] = config["Standard Deviation"]

# The starting point of the Kalman MCMC
inp['Kalmans'] = config['Starting Kalman point']

# assumed measurement error for Kalman MCMC
inp['me'] = config['Measurement error for Kalman']

# assumed error for Kalman MCMC
inp['ie'] = config['Innovation error for Kalman'] 

# adaption setp size
inp['adapt'] = config['Adaption Step Size']
 
# mesh set up
inp['mesh'] = [config['Mesh grid']['quad'],
               config['Mesh grid']['sf'],
               config['Mesh grid']['Nodal Coordinates'],
               config['Mesh grid']['Element Node Numbers'],
               config['Mesh grid']['Number of elements'],
               config['Mesh grid']['Force Magnitude'],
               config['Mesh grid']['Force Nodes'],
               config['Mesh grid']['Fixed Nodes'],
               config['Mesh grid']['Element ID'],
               config['Mesh grid']['thickness']]

ini = [config['True Material Parameters']['Youngs Modulus'][0],
       config['True Material Parameters']['Youngs Modulus'][1],
       config['True Material Parameters']['Poissons Ratio'][0],
       config['True Material Parameters']['Poissons Ratio'][1]]

measurements = utilities.forward_model(np.array(ini), inp['mesh'])
measurements1 = measurements + np.random.normal(0, config['Measurement Noise'],
                                                size=np.shape(measurements))
# measurements1[[0,1,8,9,16,17]] = measurements[[0,1,8,9,16,17]]
inp['measurement'] = measurements1
# print(measurements1)
# print(measurements)

# figd, axd = plt.subplots()

# axd.scatter(range(len(measurements)),measurements, s =10)
# axd.scatter(range(len(measurements1)), measurements1, s= 10)

# plt.show()
 
lines = []
my_mesh = Mesh(inp['mesh'])

fig11, axl1 = plt.subplots()
plt.subplots_adjust(bottom=0.05)

true_displacement = my_mesh.displacement(config['True Material Parameters']['Youngs Modulus'],
                                         config['True Material Parameters']['Poissons Ratio'])
# my_mesh.deformation_plot(label = 'a', colour= 'black', ch = 1, ax = axl1, lines = lines,  D = measurements/1000, ls = 'solid', non=False)
# my_mesh.deformation_plot(label = f'Noisy Deformation', colour= 'palevioletred', ch = 0.95, ax = axl1, lines = lines, ls = (0,(5,5)), D = measurements1/1000, non=False)

# plt.show()
# fig2, ax2 = plt.subplots(2, 1)
# my_mesh.contour_plot('True', fig2, ax2)

inp['Method'] = config['Methods']['Choosen Method']


# inp['theta0'] = np.array([np.random.choice(range(int(minis[0]*1e6), int(maxis[0]*1e6)), 1)/1e6, np.random.choice(range(int(minis[1]*1e6), int(maxis[1]*1e6)), 1)/1e6, np.random.choice(range(int(minis[2]*1e6), int(maxis[2]*1e6)), 1)/1e6, np.random.choice(range(int(minis[3]*1e6), int(maxis[3]*1e6)), 1)/1e6])

print()
print()
for i in range(config['Number of Materials']):
    print('True Youngs Modulus %.0f: %.3f' % (i, config['True Material Parameters']['Youngs Modulus'][i]))
    print('True Poissons Ratio %.0f: %.3f' % (i, config['True Material Parameters']['Poissons Ratio'][i]))
print('Standard Deviation of Noise on Measurement Data: %.10f' %(config['Measurement Noise']))
print()
st = time.perf_counter()
if inp['Method'] == 0:
    # The Metropolis-Hastings technique
    C = MH_mcmc(inp)
    results = C.MH_go()
    print(f'End Time: {time.perf_counter() - st}')
    if config['Print Chain'] == 1:
        print(results['MCMC'])
    # utilities.histogram(results['MCMC'], ['Youngs Modulus', 'Youngs Modulus','Poissons Ratio', 'Poissons Ratio'], ini, inp['range'])
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
    # utilities.histogram(results['MCMC'], ['Youngs Modulus', 'Youngs Modulus','Poissons Ratio', 'Poissons Ratio'], ini, inp['range'])
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
    # utilities.histogram(results['MCMC'],['Youngs Modulus', 'Youngs Modulus','Poissons Ratio', 'Poissons Ratio'], ini, inp['range'])
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
    # utilities.histogram(results['MCMC'], ['Youngs Modulus', 'Youngs Modulus','Poissons Ratio', 'Poissons Ratio'], ini, inp['range'])
    print('----------------------------------------------')
    print('Delayed Rejection Adaptive Metropolis Hastings')
    print('----------------------------------------------')

elif inp['Method'] == 4:
    # The EnKF algorithm 
    B = EnKF_mcmc(inp)
    results = B.EnKF_go()
    print(f'End Time: {time.perf_counter() - st}')
    if config['Print Chain'] == 1:
        print(results['MCMC'])
    # utilities.histogram(results['MCMC'], ['Youngs Modulus', 'Youngs Modulus','Poissons Ratio', 'Poissons Ratio'], ini, inp['range'])
    # print('----------------------------------------------')
    # print('Ensemble Kalman Filter')
    # print('----------------------------------------------')
   
elif inp['Method'] == 5:
    # The Baby algorithm 
    B = S_EnKF_mcmc(inp)
    results = B.S_EnKF_go()
    print(f'End Time: {time.perf_counter() - st}')
    if config['Print Chain'] == 1:
        print(results['MCMC'])
    # utilities.histogram(results['MCMC'], ['Youngs Modulus', 'Youngs Modulus','Poissons Ratio', 'Poissons Ratio'], ini, inp['range'])
    print('----------------------------------------------')
    print('Stochastic Adapt EnKF')
    print('----------------------------------------------')

print('Acceptance Rate: %.3f' % results['accepted'])
print('Number of Samples: %.0f' % config['Number of samples'])
print('The median of the Youngs Modulus 1 posterior is: %f, with uncertainty +/- %.5f' % (np.median(results['MCMC'][0]), np.sqrt(np.var(results['MCMC'][0]))))
print('The median of the Youngs Modulus 2 posterior is: %f, with uncertainty +/- %.5f' % (np.median(results['MCMC'][1]), np.sqrt(np.var(results['MCMC'][1]))))
print('The median of the Poissons Ratio 1 posterior is: %f, with uncertainty +/- %.5f' % (np.median(results['MCMC'][2]), np.sqrt(np.var(results['MCMC'][2]))))
print('The median of the Poissons Ratio 2 posterior is: %f, with uncertainty +/- %.5f' % (np.median(results['MCMC'][3]), np.sqrt(np.var(results['MCMC'][3]))))
print()

outi = np.array([[np.median(results['MCMC'][0])],
                 [np.median(results['MCMC'][1])],
                 [np.median(results['MCMC'][2])],
                 [np.median(results['MCMC'][3])]])

res = utilities.forward_model(outi, inp['mesh'])

print(f'RMSE: {np.linalg.norm(measurements/1000 - res/1000)/len(measurements)}')


fig, ax = plt.subplots(4, 1)
ax[0].plot(range(len(results['MCMC'][0])), results['MCMC'][0])
ax[0].axhline(10, color='black')
ax[1].plot(range(len(results['MCMC'][1])), results['MCMC'][1])
ax[1].axhline(1 , color='black')
ax[2].plot(range(len(results['MCMC'][2])), results['MCMC'][2])
ax[2].axhline(0.3, color='black')
ax[3].plot(range(len(results['MCMC'][3])), results['MCMC'][3])
ax[3].axhline(0.3, color='black')
res = my_mesh.displacement([np.median(results['MCMC'][0]),
                            np.median(results['MCMC'][1])], 
                           [np.median(results['MCMC'][2]),
                            np.median(results['MCMC'][3])])

my_mesh.deformation_plot(label=f'Noisy Deformation', colour='palevioletred',
                         ch=0.95, ax=axl1, lines=lines, ls=(0,(5,5)), D=res/1000, non=False)
plt.show()

lags = range(0, 100, 1)
print(f'{utilities.effective_sample_size_ratio(results['MCMC'][0], lags)}')
print(f'{utilities.effective_sample_size_ratio(results['MCMC'][1], lags)}')
print(f'{utilities.effective_sample_size_ratio(results['MCMC'][2], lags)}')
print(f'{utilities.effective_sample_size_ratio(results['MCMC'][3], lags)}')
