from config import config
import numpy as np
import utilities
from MH import MH_mcmc
from EnKF import EnKF_mcmc
from DRAM import DRAM_algorithm
from AMH import AMH_mcmc
from MH_DR import MH_DR_mcmc
from SEnKF import S_EnKF_mcmc
import matplotlib.pyplot as plt
import seaborn as sns
import time
sns.set_context('talk')

# increase the precision
np.set_printoptions(precision=16)
from decimal import Decimal, getcontext 
getcontext().prec = 16

 # Figure width in inches, approximately A4-width - 2*1.25in margin
plt.rcParams.update({    # 4:3 aspect ratio
    'font.size' : 13,                   # Set font size to 11pt
    'axes.labelsize': 15,               # -> axis labels
    'legend.fontsize': 15,              # -> legends
    'font.family': 'charter',
    'text.usetex': True,
    'text.latex.preamble': (            # LaTeX preamble
        r'\usepackage[T1]{fontenc}'
    ),
    "font.weight": 'bold',
    "axes.labelweight": 'bold',
    'axes.titlesize' : 15
})


inp = {}

# data = np.load('RVE_EnKF2D.npy', allow_pickle=True)
# data = data.tolist()

# range of the parameters based on the prior density
inp['range']=np.array([[config['Imposed Limits']['Youngs Modulus'][0], config['Imposed Limits']['Poissons Ratio'][0], config['Imposed Limits']['Yield Stress'][0]], 
                       [config['Imposed Limits']['Youngs Modulus'][1], config['Imposed Limits']['Poissons Ratio'][1], config['Imposed Limits']['Yield Stress'][1]]])    
inp['s'] = config['s']                      

# number of iteration in MCMC
inp['nsamples']=config['Number of samples']

# initial covariance                          
inp['icov']=np.array([[config['Initial Variance']['Youngs Modulus'], 0, 0],[0, config['Initial Variance']['Poissons Ratio'], 0],
                      [0, 0, config['Initial Variance']['Yield Stress']]])                                   

# initial guesss of the parameters based on the prior
inp['theta0']=np.array([[config['Initial Material Parameters']['Youngs Modulus']],[config['Initial Material Parameters']['Poissons Ratio']],
                        [config['Initial Material Parameters']['Yield Stress']]])  

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
               config['Mesh grid']['Element ID'],
               config['Mesh grid']['thickness']]

ini = np.array([[config['True Material Parameters']['Youngs Modulus']], [config['True Material Parameters']['Poissons Ratio']], [config['True Material Parameters']['Yield Stress']]])

measurements=utilities.forward_model(ini, inp['mesh'])

Noise = np.zeros_like(measurements)

Noise[0::2] = np.random.normal(0, config["Measurement Noise"]*np.sqrt(np.var(measurements[0::2])), size = np.shape(measurements[0::2]))
Noise[1::2] = np.random.normal(0, config["Measurement Noise"]*np.sqrt(np.var(measurements[1::2])), size = np.shape(measurements[1::2]))
#inp['measurement'] = np.array(config['Measurements'])
inp['measurement'] = measurements + Noise

#print(inp['measurement'])
figd, axd = plt.subplots()
a,b = [], []
for i in range(len(inp['measurement'])//2):
    a.append(inp['measurement'][2*i]/1000)
    b.append(inp['measurement'][2*i + 1]/1000)

xx,yy = [], []
for i in range(len(measurements)//2):
    xx.append(measurements[2*i]/1000)
    yy.append(measurements[2*i + 1]/1000)

axd.scatter(a, b, label = 'Noisy Data', s = 40, linewidth = 1.5, marker = 'x', color = 'black', zorder = 10)
axd.plot(xx, yy, label = 'True Data', linewidth = 2)
axd.legend()
plt.grid()
# plt.xlabel('Displacement (mm)')
# plt.ylabel('External Force (N)')
# plt.show()

inp['Method'] = config['Methods']['Choosen Method']
#inp['theta0'] = np.array([np.random.choice(range(int(inp['range'][0][0]*1e6), int(inp['range'][1][0]*1e6)), 1)/1e6, np.random.choice(range(int(inp['range'][0][1]*1e6), int(inp['range'][1][1]*1e6)), 1)/1e6])

print()
print()
print('True Youngs Modulus: %.3f' % config['True Material Parameters']['Youngs Modulus'])
print('True Poissons Ratio: %.3f' % config['True Material Parameters']['Poissons Ratio'])
print('True Yield Stress: %.3f' % config['True Material Parameters']['Yield Stress'])
#print('Standard Deviation of Noise on Measurement Data: %f' %(config['Measurement Noise']))
print()
st = time.perf_counter()
if inp['Method'] == 0:
    # The Metropolis-Hastings technique
    C = MH_mcmc(inp)
    results = C.MH_go()
    print(f'End Time: {time.perf_counter() - st}')
    if config['Print Chain'] == 1:
        print(results['MCMC'])
    fig5, ax5 = plt.subplots(2, 1)
    #utilities.histogram(results['MCMC'], 100, ['Youngs Modulus', 'Poissons Ratio'], ini, inp['range'], fig5, ax5)
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
    fig5, ax5 = plt.subplots(2, 1)
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
    fig5, ax5 = plt.subplots(2, 1)
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
    fig5, ax5 = plt.subplots(2, 1)
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

elif inp['Method'] == 5:
    #The Baby algorithm 
    B = S_EnKF_mcmc(inp)
    results = B.S_EnKF_go()
    print(f'End Time: {time.perf_counter() - st}')
    if config['Print Chain'] == 1:
        print(results['MCMC'])
    fig5, ax5 = plt.subplots(2, 1)
    #utilities.histogram(results['MCMC'], 100, ['Youngs Modulus', 'Youngs Modulus','Poissons Ratio', 'Poissons Ratio'], ini, inp['range'], fig5, ax5)
    print('----------------------------------------------')
    print('sEnKF')
    print('----------------------------------------------')
   
print('Acceptance Rate: %.3f' % results['accepted'])
print('Number of Samples: %.0f' % config['Number of samples'])
print('The median of the Youngs Modulus posterior is: %f, with uncertainty +/- %.5f' % (np.median(results['MCMC'][0]), np.sqrt(np.var(results['MCMC'][0]))))
print('The median of the Poissons Ratio posterior is: %f, with uncertainty +/- %.5f' % (np.median(results['MCMC'][1]), np.sqrt(np.var(results['MCMC'][1]))))
print('The median of the Yield Stress posterior is: %f, with uncertainty +/- %.5f' % (np.median(results['MCMC'][2]), np.sqrt(np.var(results['MCMC'][2]))))
print()

outi =np.array([[np.median(results['MCMC'][0])], [np.median(results['MCMC'][1])], [np.median(results['MCMC'][2])]])

measurements2=utilities.forward_model(outi, inp['mesh'])
print(f'RMSE: {np.linalg.norm(measurements2/1000 - measurements/1000)/len(measurements)}')

X,Y = [], []
for i in range(len(measurements2)//2):
    X.append(measurements2[2*i])
    Y.append(measurements2[2*i + 1])
axd.plot(X, Y, label = 'MCMC result')
axd.legend()

figt, axt = plt.subplots(4,1)
axt[0].plot(range(len(results['MCMC'][0])), results['MCMC'][0])
axt[1].plot(range(len(results['MCMC'][1])), results['MCMC'][1])
axt[2].plot(range(len(results['MCMC'][2])), results['MCMC'][2])

plt.show()

