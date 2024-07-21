from config import config
import numpy as np
import utilities
from MH import MH_mcmc
from EnKF import EnKF_mcmc
from DRAM import DRAM_algorithm
from AMH import AMH_mcmc
from MH_DR import MH_DR_mcmc
from crank import Crank_mcmc
from baby import Baby_mcmc
import matplotlib.pyplot as plt


inp = {}

# data = np.load('RVE_EnKF2D.npy', allow_pickle=True)
# data = data.tolist()

# range of the parameters based on the prior density
inp['range']=np.array([[config['Imposed Limits']['Youngs Modulus'][0], config['Imposed Limits']['Poissons Ratio'][0], config['Imposed Limits']['Yield Stress'][0], config['Imposed Limits']['Hardening Modulus'][0]], 
                       [config['Imposed Limits']['Youngs Modulus'][1], config['Imposed Limits']['Poissons Ratio'][1], config['Imposed Limits']['Yield Stress'][1], config['Imposed Limits']['Hardening Modulus'][1]]])    
inp['s'] = config['s']                      

# number of iteration in MCMC
inp['nsamples']=config['Number of samples']

# initial covariance                          
inp['icov']=np.array([[config['Initial Variance']['Youngs Modulus'], 0, 0, 0],[0, config['Initial Variance']['Poissons Ratio'], 0, 0],
                      [0, 0, config['Initial Variance']['Yield Stress'], 0], [0, 0, 0, config['Initial Variance']['Hardening Modulus']]])                                   

# initial guesss of the parameters based on the prior
inp['theta0']=np.array([[config['Initial Material Parameters']['Youngs Modulus']],[config['Initial Material Parameters']['Poissons Ratio']],
                        [config['Initial Material Parameters']['Yield Stress']], [config['Initial Material Parameters']['Hardening Modulus']]])  

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

ini = np.array([[config['True Material Parameters']['Youngs Modulus']], [config['True Material Parameters']['Poissons Ratio']], [config['True Material Parameters']['Yield Stress']],[config['True Material Parameters']['Hardening Modulus']]])

measurements=utilities.forward_model(ini, inp['mesh'])

Noise = np.zeros_like(measurements)

Noise[0::2] = np.random.normal(0, config["Measurement Noise"]*np.var(measurements[0::2]), size = np.shape(measurements[0::2]))
Noise[1::2] = np.random.normal(0, config["Measurement Noise"]*np.var(measurements[1::2]), size = np.shape(measurements[1::2]))
#inp['measurement'] = np.array(config['Measurements'])
inp['measurement'] = measurements + Noise

#print(inp['measurement'])
figd, axd = plt.subplots()
a,b = [], []
for i in range(len(inp['measurement'])//2):
    a.append(inp['measurement'][2*i])
    b.append(inp['measurement'][2*i + 1])

xx,yy = [], []
for i in range(len(measurements)//2):
    xx.append(measurements[2*i])
    yy.append(measurements[2*i + 1])

axd.scatter(a, b, label = 'Noisy Data', marker = 'x')
axd.plot(xx, yy, label = 'True Data')

inp['Method'] = config['Methods']['Choosen Method']
#inp['theta0'] = np.array([np.random.choice(range(int(inp['range'][0][0]*1e6), int(inp['range'][1][0]*1e6)), 1)/1e6, np.random.choice(range(int(inp['range'][0][1]*1e6), int(inp['range'][1][1]*1e6)), 1)/1e6])

print()
print()
print('True Youngs Modulus: %.3f' % config['True Material Parameters']['Youngs Modulus'])
print('True Poissons Ratio: %.3f' % config['True Material Parameters']['Poissons Ratio'])
print('True Yield Stress: %.3f' % config['True Material Parameters']['Yield Stress'])
print('True Hardening Modulus: %.3f' % config['True Material Parameters']['Hardening Modulus'])
#print('Standard Deviation of Noise on Measurement Data: %f' %(config['Measurement Noise']))
print()
if inp['Method'] == 0:
    # The Metropolis-Hastings technique
    C = MH_mcmc(inp)
    results = C.MH_go()
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
    if config['Print Chain'] == 1:
        print(results['MCMC'])
    fig5, ax5 = plt.subplots(2, 1)
    #utilities.histogram(results['MCMC'], 100, ['Youngs Modulus', 'Youngs Modulus','Poissons Ratio', 'Poissons Ratio'], ini, inp['range'], fig5, ax5)
    print('----------------------------------------------')
    print('Delayed Rejection Adaptive Metropolis Hastings')
    print('----------------------------------------------')

elif inp['Method'] == 4:
    #The Crank algorithm 
    B = Crank_mcmc(inp)
    results = B.Crank_go()
    if config['Print Chain'] == 1:
        print(results['MCMC'])
    fig5, ax5 = plt.subplots(2, 1)
    #utilities.histogram(results['MCMC'], 100, ['Youngs Modulus', 'Youngs Modulus','Poissons Ratio', 'Poissons Ratio'], ini, inp['range'], fig5, ax5)
    print('----------------------------------------------')
    print('Preconditioned Crank-Nicolson')
    print('----------------------------------------------')

elif inp['Method'] == 5:
    #The EnKF algorithm 
    B = EnKF_mcmc(inp)
    results = B.EnKF_go()
    if config['Print Chain'] == 1:
        print(results['MCMC'])
    #fig5, ax5 = plt.subplots(2, 1)
    #utilities.histogram(results['MCMC'], 100, ['Youngs Modulus', 'Youngs Modulus','Poissons Ratio', 'Poissons Ratio'], ini, inp['range'], fig5, ax5)
    print('----------------------------------------------')
    print('Ensemble Kalman Filter')
    print('----------------------------------------------')

elif inp['Method'] == 6:
    #The Baby algorithm 
    B = Baby_mcmc(inp)
    results = B.Baby_go()
    if config['Print Chain'] == 1:
        print(results['MCMC'])
    fig5, ax5 = plt.subplots(2, 1)
    #utilities.histogram(results['MCMC'], 100, ['Youngs Modulus', 'Youngs Modulus','Poissons Ratio', 'Poissons Ratio'], ini, inp['range'], fig5, ax5)
    print('----------------------------------------------')
    print('Baby')
    print('----------------------------------------------')
   
print('Acceptance Rate: %.3f' % results['accepted'])
print('Number of Samples: %.0f' % config['Number of samples'])
print('The median of the Youngs Modulus posterior is: %f, with uncertainty +/- %.5f' % (np.median(results['MCMC'][0]), np.sqrt(np.var(results['MCMC'][0]))))
print('The median of the Poissons Ratio posterior is: %f, with uncertainty +/- %.5f' % (np.median(results['MCMC'][1]), np.sqrt(np.var(results['MCMC'][1]))))
print('The median of the Yield Stress posterior is: %f, with uncertainty +/- %.5f' % (np.median(results['MCMC'][2]), np.sqrt(np.var(results['MCMC'][2]))))
print('The median of the Hardening Modulus posterior is: %f, with uncertainty +/- %.5f' % (np.median(results['MCMC'][3]), np.sqrt(np.var(results['MCMC'][3]))))
print()

outi =np.array([[np.median(results['MCMC'][0])], [np.median(results['MCMC'][1])], [np.median(results['MCMC'][2])], [np.median(results['MCMC'][3])]])

measurements=utilities.forward_model(outi, inp['mesh'])
X,Y = [], []
for i in range(len(measurements)//2):
    X.append(measurements[2*i])
    Y.append(measurements[2*i + 1])
axd.plot(X, Y, label = 'MCMC result')
axd.legend()

figt, axt = plt.subplots(4,1)
axt[0].plot(range(len(results['MCMC'][0])), results['MCMC'][0])
axt[1].plot(range(len(results['MCMC'][1])), results['MCMC'][1])
axt[2].plot(range(len(results['MCMC'][2])), results['MCMC'][2])
axt[3].plot(range(len(results['MCMC'][3])), results['MCMC'][3])

plt.show()

