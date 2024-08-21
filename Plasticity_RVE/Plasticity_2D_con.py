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
import pandas as pd

past_chain_info = np.load('MH.npy',allow_pickle='TRUE').item()
#DATA = {'thetaj': dumy_theta, 'oldvalue': dumy_values, 'past_cov': dumy_cov}

inp = {}

inp['Method'] = config['Methods']['Choosen Method']

# range of the parameters based on the prior density
inp['range']=np.array([[config['Imposed Limits']['Youngs Modulus'][0], config['Imposed Limits']['Poissons Ratio'][0], config['Imposed Limits']['Yield Stress'][0], config['Imposed Limits']['Hardening Modulus'][0]], 
                       [config['Imposed Limits']['Youngs Modulus'][1], config['Imposed Limits']['Poissons Ratio'][1], config['Imposed Limits']['Yield Stress'][1], config['Imposed Limits']['Hardening Modulus'][1]]])    

#pcn step size
inp['s'] = config['s']                      

# number of iteration in MCMC
inp['nsamples']=config['Number of samples']

# The starting point of the Kalman MCMC           
inp['Kalmans']= config['Starting Kalman point']                        

# initial covariance                          
inp['icov']=np.array([[config['Initial Variance']['Youngs Modulus'], 0, 0, 0],[0, config['Initial Variance']['Poissons Ratio'], 0, 0],
                      [0, 0, config['Initial Variance']['Yield Stress'], 0], [0, 0, 0, config['Initial Variance']['Hardening Modulus']]])    

#adaptive covariance
if inp['Method'] == 1 or inp['Method'] == 3:
    inp['icov'] = past_chain_info['past_cov'] 

if inp['Method'] == 5 and len(past_chain_info['thetaj'][0]) < inp['Kalmans']:
    inp['icov'] = past_chain_info['past_cov']    

#continued theta
inp['theta0'] = past_chain_info['thetaj'][:,-1].reshape(4, 1)

# std                               
inp['sigma']=config["Standard Deviation"]

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

#measurements=utilities.forward_model(ini, inp['mesh'])
#measurements1 = measurements + np.random.normal(0, config['Measurement Noise']*config['Noise scale'], size = [np.size(measurements, 0), np.size(measurements, 1)])
inp['measurement'] = config['Mesurements']

print()
print()
print('True Youngs Modulus: %.3f' % config['True Material Parameters']['Youngs Modulus'])
print('True Poissons Ratio: %.3f' % config['True Material Parameters']['Poissons Ratio'])
print('True Yield Stress: %.3f' % config['True Material Parameters']['Yield Stress'])
print('True Hardening Modulus: %.3f' % config['True Material Parameters']['Hardening Modulus'])
print('Standard Deviation of Noise on Measurement Data: %f' %(config['Measurement Noise']))
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

# measurements=utilities.forward_model(outi, inp['mesh'])
# X,Y = [], []
# for i in range(len(measurements)//2):
#     X.append(measurements[2*i])
#     Y.append(measurements[2*i + 1])
# axd.plot(X, Y, label = 'MCMC result')
# axd.legend()


print(len(past_chain_info['thetaj'][0]) + len(results['MCMC'][0]))
figt, axt = plt.subplots(4,1)

axt[0].plot(range(len(past_chain_info['thetaj'][0])), past_chain_info['thetaj'][0])
axt[1].plot(range(len(past_chain_info['thetaj'][0])), past_chain_info['thetaj'][1])
axt[2].plot(range(len(past_chain_info['thetaj'][0])), past_chain_info['thetaj'][2])
axt[3].plot(range(len(past_chain_info['thetaj'][0])), past_chain_info['thetaj'][3])

axt[0].plot(range(inp['nsamples']), results['MCMC'][0])
axt[1].plot(range(inp['nsamples']), results['MCMC'][1])
axt[2].plot(range(inp['nsamples']), results['MCMC'][2])
axt[3].plot(range(inp['nsamples']), results['MCMC'][3])

plt.show()

