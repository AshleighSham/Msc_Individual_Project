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
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()
plt.subplots_adjust(bottom = 0.1)

inp = {}

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
               config['Mesh grid']['Fixed Nodes']]

ini = [config['True Material Parameters']['Youngs Modulus'], config['True Material Parameters']['Poissons Ratio']]

measurements=utilities.forward_model(np.array([[config['True Material Parameters']['Youngs Modulus']],[config['True Material Parameters']['Poissons Ratio']]]), inp['mesh'])
measurements += np.random.normal(0, config['Measurement Noise']*config['Mesh grid']['sf'], size = [np.size(measurements, 0), np.size(measurements, 1)])
inp['measurement'] = measurements
# inp['measurement'] = [[-3.55564323e-04], [ 1.30177716e-04], [-2.99018843e-04], [-6.05993608e-05], [-5.93661063e-04], [ 1.07172160e-04], 
#                       [-4.25997740e-04], [-1.27344989e-04], [-5.58510424e-04], [-4.74301048e-04], [-5.73632799e-04], [-1.83340375e-04], 
#                       [-6.10444416e-04], [-3.06056584e-04], [-7.48990977e-04], [-1.45275771e-03], [-9.37630122e-04], [-1.23757180e-03], 
#                       [-7.18926761e-04], [-1.51544237e-03], [-1.19011278e-03], [-2.24539487e-03], [-1.01805782e-03], [-2.31804191e-03], 
#                       [-1.17199282e-03], [-2.50449342e-03], [-1.16996533e-03], [-2.84106073e-03], [-1.53811162e-03], [-3.28508626e-03], 
#                       [-1.63961483e-03], [-3.75535587e-03], [-1.27796104e-03], [-4.10986112e-03], [-1.42537979e-03], [-4.71832776e-03], 
#                       [-1.46452292e-03], [-5.34728404e-03], [-1.79579884e-03], [-5.83920097e-03], [-1.67205821e-03], [-6.12358746e-03], 
#                       [-1.38552256e-03], [-6.93056931e-03], [-1.62870834e-03], [-7.24702501e-03], [-1.63432209e-03], [-7.99394256e-03], 
#                       [-1.57463983e-03], [-8.35556535e-03], [-1.49247002e-03], [-9.18868552e-03], [-1.95368786e-03], [-9.31724457e-03], 
#                       [-1.96949545e-03], [-1.05187567e-02], [-1.63378275e-03], [-1.12567226e-02], [-1.45892800e-03], [-1.13834781e-02], 
#                       [-1.87208620e-03], [-1.18805369e-02], [ 2.52931302e-04], [-4.43824680e-04], [-1.81648989e-04], [-5.99511491e-05], 
#                       [-2.84527805e-04], [-4.98552650e-05], [-2.71037602e-04], [-6.40780991e-05], [-2.82829225e-04], [-2.76525499e-04], 
#                       [-6.32580141e-05], [-6.61810851e-04], [-2.34814606e-04], [-9.28921294e-04], [-7.02690261e-04], [-9.06083100e-04], 
#                       [-6.60550740e-04], [-1.38187243e-03], [ 9.23959439e-05], [-1.70697296e-03], [-8.28277738e-04], [-1.99408912e-03], 
#                       [-6.26101088e-04], [-2.20575205e-03], [-6.57416823e-04], [-2.73271219e-03], [-7.31262718e-04], [-3.14826377e-03], 
#                       [-1.07990409e-03], [-3.38955042e-03], [-8.89096523e-04], [-3.91772228e-03], [-1.28192187e-03], [-4.20552822e-03], 
#                       [-8.40734304e-04], [-5.07925586e-03], [-1.04980683e-03], [-4.91953074e-03], [-6.37357190e-04], [-6.36393017e-03], 
#                       [-9.75239431e-04], [-6.54894025e-03], [-8.07641656e-04], [-7.23509443e-03], [-7.95606637e-04], [-7.33311586e-03], 
#                       [-1.21030814e-03], [-8.21391718e-03], [-1.33690101e-03], [-8.19226567e-03], [-1.37784630e-03], [-9.23640726e-03], 
#                       [-1.51839267e-03], [-9.42310280e-03], [-1.25643758e-03], [-1.01494548e-02], [-1.28681908e-03], [-1.12820539e-02], 
#                       [-1.19891973e-03], [-1.15449273e-02], [-9.81969582e-04], [-1.21377485e-02], [-1.73118250e-04], [-9.62889791e-05], 
#                       [-1.83827970e-04], [-3.92845026e-05], [ 3.62850119e-04], [ 1.05967657e-04], [-3.18870911e-04], [-1.06744131e-05], 
#                       [ 1.44649919e-04], [-2.62900743e-04], [-4.20606244e-05], [-3.80736309e-04], [-2.23798435e-04], [-9.88603388e-04], 
#                       [-4.22968114e-04], [-9.96188155e-04], [-5.32586315e-04], [-1.34868354e-03], [-3.54632735e-04], [-1.69002202e-03], 
#                       [-3.82391767e-04], [-1.77457547e-03], [ 1.83493086e-04], [-2.23141026e-03], [-7.48168002e-04], [-2.40647173e-03], 
#                       [-2.00748231e-04], [-2.95327976e-03], [-2.30099149e-04], [-3.40386532e-03], [-7.23644911e-04], [-3.62234413e-03], 
#                       [-5.10249709e-04], [-4.59449363e-03], [-2.01708719e-04], [-5.18724786e-03], [-5.62033023e-04], [-5.28446797e-03], 
#                       [-4.80737298e-04], [-5.60077952e-03], [-4.04575079e-04], [-6.45532337e-03], [-5.81077614e-04], [-6.65197238e-03], 
#                       [-7.65497362e-04], [-7.51812737e-03], [-1.79670961e-04], [-7.79840463e-03], [-4.10658967e-04], [-8.57635262e-03], 
#                       [-6.87157393e-04], [-9.22697136e-03], [-5.23706520e-04], [-9.77388689e-03], [-5.92228459e-04], [-1.02825123e-02], 
#                       [-6.48146387e-04], [-1.07709674e-02], [-6.30968192e-04], [-1.14824742e-02], [-2.00595120e-04], [-1.18989668e-02], 
#                       [ 2.74916991e-04], [ 6.01588009e-05], [-1.64499573e-05], [ 2.13753354e-04], [-2.37619447e-04], [ 1.75395508e-04], 
#                       [ 1.79747564e-04], [-2.19452298e-04], [-4.80775246e-04], [-1.57081246e-04], [-1.89583464e-04], [-4.74832213e-04], 
#                       [-1.45087499e-04], [-7.98846515e-04], [ 1.52513522e-04], [-9.85133417e-04], [-1.11657554e-04], [-1.39990338e-03], 
#                       [ 1.41090418e-04], [-1.21948772e-03], [-3.29914872e-04], [-1.90949813e-03], [-1.50876006e-05], [-2.16231545e-03], 
#                       [-4.79137059e-04], [-2.76886827e-03], [ 3.68019357e-05], [-3.18033872e-03], [-5.88100225e-05], [-3.45692300e-03], 
#                       [-3.61814914e-06], [-3.61509751e-03], [ 6.55703637e-05], [-4.62154306e-03], [-2.72362507e-04], [-4.83402467e-03], 
#                       [ 4.46964088e-04], [-5.35644923e-03], [-1.21261943e-04], [-5.82850208e-03], [ 1.70865437e-04], [-6.49572528e-03], 
#                       [ 3.16952212e-04], [-6.92955374e-03], [ 1.10920662e-04], [-7.41288834e-03], [-3.32198676e-04], [-7.89649504e-03], 
#                       [-2.05932995e-04], [-8.56170640e-03], [ 7.65109750e-05], [-9.38544332e-03], [ 1.17608009e-04], [-9.61536981e-03], 
#                       [ 1.29451472e-05], [-1.01561306e-02], [ 1.28525208e-04], [-1.08475044e-02], [ 3.56352378e-05], [-1.13265548e-02], 
#                       [ 1.40982436e-04], [-1.16532588e-02], [ 2.96127377e-04], [ 6.06898351e-04], [-1.56653026e-04], [ 5.51312351e-05], 
#                       [ 1.57511935e-04], [-3.28567568e-04], [-1.29859445e-04], [-3.48481767e-04], [ 2.02322621e-04], [ 2.20629604e-04], 
#                       [ 3.81351319e-04], [-3.88225416e-04], [ 3.70185100e-04], [-9.91448769e-04], [-6.12848642e-05], [-1.14510549e-03], 
#                       [ 5.11130660e-04], [-9.38089883e-04], [ 4.52764267e-04], [-1.71958262e-03], [ 7.34856211e-04], [-1.95648126e-03], 
#                       [ 2.85018357e-04], [-2.38619830e-03], [ 1.80854438e-04], [-2.56592655e-03], [ 2.66985855e-04], [-2.81164286e-03], 
#                       [ 9.16669457e-04], [-3.43832495e-03], [ 1.23399659e-04], [-3.82044050e-03], [ 5.11768701e-04], [-4.27782481e-03], 
#                       [ 5.99683282e-04], [-5.02951045e-03], [ 3.63660154e-04], [-5.28055171e-03], [ 7.31223634e-04], [-6.14540397e-03], 
#                       [ 7.52563622e-04], [-6.63509734e-03], [ 3.91032799e-04], [-6.83880631e-03], [ 5.41321599e-04], [-7.04491988e-03], 
#                       [ 4.72254673e-04], [-8.32115333e-03], [ 3.85227541e-04], [-8.57308991e-03], [ 5.08257326e-04], [-9.17596245e-03], 
#                       [ 9.34764966e-04], [-9.64337997e-03], [ 5.62329798e-04], [-1.05242231e-02], [ 3.83574570e-04], [-1.09419262e-02], 
#                       [ 3.19615923e-04], [-1.16365103e-02], [ 5.10202989e-04], [-1.20868924e-02], [-6.66937991e-05], [ 8.54458999e-05], 
#                       [-1.72887105e-04], [-1.31750326e-04], [ 5.87430860e-04], [-5.30925742e-04], [ 2.75502261e-04], [-9.41371239e-05], 
#                       [ 3.90154923e-04], [-4.89565607e-04], [ 1.32525042e-04], [-2.79913490e-04], [ 1.85552165e-04], [-7.71210305e-04], 
#                       [ 6.73529529e-04], [-9.45891607e-04], [ 5.32812648e-04], [-1.48301500e-03], [ 4.68455358e-04], [-1.42984824e-03], 
#                       [ 6.50678912e-04], [-1.69375041e-03], [ 7.72197520e-04], [-2.14883260e-03], [ 1.13888805e-03], [-2.37129476e-03], 
#                       [ 4.19804295e-04], [-3.01783597e-03], [ 8.38912700e-04], [-3.28772982e-03], [ 8.76551073e-04], [-3.55595246e-03], 
#                       [ 1.03305316e-03], [-4.32956047e-03], [ 1.04247199e-03], [-4.39829961e-03], [ 1.24670131e-03], [-5.31874056e-03], 
#                       [ 9.71253229e-04], [-5.82608286e-03], [ 8.02824012e-04], [-6.28870094e-03], [ 9.62486063e-04], [-6.94445410e-03], 
#                       [ 1.05609206e-03], [-7.36871369e-03], [ 9.96013406e-04], [-8.08751903e-03], [ 1.17815980e-03], [-8.59635515e-03], 
#                       [ 1.15612958e-03], [-9.17355035e-03], [ 1.28043242e-03], [-1.02148275e-02], [ 9.31725455e-04], [-1.02347487e-02], 
#                       [ 1.25626447e-03], [-1.11561242e-02], [ 1.40271878e-03], [-1.15380112e-02], [ 1.29098210e-03], [-1.23545011e-02], 
#                       [-5.01153787e-05], [ 3.30629678e-04], [ 2.29557536e-04], [-2.45651838e-04], [ 3.55155968e-04], [-1.54585045e-04], 
#                       [ 1.08933541e-05], [-1.92669566e-04], [ 4.78068958e-04], [-2.74472363e-04], [ 2.00095293e-04], [-5.17173328e-04], 
#                       [ 7.92255165e-04], [-6.53344232e-04], [ 1.03202662e-03], [-1.11636905e-03], [ 9.75968924e-04], [-1.13071621e-03], 
#                       [ 4.98927642e-04], [-2.09493123e-03], [ 1.23132292e-03], [-1.84693657e-03], [ 1.22427136e-03], [-2.13788885e-03],
#                       [ 5.77948449e-04], [-2.54236349e-03], [ 1.17048704e-03], [-3.10982249e-03], [ 1.31594341e-03], [-3.60669909e-03], 
#                       [ 1.16245659e-03], [-3.57401120e-03], [ 1.29531530e-03], [-4.37374277e-03], [ 1.23243658e-03], [-4.91288312e-03], 
#                       [ 1.39366359e-03], [-5.13852434e-03], [ 1.47060523e-03], [-5.74002865e-03], [ 1.93798528e-03], [-6.07629806e-03], 
#                       [ 1.99705462e-03], [-6.94788694e-03], [ 1.91751803e-03], [-7.26176091e-03], [ 1.87346923e-03], [-8.12547086e-03], 
#                       [ 1.59260249e-03], [-8.32200347e-03], [ 1.80949061e-03], [-9.17006630e-03], [ 1.57214401e-03], [-1.01100087e-02], 
#                       [ 1.64640631e-03], [-1.04620246e-02], [ 1.91618796e-03], [-1.10348578e-02], [ 1.55699717e-03], [-1.17334477e-02], 
#                       [ 1.65046849e-03], [-1.21039943e-02]]  

lines = []
my_mesh = Mesh(inp['mesh'])
true_displacement = my_mesh.displacement(config['True Material Parameters']['Youngs Modulus'], config['True Material Parameters']['Poissons Ratio'])

my_mesh.deformation_plot(label = f'True Deformation, E: %.3f, v: %.3f' % (config['True Material Parameters']['Youngs Modulus'], config['True Material Parameters']['Poissons Ratio']), colour= 'plum', ch = 1, ax = ax1, lines = lines, ls = 'solid')

fig2, ax2 = plt.subplots(2, 1)
my_mesh.contour_plot('True', fig2, ax2)

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
    fig5, ax5 = plt.subplots(2, 1)
    utilities.histogram(results['MCMC'], 100, ['Youngs Modulus', 'Youngs Modulus','Poissons Ratio', 'Poissons Ratio'], ini, inp['range'], fig5, ax5)
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
    utilities.histogram(results['MCMC'], 100, ['Youngs Modulus', 'Youngs Modulus','Poissons Ratio', 'Poissons Ratio'], ini, inp['range'], fig5, ax5)
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
    utilities.histogram(results['MCMC'], 100, ['Youngs Modulus', 'Youngs Modulus','Poissons Ratio', 'Poissons Ratio'], ini, inp['range'], fig5, ax5)
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
    utilities.histogram(results['MCMC'], 100, ['Youngs Modulus', 'Youngs Modulus','Poissons Ratio', 'Poissons Ratio'], ini, inp['range'], fig5, ax5)
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
    utilities.histogram(results['MCMC'], 100, ['Youngs Modulus', 'Youngs Modulus','Poissons Ratio', 'Poissons Ratio'], ini, inp['range'], fig5, ax5)
    print('----------------------------------------------')
    print('Preconditioned Crank-Nicolson')
    print('----------------------------------------------')

elif inp['Method'] == 5:
    #The EnKF algorithm 
    B = EnKF_mcmc(inp)
    results = B.EnKF_go()
    if config['Print Chain'] == 1:
        print(results['MCMC'])
    fig5, ax5 = plt.subplots(2, 1)
    utilities.histogram(results['MCMC'], 100, ['Youngs Modulus', 'Youngs Modulus','Poissons Ratio', 'Poissons Ratio'], ini, inp['range'], fig5, ax5)
    print('----------------------------------------------')
    print('Ensemble Kalman Filter')
    print('----------------------------------------------')
   
print('Acceptance Rate: %.3f' % results['accepted'])
print('Number of Samples: %.0f' % config['Number of samples'])
print('The median of the Youngs Modulus posterior is: %f, with uncertainty +/- %.5f' % (np.median(results['MCMC'][0]), np.sqrt(np.var(results['MCMC'][0]))))
print('The median of the Poissons Ratio posterior is: %f, with uncertainty +/- %.5f' % (np.median(results['MCMC'][1]), np.sqrt(np.var(results['MCMC'][1]))))
print()
my_mesh = Mesh(inp['mesh'])
my_mesh.displacement(np.median(results['MCMC'][0]), np.median(results['MCMC'][1]))
my_mesh.deformation_plot(label = f'Estimated Deformation, E: %.3f, v: %.3f' % (np.median(results['MCMC'][0]), np.median(results['MCMC'][1])), ls =(0,(3,5)),colour = 'rebeccapurple', ch = 0.9, ax = ax1, lines = lines)

fig3, ax3 = plt.subplots(2, 1)
my_mesh.contour_plot('Estimated', fig3, ax3)

fig4, ax4 = plt.subplots(2, 1)
my_mesh.error_plot(true_displacement, fig4, ax4)

ax1.set_title('Deformation Plot', fontsize = 25)
fig.legend(loc = 'lower center', ncols=2)

plt.show()
