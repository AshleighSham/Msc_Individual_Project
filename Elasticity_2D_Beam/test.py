import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r'C:\Users\ashle\Documents\GitHub\Portfolio\ES98C\Elasticity_2D_4DMCMC\EnKF.csv')

numd = data.to_numpy()

dumy_theta = np.zeros((4, len(data['E1'])))
dumy_theta[0,:] = data['E1']
dumy_theta[1,:] = data['E2']
dumy_theta[2,:] = data['v1']
dumy_theta[3,:] = data['v2']

dumy_values = np.zeros((len(data['E1']), len(numd[-1])-1-4-16))

for i in range(len(dumy_values[0])):
    dumy_values[:,i] = data[f'{i}']

dumy_cov = np.zeros(16)
var_lab = ['var0','var1','var2','var3','var4','var5','var6','var7','var8','var9','var10','var11','var12','var13','var14','var15']
for i,q in zip(var_lab, range(16)):
    dumy_cov[q] = data[i].to_numpy()[-1]
dumy_cov.reshape(4,4)

DATA = {'thetaj': dumy_theta, 'oldvalue': dumy_values, 'past_cov': dumy_cov}

np.save('test_data.npy', DATA)
