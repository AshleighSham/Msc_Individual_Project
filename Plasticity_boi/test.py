import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

past_chain_info = np.load('MH.npy',allow_pickle='TRUE').item()
data = pd.read_csv(r'C:\Users\ashle\Documents\GitHub\Portfolio\ES98C\Plasticity_boi\MH.csv')

numd = data.to_numpy()

dumy_theta = np.zeros((4, len(data['E'])))
dumy_theta[0,:] = data['E']
dumy_theta[1,:] = data['v']
dumy_theta[2,:] = data['sy']
dumy_theta[3,:] = data['H']

dumy_values = np.zeros((len(data['E']), len(numd[-1])-5))

for i in range(len(dumy_values[0])):
    dumy_values[:,i] = data[f'{i}']

# dumy_cov = np.zeros(16)
# var_lab = ['var0','var1','var2','var3','var4','var5','var6','var7','var8','var9','var10','var11','var12','var13','var14','var15']
# for i,q in zip(var_lab, range(16)):
#     dumy_cov[q] = data[i].to_numpy()[-1]
# dumy_cov.reshape(4,4)

# DATA = {'thetaj': dumy_theta, 'oldvalue': dumy_values}

# np.save('MH.npy', DATA)

print(dumy_theta[:,0])
print(past_chain_info['thetaj'][:,-1])
