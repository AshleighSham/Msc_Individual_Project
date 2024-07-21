import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#past_chain_info = np.load('MH.npy',allow_pickle='TRUE').item()
data = pd.read_csv(r'C:\Users\ashle\Documents\GitHub\Portfolio\ES98C\Plasticity_boi\MH_DR.csv')

numd = data.to_numpy()

dumy_theta = np.zeros((4, len(data['E'])))
dumy_theta[0,:] = data['E']
dumy_theta[1,:] = data['v']
dumy_theta[2,:] = data['sy']
dumy_theta[3,:] = data['H']

fig, ax = plt.subplots(4, 1)

ax[0].plot(range(len(data['E'])), data['E'])
ax[1].plot(range(len(data['E'])), data['v'])
ax[2].plot(range(len(data['E'])), data['sy'])
ax[3].plot(range(len(data['E'])), data['H'])


ax[0].axhline(206.9, color = 'black', linestyle = (0,(5,5)))
ax[1].axhline(0.29, color = 'black', linestyle = (0,(5,5)))
ax[2].axhline(0.45, color = 'black', linestyle = (0,(5,5)))
ax[3].axhline(0.2, color = 'black', linestyle = (0,(5,5)))

plt.show()

print('The median of the Youngs Modulus posterior is: %f, with uncertainty +/- %.5f' % (np.median(data['E'][500:]), np.sqrt(np.var(data['E'][500:]))))
print('The median of the Poissons Ratio posterior is: %f, with uncertainty +/- %.5f' % (np.median(data['v'][500:]), np.sqrt(np.var(data['v'][500:]))))
print('The median of the Yield Stress posterior is: %f, with uncertainty +/- %.5f' % (np.median(data['sy'][500:]), np.sqrt(np.var(data['sy'][500:]))))
print('The median of the Hardening Modulus posterior is: %f, with uncertainty +/- %.5f' % (np.median(data['H'][500:]), np.sqrt(np.var(data['H'][500:]))))
print()

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
