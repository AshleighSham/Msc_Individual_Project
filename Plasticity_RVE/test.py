import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#past_chain_info = np.load('MH.npy',allow_pickle='TRUE').item()
data = pd.read_csv(r'C:\Users\ashle\Documents\GitHub\Portfolio\ES98C\Plasticity_RVE\AMH.csv')



numd = data.to_numpy()

# dumy_theta = np.zeros((4, len(data['E'])))
# dumy_theta[0,:] = data['E']
# dumy_theta[1,:] = data['v']
# dumy_theta[2,:] = data['sy']
# dumy_theta[3,:] = data['H']
# dumy_theta[0,:] = data['E']
# dumy_theta[1,:] = data['v']
# dumy_theta[2,:] = data['sy']
# dumy_theta[3,:] = data['H']

fig, ax = plt.subplots(8, 1)

ax[0].plot(range(len(data['Ei'])), data['Ei'])
ax[1].plot(range(len(data['Ei'])), data['vi'])
ax[2].plot(range(len(data['Ei'])), data['syi'])
ax[3].plot(range(len(data['Ei'])), data['Hi'])
ax[4].plot(range(len(data['Ei'])), data['Em'])
ax[5].plot(range(len(data['Ei'])), data['vm'])
ax[6].plot(range(len(data['Ei'])), data['sym'])
ax[7].plot(range(len(data['Ei'])), data['Hm'])


ax[0].axhline(10, color = 'black', linestyle = (0,(5,5)))
ax[1].axhline(0.3, color = 'black', linestyle = (0,(5,5)))
ax[2].axhline(0.005, color = 'black', linestyle = (0,(5,5)))
ax[3].axhline(0.001, color = 'black', linestyle = (0,(5,5)))
ax[4].axhline(1, color = 'black', linestyle = (0,(5,5)))
ax[5].axhline(0.3, color = 'black', linestyle = (0,(5,5)))
ax[6].axhline(0.007, color = 'black', linestyle = (0,(5,5)))
ax[7].axhline(0.002, color = 'black', linestyle = (0,(5,5)))

plt.show()

print('The median of the Youngs Modulus posterior is: %f, with uncertainty +/- %.5f' % (np.median(data['E'][500:]), np.sqrt(np.var(data['E'][500:]))))
print('The median of the Poissons Ratio posterior is: %f, with uncertainty +/- %.5f' % (np.median(data['v'][500:]), np.sqrt(np.var(data['v'][500:]))))
print('The median of the Yield Stress posterior is: %f, with uncertainty +/- %.5f' % (np.median(data['sy'][500:]), np.sqrt(np.var(data['sy'][500:]))))
print('The median of the Hardening Modulus posterior is: %f, with uncertainty +/- %.5f' % (np.median(data['H'][500:]), np.sqrt(np.var(data['H'][500:]))))
print()
