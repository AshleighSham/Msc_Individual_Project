from config import config
import numpy as np
import utilities
import seaborn as sns
from mesh import Mesh
sns.set_context('talk')
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st


inp ={}

inp['mesh'] = [config['Mesh grid']['quad'],
               config['Mesh grid']['sf'], 
               config['Mesh grid']['Nodal Coordinates'], 
               config['Mesh grid']['Element Node Numbers'],
               config['Mesh grid']['Number of elements'],
               config['Mesh grid']['Force Magnitude'],
               config['Mesh grid']['Force Nodes'],
               config['Mesh grid']['Fixed Nodes']]

def forward_model_SA(x, ms = inp['mesh']):
    #generate mesh
    my_mesh = Mesh(ms)
    Ex = x[0]
    vx = x[1]
    d1 = my_mesh.displacement(Ex, vx)
    return d1

E = 10
v = 0.3

E_1 = st.norm(E, 5)
v_1 = st.norm(v, 0.5)

mu = [E, v]
variables = [E_1, v_1]
variable_names = ['$E$', '$v$']


def first_order_sensitivities(Function, Variables, F0, eps=1e-6):
    x0 = np.array([v.mean() for v in Variables])
    dF_dx = np.zeros(len(x0))
    for i, xi in enumerate(x0):
        xp = x0.copy()
        h = eps*xi
        xp[i] = xi + h
        dF_dx[i] = (np.linalg.norm(Function(xp) - F0))/h
    return dF_dx

x0 = np.array([v.mean() for v in variables])
sigma0 = np.array([v.std() for v in variables])

F0 = forward_model_SA(x0)

dF_dx = first_order_sensitivities(forward_model_SA, variables, F0)

df_disp = pd.DataFrame.from_dict(dict(parameter=variable_names,
                                 mean=x0,
                                 sigma=sigma0,
                                 sensitivity=dF_dx,
                                 scaled_sensitivity=x0*dF_dx,
                                 sensitivity_index=sigma0*dF_dx))

fig, ax = plt.subplots(figsize = (5,5))

ax = df_disp['scaled_sensitivity'].abs().plot(xticks=df_disp.index, color = 'cornflowerblue')

ax.set_xticklabels(df_disp.parameter)
ax.set_ylabel('Absolute scaled sensitivity')
ax.set_xlabel('Parameter')
ax.grid()

plt.show()