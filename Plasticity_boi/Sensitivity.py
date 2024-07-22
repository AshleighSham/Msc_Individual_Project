from config import config
import numpy as np
import seaborn as sns
from mesh import Mesh
from plasticity_Von_Mises_quasi_static_pst_2d import forward_model
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
               config['Mesh grid']['Fixed Nodes'],
               config['Mesh grid']['Element ID'],
               config['Mesh grid']['thickness']]

E = 206.9
v = 0.29
Y = 0.45
H = 0.2
 
E_1 = st.norm(E, 10)
v_1 = st.norm(v, 0.01)
Y_1 = st.norm(Y, 0.01)
H_1 = st.norm(H, 0.01)

mu = [E, v, Y, H]
variables = [E_1, v_1, Y_1, H_1]
variable_names = ['$E$', '$v$', '$\sigma_y$', '$H$']


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

F0 = forward_model(x0)

dF_dx = first_order_sensitivities(forward_model, variables, F0)

df_disp = pd.DataFrame.from_dict(dict(parameter=variable_names,
                                 mean=x0,
                                 sigma=sigma0,
                                 sensitivity=dF_dx,
                                 scaled_sensitivity=x0*dF_dx,
                                 sensitivity_index=sigma0*dF_dx))

ax = df_disp['scaled_sensitivity'].abs().plot(xticks=df_disp.index, color = 'cornflowerblue', marker ='s', markersize = 7)

ax.set_xticklabels(df_disp.parameter)
ax.set_ylabel('Absolute scaled sensitivity')
ax.set_xlabel('Parameter')
ax.grid()

plt.show()