import numpy as np
from mesh import Mesh
import matplotlib.pyplot as plt

def check_bounds(x, rang):
    """Check that the proposed thetas are within the defined ranges.

        Args:
            measurement (numpy.array): np.array([E, G_f]), np.array([[E_min, G_min],[E_max, G_max]])

        Returns:
            numpy.array: np.array([E, G_f]) with values previously falling outside of the range replaced with the bounds
    """
    
    mini = rang[0]
    maxi = rang[1]
    # R = [maxi[i] - mini[i] for i in  range(np.size(rang, 1))]
    # for i in range(np.size(rang, 1)):
    #     if x[i]<mini[i]:
    #         if mini[i] - x[i] > R[i]:
    #             x[i] = mini[i]
    #         else:
    #             x[i] = 2*mini[i] - x[i]
    #     if x[i]> maxi[i]:
    #         if x[i] - maxi[i] > R[i]:
    #             x[i] = maxi[i]
    #         else: 
    #             x[i] = maxi[i] - (x[i] - maxi[i])

    for i in range(np.size(rang, 1)):
        if x[i]<mini[i]:
            x[i] = mini[i]
        elif x[i]> maxi[i]:
            x[i] = maxi[i]

    return x

def ESS(measurements, e, ms):
    """Calculates EES results of FEM with etsimated E, G_f and the measured data

        Args:
            measurement (numpy.array): np.array([observations]), np.array([E, G_f])

        Returns:
            numpy.array: ESS, resulting FEM
    """
    arr = forward_model(e, ms)
    mae = np.mean(abs(measurements - arr))
    mav = np.mean(abs(measurements - np.mean(measurements)))
    ss1 = mae / mav
    return np.linalg.norm(measurements - arr), arr

def forward_model(args, ms):
    #generate mesh
    my_mesh = Mesh(ms)
    E1 = [args[0], args[1]]
    nu1 = [args[2], args[3]]
    d1 = my_mesh.displacement(E1, nu1)
    return d1

f = lambda x: np.exp(-0.5*x**2)*(2*np.pi)**-1
def normalkernel(x, u):
    a = 0
    h = 1.06*np.sqrt(np.var(u))*len(u)**-0.2
    N = len(u)
    for i in u:
        a += f((x - i)* h**-1)
    a *= (N*h)**-1
    return a

def histogram(data, titles, truevalues, ranges):
    figh, axes = plt.subplots(len(data), 1)
    for i in range(len(data)):
        axes[i].hist(data[i], 70, density = True, alpha = 0.9, color = 'plum')
        axes[i].set_title(f'Prosterior Distribution for the {titles[i]}')
        X = np.linspace(min(data[i]), max(data[i]), 200)
        axes[i].plot(X, [4*normalkernel(x, data[i]) for x in X], color = 'rebeccapurple', alpha =0.9, linewidth = 2)
        axes[i].axvline(np.median(data[i]), linestyle = (0,(4,2)), alpha = 0.75, color = 'rebeccapurple', label = 'Posterior Median', linewidth = 1.8)
        axes[i].axvline(truevalues[i], alpha = 0.8, linestyle = (0,(4,8)), color = 'k', label = 'True Value', linewidth = 1.8)
        axes[i].set_xlim([ranges[0][i], ranges[1][i]])
        axes[i].grid()
        axes[i].legend()

def histogram_bulk(data, label, ranges, f, ax, colour, a, l, h):
    for i in range(len(data)):
        X = np.linspace(min(data[i]), max(data[i]), 200)
        o = [X, [4*normalkernel(x, data[i]) for x in X]]
        ax[i].plot(X, [4*normalkernel(x, data[i]) for x in X], color = colour, alpha =a, linewidth = l, label = label)
        h[i].append(o)
        ax[i].grid()
        ax[i].legend()
