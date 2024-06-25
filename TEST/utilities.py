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
    R = [maxi[i] - mini[i] for i in  range(np.size(rang, 1))]
    for i in range(np.size(rang, 1)):
        if x[i]<mini[i]:
            if mini[i] - x[i] > R[i]:
                x[i] = mini[i]
            else:
                x[i] = 2*mini[i] - x[i]
        if x[i]> maxi[i]:
            if x[i] - maxi[i] > R[i]:
                x[i] = maxi[i]
            else: 
                x[i] = x[i] - 2*maxi[i]

    return x

def ESS(measurements, e, ms):
    """Calculates EES results of FEM with etsimated E, G_f and the measured data

        Args:
            measurement (numpy.array): np.array([observations]), np.array([E, G_f])

        Returns:
            numpy.array: ESS, resulting FEM
    """
    edges_ind = []
    if ms[0] != 0:
        A = range(ms[0][1])
        edges_ind = [a for a in A]
        for i in range(ms[0][0]-1):
            edges_ind.append(A[-1] + 1 + ms[0][1]*i)
            edges_ind.append(A[-1] + ms[0][1]*i)
        for i in range(ms[0][1]):
            edges_ind.append(ms[0][1]*ms[0][0]-1 - i)

    meas_edge = []
    for i in edges_ind:
        meas_edge.append(2*i)
        meas_edge.append(2*i + 1)
    arr = forward_model(e, ms)

    indexs = np.random.choice(range(0, len(measurements)//2), int(0.7*len(measurements)//2), replace = False)
    rand_ind = []
    for i in indexs:
        rand_ind.append(2*i)
        rand_ind.append(2*i + 1)

    diff2 = np.linalg.norm(measurements[meas_edge] - arr[meas_edge])
    ss1 = np.linalg.norm(measurements[rand_ind] - arr[rand_ind])
    return ss1, arr


def plane_strain(E, nu): 
    """Calculates plane strain elasticity matrix

    Args:
        material properties: Youngs Modulus E, Poissons Ratio nu

    Returns:
        numpy.array: Plane strain elastcity matrix 3x3
    """
    return E/((1 + nu)*(1 - 2*nu))*np.array([[1-nu, nu, 0],[nu, 1-nu ,0],[0,0,0.5*(1-2*nu)]])

def plane_stress(E, nu): 
    """Calculates plane strain elasticity matrix

    Args:
        material properties: Youngs Modulus E, Poissons Ratio nu

    Returns:
        numpy.array: Plane strain elastcity matrix 3x3
    """
    return E/(1 + nu**2)*np.array([[1, nu, 0],[nu, 1 ,0],[0,0,0.5*(1-nu)]])

def Jacobian(xyze, xi, eta): 
    """Calculates Jacobian

    Args: nodal coordinates for a single element, natural coordinate - horizontal, natural coordinate - vertical 
        

    Returns:
        element Jacobian matrix and determinant
    """
    # natural nodal coordinates
    natcoord = np.array([c for c in xyze])
        
    # derivatives of shape functions w.r.t. natural coordinates 
    dNdnat = np.zeros((2,4))
    dNdnat[0,:]= [0.25*(eta - 1), 0.25*(1 - eta), 0.25*(1 + eta), 0.25*(-eta-1)]
    dNdnat[1,:]= [0.25*(xi - 1), 0.25*(-xi -1), 0.25*(1 + xi), 0.25*(1 - xi)]

    # element Jacobian matrix and determinant
    Jmat = dNdnat @ natcoord
    return Jmat, np.linalg.det(Jmat)

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

def histogram(data, titles, truevalues, ranges, f, axes):
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

