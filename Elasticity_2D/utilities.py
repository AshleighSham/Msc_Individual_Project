import numpy as np
from mesh import Mesh

def check_bounds(x, rang):
    """Check that the proposed thetas are within the defined ranges.

        Args:
            measurement (numpy.array): np.array([E, G_f]), np.array([[E_min, G_min],[E_max, G_max]])

        Returns:
            numpy.array: np.array([E, G_f]) with values previously falling outside of the range replaced with the bounds
    """
    
    mini = rang[0]
    maxi = rang[1]
    
    for i in range(np.size(rang, 1)):
        x[i][x[i]<mini[i]] = mini[i]
        x[i][x[i]>maxi[i]] = maxi[i]

    return x

def ESS(measurements, e, ms):
    """Calculates EES results of FEM with etsimated E, G_f and the measured data

        Args:
            measurement (numpy.array): np.array([observations]), np.array([E, G_f])

        Returns:
            numpy.array: ESS, resulting FEM
    """
    arr = forward_model(e, ms)
    ss1 = np.linalg.norm(measurements - arr)

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
    d1 = my_mesh.displacement(args[0], args[1])
    return d1