import numpy as np

def check_bounds(x, range):
    """Check that the proposed thetas are within the defined ranges.

        Args:
            measurement (numpy.array): np.array([E, G_f]), np.array([[E_min, G_min],[E_max, G_max]])

        Returns:
            numpy.array: np.array([E, G_f]) with values previously falling outside of the range replaced with the bounds
    """
    
    mini = range[0]
    maxi = range[1]

    x[x<mini] = mini
    x[x>maxi] = maxi

    return x

def ESS(measurements, e):
    """Calculates EES results of FEM with etsimated E, G_f and the measured data

        Args:
            measurement (numpy.array): np.array([observations]), np.array([E, G_f])

        Returns:
            numpy.array: ESS, resulting FEM
    """
    arr = forward_model(e)
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

def forward_model(x):
    #e=4.259
    a = 1
    que = 14
    etr = 0.5
    l = 1
    nel = 1000
    d1,dl1,sig1,step1,nel1 = linelast(nel,x,a,que,etr,l)
    return d1

def linelast(nel,e,a,que,etr,l):
    ea = e*a

    ng = nel + 1

    d = np.zeros([ng,1])

    dl  = l/nel

    ke = ea/dl*np.array([[1, -1],[-1, 1]])

    q = np.zeros([ng,1])
    kg = np.zeros([ng,ng])

    for i in range(nel):
        x1 = (i)*dl
        x2 = (i+1)*dl
        for j in range(2):
            for k in range(2):
                kg[j+i, k+i] += ke[j, k]
            if j == 0:
                q[i] = q[i] + (l*np.cos(2.0*np.pi*x1/l)/(2.0*np.pi) -l**2*np.sin(2.0*np.pi*x2/l)/(4.0*np.pi**2*dl) + l**2*np.sin(2.0*np.pi*x1/l)/(4.0*np.pi**2*dl))*que
            elif j == 1:
                q[i+1] = q[i+1] + (-l*np.cos(2.0*np.pi*x2/l)/(2.0*np.pi)+l**2*np.sin(2.0*np.pi*x2/l)/(4.0*np.pi**2*dl)-l**2*np.sin(2.0*np.pi*x1/l)/(4.0*np.pi**2*dl))*que

    kg[0,0] = 1
    kg[0,1:] = np.zeros(nel)
    kg[1:, 0] = np.zeros(nel)

    q[0] = 0
    q[-1] += etr*a
    d[:] = np.linalg.solve(kg, q)

    sig = np.zeros([2*nel, nel])
    #INDEXES ARE WRONG FIX LATER
    step = np.zeros(2*nel)
    for i in range(nel - 1):
        sig[2*i, :] = e/dl * (d[i+1] - d[i])
        sig[2*i+1,:] = e/dl * (d[i+1] - d[i])
        step[2*i] = dl*(i)
        step[2*i+1] = dl*(i+1) 

    return d, dl, sig, step, nel