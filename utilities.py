import numpy as np

def forward_model(x):



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
