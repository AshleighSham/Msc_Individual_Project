import numpy as np

def forward_model(x):

    

def check_bounds(x, range):

    mini = range[0]
    maxi = range[1]

    x[x<mini] = mini
    x[x>maxi] = maxi

    return x

def ESS(measurements, e):

    arr = forward_model(e)
    ss1 = np.linalg.norm(measurements - arr)

    return ss1, arr
