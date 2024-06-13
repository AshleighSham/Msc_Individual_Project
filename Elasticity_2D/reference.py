import numpy as np
import scipy as sp
import utilities

mesurements = utilities.forward_model(np.array([30, 0.1]))
print(mesurements)