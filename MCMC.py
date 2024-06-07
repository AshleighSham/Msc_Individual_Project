from EnKF import EnKFmcmc
from MH import MHmcmc
import numpy as np


from config import config

class MCMC():
    def __init__(self, method = config['method'], n = 10000):
        self.n = n
        self.method = method

    def runMCMC(self):
        if self.method == "MH":
            return MHmcmc()