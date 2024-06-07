import numpy as np
from DRAM import DRAM_algorithm

class EnKF_mcmc():
    def __init__(self, inp):

        self.range = inp['range']
        self.nsamples = inp['samples']
        self.initial_cov = inp['icov']
        self.initial_theta = inp['theta0']
        self.sigma = inp['sigma']
        self.observations = inp['measurement']
        self.K0 = inp['Kalmans']
        self.m0 = inp['me']

        samples = self.
        results = DRAM_algorithm(inp)



    def 

