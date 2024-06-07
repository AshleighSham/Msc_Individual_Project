import numpy as np
import scipy as sp
import utilities

class MH_mcmc():
    def __init__(self, inp):

        self.range = inp['range']
        self.nsamples = inp['samples']
        self.initial_cov = inp['icov']
        self.initial_theta = inp['theta0']
        self.sigma = inp['sigma']
        self.observations = inp['measurement']
        self.K0 = inp['Kalmans']
        self.m0 = inp['me']

        self.Rj = sp.linalg.cholesky(self.initial_cov)
        self.dim = len(self.range[0])

        self.MCMC = np.zeros(self.nsamples, self.dim)
        self.oldpi, self.oldvalue = utilities.ESS(self.observations, self.initial_theta)
        self.results = {}

        self.results['values'] = [self.oldvalue]

        self.accepted = 0
        self.MCMC[0] = self.initial_theta

        self.thetaj = self.initial_theta.T

    def Run(self):
        j = 1
        while j < self.nsamples:
            thetas = self.thetaj + self.Rj * np.random.normal(self.dim, 1)
            thetas = utilities.check_bounds(thetas, self.range)

            newpi, newvalue = utilities.ESS(self.observations, thetas)
            lam = min(1, np.exp(-0.5*(newpi - self.oldpi)/self.sigma))

            if np.random.uniform(0, 1) < lam:
                self.accepted += 1
                self.thetaj = thetas
                self.oldpi = newpi
                self.oldvalue = newvalue

            self.results['values'].append(self.oldvalue)

            self.MCMC[j] = self.thetaj

            if j % 500 == 0:
                print('number of sample: %d\n', j)

            j += 1

        self.results['MCMC'] = self.MCMC.T
        self.results['accepted'] = 100*self.accepted/self.nsamples
