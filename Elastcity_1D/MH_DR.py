import numpy as np
import scipy as sp
import utilities as utilities

class MH_DR_mcmc():
    def __init__(self, inp):

        self.range = inp['range']

        self.nsamples = inp['nsamples']
        self.initial_cov = inp['icov']
        self.initial_theta = inp['theta0']
        self.sigma = inp['sigma']
        self.observations = inp['measurement']
        self.K0 = inp['Kalmans']
        self.m0 = inp['me']
        self.results ={}

        self.Rj = sp.linalg.cholesky(self.initial_cov)
        self.dim = np.size(self.range, 1)

        self.R2 = self.Rj*0.2
        self.invR = np.linalg.solve(self.Rj, np.eye(len(self.Rj[0])))

        self.MCMC = np.zeros([self.nsamples, self.dim])
        self.oldpi, _ = utilities.ESS(self.observations, self.initial_theta)

        self.accepted = 0
        self.MCMC[0,:] = self.initial_theta
        self.thetaj = self.initial_theta.T


    def MH_DR_go(self):
        j = 1
        while j < self.nsamples:
            thetas = self.thetaj + self.Rj@np.random.normal(size = [self.dim, 1])
            thetas = utilities.check_bounds(thetas, self.range)
            newpi, _ = utilities.ESS(self.observations, thetas)
            lam = min(1, np.exp(-0.5*(newpi - self.oldpi)/self.sigma))
            if np.random.uniform(0,1) < lam:
                self.accepted += 1
                self.thetaj = thetas
                self.oldpi = newpi

            else:
                thetass = self.thetaj + self.R2@np.random.normal(size = [self.dim, 1])
                thetass = utilities.check_bounds(thetass, self.range)

                newss2, _ = utilities.ESS(self.observations, thetass)
                k1 = min(1, np.exp(-0.5*(newpi - newss2)/self.sigma))
                k2 = np.exp(-0.5*(newss2-self.oldpi)/self.sigma)
                k3 = np.exp(-0.5*(np.linalg.norm(self.invR@(thetass - thetas))**2))
                lam2 = k2*k3*(1-k1)/(1-lam)
                if np.random.uniform(0,1) < lam2:
                    self.accepted += 1
                    self.thetaj = thetass
                    self.oldpi = newss2

            self.MCMC[j, :] = self.thetaj*1

            j += 1

        self.results['MCMC'] = self.MCMC.T
        self.results['accepted'] = 100*self.accepted/self.nsamples

        return self.results
 