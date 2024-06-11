import numpy as np
import scipy as sp
import utilities

class DRAM_algorithm:
    def __init__(self, inp):

        self.range = inp['range']

        self.nsamples = inp['samples']
        self.initial_cov = inp['icov']
        self.initial_theta = inp['theta0']
        self.sigma = inp['sigma']
        self.observations = inp['measurement']
        self.K0 = inp['Kalmans']
        self.m0 = inp['me']

        self.eps = 1e-6
        self.adpt = 100
        self.Rj = sp.linalg.cholesky(self.initial_cov)
        self.dim = len(self.range[0])
        self.Kp = 2.4/np.sqrt(self.dim)

        self.MCMC = np.zeros(self.nsamples, self.dim)
        self.oldpi, self.oldvalue = utilities.ESS(self.observations, self.initial_theta)

        self.results = {}
        self.results['values'] = [self.oldvalue]

        self.R2 = np.divide(self.Rj, 5)
        self.invR = np.linalg.solve(self.Rj, np.eye(len(self.Rj[0])))
        self.accepted = 0
        self.MCMC[0,:] = self.initial_theta
        self.thetaj = self.initial_theta.T

        #initiase Kalman features
        self.MCMC_cov = np.zeros_like(self.initial_cov)
        self.MCMC_mean = np.zeros_like(self.initial_theta)
        self.ss = np.array([0])
        self.ii = 0

        DRAM_go()

        def update_cov(self, w, ind):
            x = self.MCMC[self.ii:ind]
            n = np.size(x, 0) #num of rows
            p = np.size(x, 1) #num of cols

            if len(w) == 1:
                w *= np.ones(n, 1)

            i = 0
            while i < n:
                xi = x[i]
                wsum = w[i]
                xmeann = xi

                xmean = self.MCMC_mean + (xmeann - self.MCMC_mean)@(wsum/(wsum + self.ss))
                xcov = self.MCMC_cov + (wsum/(wsum + self.MCMC_cov - 1))@((self.ss/(wsum + self.ss))@(xi-self.MCMC_mean).T@(xi - self.MCMC_mean) - self.MCMC_cov)

                wsum += self.ss
                self.MCMC_cov = xcov
                self.MCMC_mean = xmean
                self.ss = wsum

        def DRAM_go(self):
            j = 1
            while j < self.nsamples:
                thetas = self.thetaj + self.Rj*np.random.normal(self.dim, 1)

                thetas = utilities.check_bounds(thetas, self.range)

                newpi, newvalue = utilities.ESS(self.observations, thetas)
                lam = min(1, np.exp(-0.5*(newpi - self.oldpi)/self.sigma))
                if np.random.uniform(0,1) < lam:
                    self.accepted += 1
                    self.thetaj = thetas
                    self.oldpi = newpi
                    self.oldvalue = newvalue

                else:
                    thetass = self.thetaj + self.R2*np.random.normal(self.dim, 1)
                    thetass = utilities.check_bounds(thetass, self.range)

                    newss2, newvalue2 = utilities.ESS(self.observations, thetass)
                    k1 = min(1, np.exp(-0.5*(newpi - newss2)/self.sigma))
                    k2 = np.exp(-0.5*(newss2-self.oldpi)/self.sigma)
                    k3 = np.exp(-0.5*(np.linalog.norm(self.invR*(thetass - thetas))**2))
                    lam2 = k2*k3*(1-k1)/(1-lam)
                    if np.random.uniform(0,1) < lam2:
                        self.accepted += 1
                        self.thetaj = thetass
                        self.oldpi = newss2
                        self.oldvalue = newvalue2

                self.results['values'].append(self.oldvalue)

                self.MCMC[j] = self.thetaj

                if j % self.adpt == 0:
                    update_cov(1, j)
                    self.ii = j
                    Ra, = np.linalg.cholesky(self.MCMC_cov + np.eye(self.dim)*self.eps)

                    self.Rj = Ra * self.Kp

                j += 1

            self.results['MCMC'] = self.MCMC.T
            self.results['accepted'] = 100*self.accepted/self.nsamples

            return self.results
            



