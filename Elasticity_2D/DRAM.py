import numpy as np
import scipy as sp
import utilities as utilities

class DRAM_algorithm():
    def __init__(self, inp):

        self.range = inp['range']

        self.nsamples = inp['nsamples']
        self.initial_cov = inp['icov']
        self.initial_theta = inp['theta0']
        self.sigma = inp['sigma']
        self.observations = inp['measurement']
        self.K0 = inp['Kalmans']
        self.m0 = inp['me']
        self.mesh = inp['mesh']

        self.eps = 1e-5
        self.adpt = 100
        self.Rj = sp.linalg.cholesky(self.initial_cov)
        self.dim = np.size(self.range, 1)
        self.Kp = 2.4/np.sqrt(self.dim)

        self.MCMC = np.zeros([self.nsamples, self.dim])
        self.oldpi, self.oldvalue = utilities.ESS(self.observations, self.initial_theta, self.mesh)

        self.results = {}
        self.results['values'] = [self.oldvalue*1]

        self.R2 = self.Rj*0.2
        self.invR = np.linalg.solve(self.Rj, np.eye(len(self.Rj[0])))
        self.accepted = 0
        self.MCMC[0,:] = self.initial_theta.T
        self.thetaj = self.initial_theta

        #initiase Kalman features
        self.MCMC_cov = np.zeros_like(self.initial_cov)
        self.MCMC_mean = np.zeros_like(self.initial_theta)
        self.ss = np.array([1])
        self.ii = 0

    def update_cov(self, w, ind):
        x = self.MCMC[self.ii+1:ind] #100, 1
        n = np.size(x, 0) #num of rows
        p = np.size(x, 1) #num of cols

        if int(w) == w:
            w *= np.ones(n)

        i = 0
        while i < n:
            xi = np.array([x[i]])
            wsum = np.array([w[i]])
            xmeann = xi

            xmean = self.MCMC_mean + np.divide(wsum,(wsum + self.ss))*(xmeann - self.MCMC_mean)
            a = np.divide(wsum,(wsum + self.ss-np.array([1])))
            b = np.multiply(np.divide(self.ss,(wsum + self.ss)),(xi-self.MCMC_mean).T)
            xcov = self.MCMC_cov + np.multiply(a,(b@(xi - self.MCMC_mean) - self.MCMC_cov))

            wsum += self.ss
            self.MCMC_cov = xcov #unsure about this
            self.MCMC_mean = xmean
            self.ss = wsum

            i += 1

    def DRAM_go(self):
        j = 1
        while j < self.nsamples:
            thetas = self.thetaj + self.Rj@np.random.normal(size = [self.dim, 1])
            thetas = utilities.check_bounds(thetas, self.range)
            newpi, newvalue = utilities.ESS(self.observations, thetas, self.mesh)
            lam = min(1, np.exp(-0.5*(newpi - self.oldpi)/self.sigma))
            if np.random.uniform(0,1) < lam:
                self.accepted += 1
                self.thetaj = thetas
                self.oldpi = newpi
                self.oldvalue = newvalue

            else:
                thetass = self.thetaj + self.R2@np.random.normal(size = [self.dim, 1])
                thetass = utilities.check_bounds(thetass, self.range)

                newss2, newvalue2 = utilities.ESS(self.observations, thetass, self.mesh)
                k1 = min(1, np.exp(-0.5*(newpi - newss2)/self.sigma))
                k2 = np.exp(-0.5*(newss2-self.oldpi)/self.sigma)
                k3 = np.exp(-0.5*(np.linalg.norm(self.invR@(thetass - thetas))**2))
                lam2 = k2*k3*(1-k1)/(1-lam)
                if np.random.uniform(0,1) < lam2:
                    self.accepted += 1
                    self.thetaj = thetass
                    self.oldpi = newss2
                    self.oldvalue = newvalue2

            self.results['values'].append(self.oldvalue)

            self.MCMC[j, :] = self.thetaj.T

            if j % self.adpt == 0:
                self.update_cov(1, j)
                Ra = np.linalg.cholesky(self.MCMC_cov + np.eye(self.dim)*self.eps)
                self.ii = j*1
                self.Rj = Ra * self.Kp

            j += 1

        self.results['MCMC'] = self.MCMC.T
        self.results['accepted'] = 100*self.accepted/self.nsamples

        return self.results
        



