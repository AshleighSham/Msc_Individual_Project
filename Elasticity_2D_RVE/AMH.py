import numpy as np
import scipy as sp
import utilities as utilities


class AMH_mcmc():
    def __init__(self, inp):

        # unpack variables
        self.range = inp['range']
        self.nsamples = inp['nsamples']
        self.initial_cov = inp['icov']
        self.initial_theta = inp['theta0']
        self.sigma = inp['sigma']
        self.observations = inp['measurement']
        self.K0 = inp['Kalmans']
        self.m0 = inp['me']
        self.mesh = inp['mesh']
        self.adpt = inp['adapt']
        self.results = {}

        # define variables
        self.eps = 1e-15
        self.Rj = sp.linalg.cholesky(self.initial_cov)
        self.dim = np.size(self.range, 1)
        self.Kp = 0.5*2.38/np.sqrt(self.dim)

        # initialise markov chain
        self.MCMC = np.zeros([self.nsamples, self.dim])
        self.oldpi, _ = utilities.ESS(self.observations, self.initial_theta, self.mesh)
        self.accepted = 0
        self.MCMC[0, :] = self.initial_theta.T
        self.thetaj = self.initial_theta

        # initialise Kalman features
        self.MCMC_cov = np.zeros_like(self.initial_cov)
        self.MCMC_mean = np.zeros_like(self.initial_theta)
        self.ss = np.array([1])
        self.ii = 0

    def update_cov(self, w, ind):
        x = self.MCMC[self.ii+1:ind]
        # num of rows
        n = np.size(x, 0)
        # num of cols
        p = np.size(x, 1)

        if int(w) == w:
            w *= np.ones(n)

        i = 0

        # update mean and covariance
        while i < n:
            xi = np.array([x[i]])
            wsum = np.array([w[i]])
            xmeann = xi.reshape(-1, 1)

            xmean = self.MCMC_mean + np.divide(wsum, (wsum + self.ss))*(xmeann - self.MCMC_mean)
            a = np.divide(wsum, (wsum + self.ss-np.array([1])))
            b = np.multiply(np.divide(self.ss, (wsum + self.ss)), (xi-self.MCMC_mean).T)
            xcov = (((self.ss-1)*((wsum + self.ss - 1)**(-1)))*self.MCMC_cov + (wsum*self.ss*((wsum + self.ss-1)**(-1))) * ((wsum + self.ss)**(-1)) * (np.dot((xmeann-self.MCMC_mean).reshape(p, 1), (xmeann-self.MCMC_mean).reshape(1, p))))

            wsum += self.ss
            self.MCMC_cov = xcov
            self.MCMC_mean = xmean
            self.ss = wsum

            i += 1

    def AMH_go(self):
        j = 1

        while j < self.nsamples:
            # propose new state
            thetas = self.thetaj + self.Rj@np.random.normal(size=[self.dim, 1])
            thetas = utilities.check_bounds(thetas, self.range)

            # probability of new state
            newpi, _ = utilities.ESS(self.observations, thetas, self.mesh)
            lam = min(1, np.exp(-0.5*(newpi - self.oldpi)/self.sigma))

            # accept reject state
            if np.random.uniform(0, 1) < lam:
                self.accepted += 1
                self.thetaj = thetas
                self.oldpi = newpi

            # append to chain
            self.MCMC[j, :] = self.thetaj.T

            # adaption check
            if j % self.adpt == 0:
                self.update_cov(1, j)
                Ra = np.linalg.cholesky(self.MCMC_cov + np.eye(self.dim)*self.eps)
                self.ii = j*1
                self.Rj = Ra * self.Kp

            j += 1

        # define output
        self.results['MCMC'] = self.MCMC.T
        self.results['accepted'] = 100*self.accepted/self.nsamples

        return self.results
