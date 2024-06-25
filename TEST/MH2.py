import numpy as np
import scipy as sp
import utilities as utilities

class MH2_mcmc:
    def __init__(self, inp):

        self.inp = inp
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
        self.delay = inp['delay']
        self.freeze = inp['freeze']
        self.priority = inp['Priority']

        self.Rj = sp.linalg.cholesky(self.initial_cov)
        self.dim = np.size(self.range, 1)

        self.MCMC = np.zeros([self.nsamples, self.dim])
        self.oldpi, self.oldvalue = utilities.ESS(self.observations, self.initial_theta, self.mesh)
        self.results = {}

        self.results['values'] = [self.oldvalue*1]

        self.accepted = 0
        self.MCMC[0,:] = self.initial_theta.T

        self.thetaj = self.initial_theta

    def MH2_go(self):
        j = 1
        while j < self.delay:
            step = np.zeros((self.dim, 1))
            step[np.random.choice(range(4), 1 , p = self.priority),0] = np.random.normal()

            thetas = self.thetaj + self.Rj@step

            thetas = utilities.check_bounds(thetas, self.range)
            
            newpi, newvalue = utilities.ESS(self.observations, thetas, self.mesh)
            lam = min(1, np.exp(-0.5*(newpi - self.oldpi)/self.sigma))

            if np.random.uniform(0, 1) < lam:
                self.accepted += 1
                self.thetaj = thetas
                self.oldpi = newpi
                self.oldvalue = newvalue

            self.results['values'].append(self.oldvalue)

            self.MCMC[j,:] = self.thetaj.T
            j += 1


        F = np.array([0,2,1,3])
        while j < self.nsamples:
            for i in range(self.dim):
                means = np.mean(self.MCMC[j-self.freeze:j,:].T, axis = 1, keepdims = True)
                self.inp['nsamples'] = self.freeze
                self.inp['theta0'] = self.MCMC[j-1,F[i]]
                T = self.MH2_5_go(F[i], means)
                self.MCMC[j:j+self.freeze,F[i]] = T

            j += self.freeze

        for q in range(self.delay, self.nsamples):
            newpi, newvalue = utilities.ESS(self.observations, self.MCMC[q], self.mesh)
            self.results['values'].append(newvalue)

        self.results['MCMC'] = self.MCMC.T
        self.results['accepted'] = 100*self.accepted/self.nsamples

        return self.results
    
    def MH2_5_go(self, inde, means):
        range = self.inp['range']
        nsamples = self.inp['nsamples']
        initial_cov = self.inp['icov']
        initial_theta = means
        initial_theta[inde] = self.inp['theta0']
        sigma = self.inp['sigma']
        observations = self.inp['measurement']
        K0 = self.inp['Kalmans']
        m0 = self.inp['me']

        Rj = sp.linalg.cholesky(initial_cov)
        dim = np.size(range, 1)

        r = np.zeros([nsamples])
        oldpi, _ = utilities.ESS(observations, initial_theta, self.inp['mesh'])

        accepted = 0
        r[0] = initial_theta[inde]

        thetaj = initial_theta

        j = 1
        while j < nsamples:
            step = np.zeros((self.dim, 1))
            step[inde] = np.random.normal()

            thetas = thetaj + Rj@step
            thetas = utilities.check_bounds(thetas, range)

            newpi, _ = utilities.ESS(observations, thetas, self.inp['mesh'])
            lam = min(1, np.exp(-0.5*(newpi - oldpi)/sigma))

            if np.random.uniform(0, 1) < lam:
                accepted += 1
                thetaj = thetas
                oldpi = newpi
            
            r[j] = thetaj[inde]
            j += 1

        return r

