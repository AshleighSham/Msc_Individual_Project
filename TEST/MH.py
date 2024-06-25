import numpy as np
import scipy as sp
import utilities as utilities

class MH_mcmc:
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
        self.adpt = inp['adapt']
        self.delay = inp['delay']
        self.freeze = inp['freeze']

        self.Rj = sp.linalg.cholesky(self.initial_cov)
        self.dim = np.size(self.range, 1)

        self.loops = inp['freeze loops'] * self.dim

        self.MCMC = np.zeros([self.nsamples, self.dim])
        self.oldpi, self.oldvalue = utilities.ESS(self.observations, self.initial_theta, self.mesh)
        self.results = {}

        self.results['values'] = [self.oldvalue*1]

        self.accepted = 0
        self.MCMC[0,:] = self.initial_theta.T

        self.thetaj = self.initial_theta

    def MH_go(self):
        f = np.array([-1,1])
        l = 0
        F = np.array([0,2])
        count = False
        j = 1
        while j < self.nsamples:
            if j % self.freeze == 0 and j > self.delay and l < self.loops:
                count = True
            if count == True:
                l += 1
                f += 1
                f = f % self.dim
                #print(f'Freeze: {j}, index: {F[f]}')
                TEMP = self.MCMC*1
                print('The median of the Youngs Modulus 1 posterior is: %f, with uncertainty +/- %.5f' % (np.median(TEMP.T[0][:j]), np.sqrt(np.var(TEMP.T[0][:j]))))
                print('The median of the Youngs Modulus 2 posterior is: %f, with uncertainty +/- %.5f' % (np.median(TEMP.T[1][:j]), np.sqrt(np.var(TEMP.T[1][:j]))))
                print('The median of the Poissons Ratio 1 posterior is: %f, with uncertainty +/- %.5f' % (np.median(TEMP.T[2][:j]), np.sqrt(np.var(TEMP.T[2][:j]))))
                print('The median of the Poissons Ratio 2 posterior is: %f, with uncertainty +/- %.5f' % (np.median(TEMP.T[3][:j]), np.sqrt(np.var(TEMP.T[3][:j]))))
                print()
                count = False
            step = np.zeros((self.dim, 1))
            #step[np.random.choice(range(4),1),0] = np.random.normal()
            #step[F[f],0] = np.random.normal()
            if j <= self.delay:
                step[np.array([0, 2]),0] = np.random.normal()

            if j >= self.delay + self.freeze * self.loops:
                step[np.array([0, 2]),0] = np.random.normal()

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

        self.results['MCMC'] = self.MCMC.T
        self.results['accepted'] = 100*self.accepted/self.nsamples

        return self.results
