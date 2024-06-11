import numpy as np
from DRAM import DRAM_algorithm
import utilities

class EnKF_mcmc:
    def __init__(self, inp):

        self.range = inp['range']
        self.nsamples = inp['samples']
        self.initial_cov = inp['icov']
        self.initial_theta = inp['theta0']
        self.sigma = inp['sigma']
        self.observations = inp['measurement']
        self.K0 = inp['Kalmans']
        self.m0 = inp['me']

        samples = self.nsamples  #maybe need deepcopy
        self.nsamples = self.K0 - 1
        results = DRAM_algorithm(inp)

        X = results['MCMC']
        Y = results['values']
        self.thetaj = X[:, self.nsamples]
        self.oldpi, self.oldvalue = utilities.ESS(self.observations, self.thetaj)
        self.accepted = np.fix(results['accepted']*(self.K0 - 1)/100)

        EnKF_go(samples)

        def Kalman_gain(self, j):
            ss2 = self.m0*np.ones(len(self.observations), 1)
            RR = np.diag(ss2)

            mX = np.mean(X, 2)*np.ones(j-1)
            mY = np.mean(Y, 2)*np.ones(j-1)

            Ctm = (X - mX)*(Y - mY).T/(j-2)
            Cmm = (Y - mY)*(Y - mY).T/(j-2)

            KK = Ctm/(Cmm+RR)
            
            return KK

        def EnKF_go(self, samples):
            j = self.K0
            while j < samples:
                KK = Kalman_gain(j)
                
                XX = utilities.forward_model(self.thetaj)

                dt = KK * (self.observations + np.random.normal(len(self.observations)*self.M0 - XX))
                thetas - self.thetaj + dt

                thetas = utilities.check_bounds(thetas, self.range)

                newpi, newvalue = utilities.ESS(self.observations, thetas)

                lam = min(1, np.exp(-0.5*(newpi - self.oldpi)/self.sigma))

                if np.random.uniform(0, 1) < lam:
                    self.accepted += 1
                    self.thetaj = thetas
                    self.oldpi = newpi
                    self.oldvalue = newvalue

                X.append(self.thetaj)
                Y.append(self.oldvalue)
                #make thetaj double precision??

                if j % 100 == 0:
                    print('number of sample: %d\n', j)

                j += 1
            
            self.results['MCMC'] = X
            self.results['accepted'] = (self.accepted/samples)*100

            return self.results