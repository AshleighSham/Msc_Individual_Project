import numpy as np
import scipy as sp
import utilities as utilities
import pandas as pd

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
        self.mesh = inp['mesh']
        self.adpt = inp['adapt']

        self.results ={}

        self.Rj = sp.linalg.cholesky(self.initial_cov)
        self.dim = np.size(self.range, 1)

        self.R2 = self.Rj*0.2
        self.invR = np.linalg.solve(self.Rj, np.eye(len(self.Rj[0])))

        self.MCMC = np.zeros([self.nsamples, self.dim])
        self.oldpi, self.oldvalue = utilities.ESS(self.observations, self.initial_theta, self.mesh)

        self.accepted = 0
        self.MCMC[0,:] = self.initial_theta.T
        self.thetaj = self.initial_theta

        self.results = {}
        self.results['values'] = [self.oldvalue*1]

        data = {}

        data['Ei'] = self.thetaj[0]
        data['vi'] = self.thetaj[1]
        data['syi'] = self.thetaj[2]
        data['Hi'] = self.thetaj[3]
        data['Em'] = self.thetaj[4]
        data['vm'] = self.thetaj[5]
        data['sym'] = self.thetaj[6]
        data['Hm'] = self.thetaj[7]
        
        for i in range(len(self.oldvalue)):
            data[i] = self.oldvalue[i]

        df = pd.DataFrame(data)

        df.to_csv(r'C:\Users\ashle\Documents\GitHub\Portfolio\ES98C\Plasticity_RVE\MH_DR.csv', mode='w', index=True)

    def save_data(self, j):
        data = {}

        data['Ei'] = self.thetaj[0]
        data['vi'] = self.thetaj[1]
        data['syi'] = self.thetaj[2]
        data['Hi'] = self.thetaj[3]
        data['Em'] = self.thetaj[4]
        data['vm'] = self.thetaj[5]
        data['sym'] = self.thetaj[6]
        data['Hm'] = self.thetaj[7]
        
        for i in range(len(self.oldvalue)):
            data[i] = self.oldvalue[i]

        df = pd.DataFrame(data)

        df.to_csv(r'C:\Users\ashle\Documents\GitHub\Portfolio\ES98C\Plasticity_RVE\MH_DR.csv', mode='a', index=True, header = False)

    def MH_DR_go(self):
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

            self.save_data(j)

            j += 1

        self.results['MCMC'] = self.MCMC.T
        self.results['accepted'] = 100*self.accepted/self.nsamples

        return self.results
 