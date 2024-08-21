import numpy as np
import scipy as sp
import utilities as utilities
import pandas as pd
import time

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

        self.Rj = sp.linalg.cholesky(self.initial_cov)
        self.dim = np.size(self.range, 1)

        self.MCMC = np.zeros([self.nsamples, self.dim])
        self.oldpi, self.oldvalue = utilities.ESS(self.observations, self.initial_theta, self.mesh)
        self.results = {}

        self.results['values'] = [self.oldvalue*1]

        self.accepted = 0
        self.MCMC[0,:] = self.initial_theta.T

        self.thetaj = self.initial_theta

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

        df.to_csv(r'C:\Users\ashle\Documents\GitHub\Portfolio\ES98C\Plasticity_RVE\MH.csv', mode='w', index=True)

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

        df.to_csv(r'C:\Users\ashle\Documents\GitHub\Portfolio\ES98C\Plasticity_RVE\MH.csv', mode='a', index=True, header = False)


    def MH_go(self):
        j = 1
        while j < self.nsamples:
            thetas = self.thetaj + self.Rj@np.random.normal(size = [self.dim, 1])

            thetas = utilities.check_bounds(thetas, self.range)
            
            newpi, newvalue = utilities.ESS(self.observations, thetas, self.mesh)
            lam = min(0, -0.5*(newpi - self.oldpi)/self.sigma)

            if np.log(np.random.uniform(0, 1)) < lam:
                self.accepted += 1
                self.thetaj = thetas
                self.oldpi = newpi
                self.oldvalue = newvalue

            self.results['values'].append(self.oldvalue)

            self.MCMC[j,:] = self.thetaj.T

            # if j % 100 == 0:
            #     print(f'{j} samples completed')
            #     print(100*self.accepted/j)
            #     print('The median of the Youngs Modulus posterior is: %f, with uncertainty +/- %.5f' % (np.median(self.MCMC.T[0][:j]), np.sqrt(np.var(self.MCMC.T[0][:j]))))
            #     print('The median of the Poissons Ratio posterior is: %f, with uncertainty +/- %.5f' % (np.median(self.MCMC.T[1][:j]), np.sqrt(np.var(self.MCMC.T[1][:j]))))
            #     print('The median of the Yield Stress posterior is: %f, with uncertainty +/- %.5f' % (np.median(self.MCMC.T[2][:j]), np.sqrt(np.var(self.MCMC.T[2][:j]))))
            #     print('The median of the Hardening Modulus posterior is: %f, with uncertainty +/- %.5f' % (np.median(self.MCMC.T[3][:j]), np.sqrt(np.var(self.MCMC.T[3][:j]))))
            #     print()

            self.save_data(j)
                
            j += 1

        self.results['MCMC'] = self.MCMC.T
        self.results['accepted'] = 100*self.accepted/self.nsamples

        return self.results
