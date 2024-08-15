import numpy as np
from DRAM import DRAM_algorithm
import utilities as utilities
import pandas as pd

class rEnKF_mcmc():
    def __init__(self, inp):

        self.range = inp['range']
        self.nsamples = inp['nsamples']
        self.initial_cov = inp['icov']
        self.initial_theta = inp['theta0']
        self.sigma = inp['sigma']
        self.observations = inp['measurement']
        self.K0 = inp['Kalmans']
        self.mesh = inp['mesh']
        self.m0 = inp['me']
        self.adpt = inp['adapt']

        self.s = self.nsamples  #maybe need deepcopy
        self.nsamples = self.K0 - 1
        inp['nsamples'] = self.nsamples
        A = DRAM_algorithm(inp)
        self.results = A.DRAM_go()

        self.X = self.results['MCMC'] #1 x nsamples
        # print('Values before the EnKF')
        # print('The median of the Youngs Modulus 1 posterior is: %f, with uncertainty +/- %.5f' % (np.median(self.X[0]), np.sqrt(np.var(self.X[0]))))
        # print('The median of the Youngs Modulus 2 posterior is: %f, with uncertainty +/- %.5f' % (np.median(self.X[1]), np.sqrt(np.var(self.X[1]))))
        # print('The median of the Poissons Ratio 1 posterior is: %f, with uncertainty +/- %.5f' % (np.median(self.X[2]), np.sqrt(np.var(self.X[2]))))
        # print('The median of the Poissons Ratio 2 posterior is: %f, with uncertainty +/- %.5f' % (np.median(self.X[3]), np.sqrt(np.var(self.X[3]))))
        # print()
        self.Y = np.squeeze(self.results['values']).T #nsamples x nel
        self.thetaj = self.X[:, self.K0 - 2].reshape(-1,1)

        self.oldpi, self.oldvalue = utilities.ESS(self.observations, self.thetaj, self.mesh)
        self.accepted = np.fix(self.results['accepted']*(self.K0 - 1)/100)

    def Kalman_gain(self, j):

        ss2 = self.m0**2*np.ones([np.size(self.observations, 0)])
        RR = np.diag(ss2) #nel, nel *keep an eye on this*

        mX = np.repeat(np.mean(self.X, 1, keepdims = True), j-1, axis = 1) #1, j-1
        mY = np.repeat(np.mean(self.Y, 1, keepdims = True), j-1, axis = 1) #nel, j-1

        Ctm = ((self.X - mX)@(self.Y - mY).T/(j-2))
        Cmm = ((self.Y - mY)@(self.Y - mY).T/(j-2)).diagonal()
        A = np.linalg.solve(np.sqrt(Cmm + RR), np.eye(np.size(RR,0))).T
        B = np.linalg.solve(np.sqrt(Cmm + RR) + np.sqrt(RR), np.eye(np.size(RR,0)))
        KK = Ctm @ (A @ B)
        return KK

    def rEnKF_go(self):
        j = self.K0
        while j < self.s:
            KK = self.Kalman_gain(j)

            dt = KK @ (- self.oldvalue + self.observations + np.random.normal(0, self.m0, size = np.shape(self.observations)))

            thetas = self.thetaj + dt

            thetas = utilities.check_bounds(thetas, self.range)

            newpi, newvalue = utilities.ESS(self.observations, thetas, self.mesh)

            lam = min(0, -0.5*(newpi - self.oldpi)/self.sigma)

            if np.log(np.random.uniform(0, 1)) < lam:
                self.accepted += 1
                self.thetaj = thetas
                self.oldpi = newpi
                self.oldvalue = newvalue

            #self.save_data(j)

            tempX = np.zeros([np.size(self.X,0), np.size(self.X,1) + 1])
            tempY = np.zeros([np.size(self.Y,0), np.size(self.Y,1) + 1])
            
            tempX[:,:-1] = self.X*1
            tempY[:,:-1] = self.Y*1
            tempX[:,-1] = self.thetaj.T
            tempY[:,-1] = np.squeeze(self.oldvalue) #keep an eye on this for more dims

            self.X = tempX
            self.Y = tempY

#            if j % 200 == 0:
                # print(f'{j} samples completed')
                # print('The median of the Youngs Modulus 1 posterior is: %f, with uncertainty +/- %.5f' % (np.median(self.X[0]), np.sqrt(np.var(self.X[0]))))
                # print('The median of the Youngs Modulus 2 posterior is: %f, with uncertainty +/- %.5f' % (np.median(self.X[1]), np.sqrt(np.var(self.X[1]))))
                # print('The median of the Poissons Ratio 1 posterior is: %f, with uncertainty +/- %.5f' % (np.median(self.X[2]), np.sqrt(np.var(self.X[2]))))
                # print('The median of the Poissons Ratio 2 posterior is: %f, with uncertainty +/- %.5f' % (np.median(self.X[3]), np.sqrt(np.var(self.X[3]))))
                # print()

            j += 1
        
        self.results['MCMC'] = self.X
        self.results['accepted'] = (self.accepted/self.s)*100

        return self.results