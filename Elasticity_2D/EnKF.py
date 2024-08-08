import numpy as np
from DRAM import DRAM_algorithm
import utilities as utilities
import pandas as pd

class EnKF_mcmc():
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

        self.s = self.nsamples  #maybe need deepcopy
        self.nsamples = self.K0 - 1
        inp['nsamples'] = self.nsamples
        A = DRAM_algorithm(inp)
        self.results = A.DRAM_go()

        self.X = self.results['MCMC'] #1 x nsamples
        self.Y = np.squeeze(self.results['values']).T #nsamples x nel
        self.thetaj = self.X[:, self.K0 - 2].reshape(-1,1)

        self.oldpi, self.oldvalue = utilities.ESS(self.observations, self.thetaj, self.mesh)
        self.accepted = np.fix(self.results['accepted']*(self.K0 - 1)/100)

    def Kalman_gain(self, j):
        ss2 = self.m0**2*np.ones([np.size(self.observations, 0)])
        RR = np.diag(ss2) #nel, nel *keep an eye on this*

        mX = np.repeat(np.mean(self.X, 1, keepdims = True), j-1, axis = 1) #1, j-1
        mY = np.repeat(np.mean(self.Y, 1, keepdims = True), j-1, axis = 1)#nel, j-1

        Ctm = (self.X - mX)@(self.Y - mY).T/(j-2)
        Cmm = (self.Y - mY)@(self.Y - mY).T/(j-2)
        KK = Ctm @ np.linalg.solve(Cmm+RR, np.eye(np.size(RR,0)))

        return KK
    
    # def save_data(self):
    #     data = {}

    #     if self.dim == 2:
    #         data['E'] = self.thetaj[0]
    #         data['v'] = self.thetaj[1]
    #     else:
    #         data['E1'] = self.thetaj[0]
    #         data['v1'] = self.thetaj[1]
    #         data['E2'] = self.thetaj[2]
    #         data['v2'] = self.thetaj[3]
        
    #     for i in range(len(self.oldvalue)):
    #         data[i] = self.oldvalue[i]

    #     df = pd.DataFrame(data)

    #     df.to_csv(r'C:\Users\ashle\Documents\GitHub\Portfolio\ES98C\Elasticity_2D\EnKF.csv', mode='a', index=False, header = False)

    def EnKF_go(self):
        j = self.K0
        while j < self.s:
            KK = self.Kalman_gain(j)
            
            dt = KK @ (self.observations + np.random.normal(0, self.m0, size = np.shape(self.observations)) - self.oldvalue)
            thetas = self.thetaj + dt

            thetas = utilities.check_bounds(thetas, self.range)

            newpi, newvalue = utilities.ESS(self.observations, thetas, self.mesh)

            lam = min(0, -0.5*(newpi - self.oldpi)/self.sigma)

            if np.log(np.random.uniform(0, 1)) < lam:
                self.accepted += 1
                self.thetaj = thetas
                self.oldpi = newpi
                self.oldvalue = newvalue

            # self.save_data()

            tempX = np.zeros([np.size(self.X,0), np.size(self.X,1) + 1])
            tempY = np.zeros([np.size(self.Y,0), np.size(self.Y,1) + 1])
            
            tempX[:,:-1] = self.X*1
            tempY[:,:-1] = self.Y*1
            tempX[:,-1] = self.thetaj.T
            tempY[:,-1] = np.squeeze(self.oldvalue) #keep an eye on this for more dims

            self.X = tempX
            self.Y = tempY

            j += 1
        
        self.results['MCMC'] = self.X
        self.results['accepted'] = (self.accepted/self.s)*100

        return self.results