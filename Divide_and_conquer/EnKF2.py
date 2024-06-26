import numpy as np
from DRAM import DRAM_algorithm
from MH import MH_mcmc
from crank import Crank_mcmc
import utilities as utilities

class EnKF_mcmc_2():
    def __init__(self, inp, results):

        self.range = inp['range']
        self.nsamples = inp['nsamples']
        self.initial_cov = inp['icov']
        self.initial_theta = inp['theta0']
        self.sigma = inp['sigma']
        self.observations = inp['measurement']
        self.K0 = np.size(results['MCMC'],1) + 1
        self.mesh = inp['mesh']
        self.m0 = inp['me']*inp['mesh'][1]
        self.adpt = inp['adapt']
        self.delay = inp['delay']

        self.s = self.nsamples + self.K0
        self.results = results

        print('Starting EnKF 2...')

        self.X = self.results['MCMC'] #1 x nsamples
        self.Y = np.squeeze(self.results['values']).T #nsamples x nel
        self.thetaj = self.X[:, self.K0 - 2].reshape(-1,1)

        self.oldpi, self.oldvalue = utilities.ESS(self.observations, self.thetaj, self.mesh)
        self.accepted = np.fix(self.results['accepted']*(self.K0 - 1)/100)

        self.edges_ind = []
        if self.mesh[0] != 0:
            A = range(self.mesh[0][1])
            self.edges_ind = [a for a in A]
            for i in range(self.mesh[0][0]-1):
                self.edges_ind.append(A[-1] + 1 + self.mesh[0][1]*i)
                self.edges_ind.append(A[-1] + self.mesh[0][1]*i)
            for i in range(self.mesh[0][1]):
                self.edges_ind.append(self.mesh[0][1]*self.mesh[0][0]-1 - i)

        self.meas_edge = []
        for i in self.edges_ind:
            self.meas_edge.append(2*i)
            self.meas_edge.append(2*i + 1)

    def Kalman_gain(self, j, indexs):

        ss2 = self.m0*np.ones([np.size(self.observations[indexs], 0)])
        RR = np.diag(ss2) #nel, nel *keep an eye on this*

        mX = np.repeat(np.mean(self.X, 1, keepdims = True), j-1, axis = 1) #1, j-1
        mY = np.repeat(np.mean(self.Y[indexs], 1, keepdims = True), j-1, axis = 1) #nel, j-1

        Ctm = (self.X - mX)@(self.Y[indexs] - mY).T/(j-2)
        Cmm = (self.Y[indexs] - mY)@(self.Y[indexs] - mY).T/(j-2)
        KK = Ctm @ np.linalg.solve(Cmm+RR, np.eye(np.size(RR,0)))

        return KK

    def EnKF_go(self):
        j = self.K0
        while j < self.s:
            indexs = np.random.choice(range(0, len(self.observations)//2), int(0.25*len(self.observations)//2), replace = False)
            rand_ind = []
            for i in indexs:
                rand_ind.append(2*i)
                rand_ind.append(2*i + 1)
            #index = self.meas_edge
            KK = self.Kalman_gain(j, indexs)
            
            XX = utilities.forward_model(self.thetaj, self.mesh)[indexs]
            dt = KK @ (self.observations[indexs] + np.random.normal(size = np.shape(self.observations[indexs]))*self.m0 - XX)

            thetas = self.thetaj + dt

            thetas = utilities.check_bounds(thetas, self.range)

            newpi, newvalue = utilities.ESS(self.observations, thetas, self.mesh)

            lam = min(1, np.exp(-0.5*(newpi - self.oldpi)/self.sigma))

            if np.random.uniform(0, 1) < lam:
                self.accepted += 1
                self.thetaj = thetas
                self.oldpi = newpi
                self.oldvalue = newvalue

            tempX = np.zeros([np.size(self.X,0), np.size(self.X,1) + 1])
            tempY = np.zeros([np.size(self.Y,0), np.size(self.Y,1) + 1])
            
            tempX[:,:-1] = self.X*1
            tempY[:,:-1] = self.Y*1
            tempX[:,-1] = self.thetaj.T
            tempY[:,-1] = np.squeeze(self.oldvalue) #keep an eye on this for more dims

            self.X = tempX
            self.Y = tempY

            if j % 100 == 0:
                print(f'{j} samples completed')
                print('The median of the Youngs Modulus 1 posterior is: %f, with uncertainty +/- %.5f' % (np.median(self.X[0]), np.sqrt(np.var(self.X[0]))))
                print('The median of the Youngs Modulus 2 posterior is: %f, with uncertainty +/- %.5f' % (np.median(self.X[1]), np.sqrt(np.var(self.X[1]))))
                print('The median of the Poissons Ratio 1 posterior is: %f, with uncertainty +/- %.5f' % (np.median(self.X[2]), np.sqrt(np.var(self.X[2]))))
                print('The median of the Poissons Ratio 2 posterior is: %f, with uncertainty +/- %.5f' % (np.median(self.X[3]), np.sqrt(np.var(self.X[3]))))
                print()

            j += 1
        
        self.results['MCMC'] = self.X
        self.results['accepted'] = (self.accepted/self.s)*100

        return self.results