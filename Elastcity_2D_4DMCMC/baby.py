import utilities
import numpy as np
import scipy as sp

class Baby_mcmc():
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
        self.s = inp['s']
        self.mav = np.mean(abs(self.observations - np.mean(self.observations)))

        self.FC = 1000

        self.Rj = sp.linalg.cholesky(self.initial_cov)
        self.dim = np.size(self.range, 1)

        self.MCMC = np.zeros([self.nsamples, self.dim])
        self.oldpi, self.oldvalue = utilities.ESS(self.observations, self.initial_theta, self.mesh)
        self.results = {}

        self.results['values'] = [self.oldvalue*1]
        self.results['MCMC'] = np.zeros([self.dim, self.nsamples])
        self.results['MCMC'][:,0] = self.initial_theta.T

        self.accepted = 0
        self.thetaj = self.initial_theta



    def Kalman_gain(self, j):

        Y = self.Y
        ss2 = self.m0*np.ones([np.size(self.observations, 0)])
        RR = np.diag(ss2) #nel, nel *keep an eye on this*

        mX = np.repeat(np.mean(self.X, 1, keepdims = True), j-1, axis = 1) #1, j-1
        mY = np.repeat(np.mean(self.Y, 1, keepdims = True), j-1, axis = 1)#nel, j-1

        Ctm = (self.X - mX)@(Y - mY).T/(j-2)
        Cmm = (Y - mY)@(Y - mY).T/(j-2)
        KK = Ctm @ np.linalg.solve(Cmm+RR, np.eye(np.size(RR,0)))

        return KK
        
    def EnKF_go(self, j):
        self.X = self.results['MCMC'][:,:j - 1] #1 x nsamples
        self.Y = np.squeeze(self.results['values'][:-1]).T #nsamples x nel
        self.thetaj = self.results['MCMC'][:,j - 1].reshape((self.dim , 1))

        KK = self.Kalman_gain(j)
            
        XX = utilities.forward_model(self.thetaj, self.mesh)
        dt = KK @ (self.observations+ np.random.normal(size = np.shape(self.observations))*self.m0 - XX)

        thetas = self.thetaj + dt
        thetas = utilities.check_bounds(thetas, self.range)

        newpi, newvalue = utilities.ESS(self.observations, thetas, self.mesh)

        lam = min(1, np.exp(-0.5*(newpi - self.oldpi)/self.sigma))

        if np.random.uniform(0, 1) < lam:
            self.accepted += 1
            self.thetaj = thetas
            self.oldpi = newpi
            self.oldvalue = newvalue

        self.results['values'].append(self.oldvalue)
        self.results['MCMC'][:,j] = self.thetaj.T

    
    def MH_go(self ,j):
        step = np.zeros((self.dim, 1))

        Inde = np.random.choice(range(self.dim))
        Rand = np.random.normal()
                    
        step[Inde, 0] = Rand

        thetas = self.thetaj + self.Rj@step

        thetas = utilities.check_bounds(thetas, self.range)
        
        newpi, newvalue = utilities.ESS(self.observations, thetas, self.mesh)
        lam = min(1, np.exp(-0.5*(newpi - self.oldpi)/self.sigma))

        if np.random.uniform(0, 1) < lam:
            self.accepted += 1
            self.thetaj = thetas
            self.oldpi = newpi
            self.oldvalue = newvalue

        else:
            step = np.zeros((self.dim, 1))
            step[Inde, 0] = np.random.normal() / 4

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
        self.results['MCMC'][:,j] = self.thetaj.T

    def Crank_go(self, j):
        thetas = (1 - self.s**2)**0.5 * self.thetaj + self.Rj@ np.random.normal(size=[self.dim, 1])

        thetas = utilities.check_bounds(thetas, self.range)
        
        newpi, newvalue = utilities.ESS(self.observations, thetas, self.mesh)
        lam = min(0, -0.5*(newpi - self.oldpi)/self.sigma)

        if np.log(np.random.uniform(0, 1)) < lam:
            self.accepted += 1
            self.thetaj = thetas
            self.oldpi = newpi
            self.oldvalue = newvalue

        self.results['values'].append(self.oldvalue)
        self.results['MCMC'][:,j] = self.thetaj.T

    def FMH_go(self, j, N):
        if j % N <= N//4 :
            f = 0
        elif j % N <= N//2 :
            f = 1
        elif j % N <= 3*N//4:
            f = 2
        else: 
            f = 3
        F = np.array([0, 1, 2, 3])
        step = np.zeros((self.dim, 1))

        Rand = np.random.normal()

        step[F[f], 0] = Rand 

        thetas = self.thetaj + self.Rj@step

        thetas = utilities.check_bounds(thetas, self.range)
        
        newpi, newvalue = utilities.ESS(self.observations, thetas, self.mesh)
        lam = min(1, np.exp(-0.5*(newpi - self.oldpi)/self.sigma))

        if np.random.uniform(0, 1) < lam:
            self.accepted += 1
            self.thetaj = thetas
            self.oldpi = newpi
            self.oldvalue = newvalue

        else:
            step = np.zeros((self.dim, 1))
            step[F[f], 0] = np.random.normal() / 4

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
        self.results['MCMC'][:,j] = self.thetaj.T

    def Baby_go(self):
        j = 1
        R = 0
        F = True
        N = 100
        rotation = np.array([0,1,2,3])
        while j < self.nsamples:
            if j % N == 0 and F == True:
                R += 1
                R = R % len(rotation)
            if rotation[R] == 0:
                self.MH_go(j)
            elif rotation[R] == 1:
                self.FMH_go(j , N)
            elif rotation[R] == 2:
                self.EnKF_go(j)
            elif rotation[R] == 3:
                self.Crank_go(j)

            if self.nsamples - j <= self.FC:
                F = False
                R = 2

            if j % 100 == 0:
                print(f'{j} Samples Completed:')
                print('The median of the Youngs Modulus 1 posterior is: %f, with uncertainty +/- %.5f' % (np.median(self.results['MCMC'][0][:j]), np.sqrt(np.var(self.results['MCMC'][0][:j]))))
                print('The median of the Youngs Modulus 2 posterior is: %f, with uncertainty +/- %.5f' % (np.median(self.results['MCMC'][1][:j]), np.sqrt(np.var(self.results['MCMC'][1][:j]))))
                print('The median of the Poissons Ratio 1 posterior is: %f, with uncertainty +/- %.5f' % (np.median(self.results['MCMC'][2][:j]), np.sqrt(np.var(self.results['MCMC'][2][:j]))))
                print('The median of the Poissons Ratio 2 posterior is: %f, with uncertainty +/- %.5f' % (np.median(self.results['MCMC'][3][:j]), np.sqrt(np.var(self.results['MCMC'][3][:j]))))
                print()

            j += 1

        self.results['accepted'] = (self.accepted/self.nsamples)*100

        return self.results
            

