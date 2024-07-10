import numpy as np
import utilities as utilities
import matplotlib.pyplot as plt

class Mesh():
    # the initialisation procedure
    def __init__(self, m):
        self.m = m
        quad, scfa, NC, ENN, NE, FM, FN, FBCN, eID, self.th = self.m
        if quad != 0:
            bot = np.array([x for x in range(quad[1])])
            top = np.array([x for x in range(quad[1])])
            left = np.array([x for x in range(quad[0])])
            right = np.array([x for x in range(quad[0])])
            
            # number of nodes and elements in the domain
            self.nnodesx = len(bot)                     # number of horizontal nodes
            self.nnodesy = len(left)                    # number of vertical nodes 
            nelx = self.nnodesx-1                       # number of horizontal elements
            nely = self.nnodesy-1                       # number of vertical elements
            nnodes = self.nnodesx*self.nnodesy               # total number of nodes    
            
            # dimensions of the domain
            lx = bot[self.nnodesx-1] - bot[0]           # length of the domain in x-direction (horizontal)
            ly = left[self.nnodesy-1] - left[0]         # length of the domain in y-direction (vertical)
            
            # GENERATE COORDINATES OF NODES 'XYZ'
            self.XYZ = np.zeros((nnodes,2))         # two-column array [nnodes x 2] containing all nodal coordinates  
            for i in range(self.nnodesy):                # loop over all nodes on the vertical sides 
                yl = left[i] - left[0]              # distance between node 'i' and left-bottom node '1'
                dy = right[i] - left[i]             # distance between the corresponing nodes j on top and bottom 
                for j in range(self.nnodesx):            # loop over all nodes on the horizontal sides
                    xb = bot[j] - bot[0]            # distance between node 'j' and bottom-left node '1' 
                    dx = top[j] - bot[j]            # distance between nodes 'j' on opposite sides (top and bottom)

                    x = (dx*yl+xb*ly)/(ly-dx*dy/lx) # x-coordinate (horizontal) of a node in the interior of the domain
                    y = dy/lx*x+yl                  # y-coordinate (vertical) of a node in the interior of the domain

                    self.XYZ[j+i*self.nnodesx, 0] = x + bot[0]  # coordinate 'x' in the global coordinate system 
                    self.XYZ[j+i*self.nnodesx, 1] = y + left[0] # coordinate 'y' in the global coordinate system
                    
            self.XYZ = self.XYZ*scfa
            
            # NODE NUMBERS FOR ELEMENTS 
            self.nel = nelx*nely                              # total number of elements in the domain
            self.CON = np.zeros((self.nel,4), dtype=int)           # [nel*4] array of node number for each element
            for i in range(nely):                        # loop over elements in the vertical direction 
                for j in range(nelx):                    # loop over elements in the horizontal direction 
                    # element 'el' and corresponding node numbers
                    self.CON[j+i*nelx, :] = [j+i*self.nnodesx, j+i*self.nnodesx+1,j+(i+1)*self.nnodesx+1, j+(i+1)*self.nnodesx] 

        else:
            self.XYZ = scfa*np.array(NC)
            self.CON = np.array(ENN)
            self.nel = NE
            self.DOF = np.zeros((self.nel,2*4), dtype=int)

        self.IDS = np.array(eID)

            

        self.DOF = np.zeros((self.nel,2*4), dtype=int)
        for i in range(self.nel):
            # defines single row of DOF for each element 'i'
            self.DOF[i,:] = [self.CON[i,0]*2, self.CON[i,1]*2-1, self.CON[i,1]*2, \
                            self.CON[i,1]*2+1, self.CON[i,2]*2, self.CON[i,2]*2+1, \
                            self.CON[i,3]*2, self.CON[i,3]*2+1]

        self.ndofs = 2 * len(self.XYZ)

        self.force = FM
        self.forcenodes = np.array(FN)
        self.BCnodes = np.array(FBCN)
    
    def eff_el(self):
        # calculate the effective elasticity of the system
        a = 1/(np.sqrt(3))
        w = 1
        Gauss = np.array([[-a, a, a, -a], [-a, -a, a, a]])

        # these are the effective elasticity matrices for each element
        # I have initialised them as attributes, incase they need to be used in later code.
        self.J_tot = 0
        self.CV_eff1 = 0
        self.CV_eff2 = 0

        for i in range(self.nel):
            # get the plane strain matrix for a given element
            C = utilities.plane_stress(self.youngs[i], self.poisson[i])
            
            # get the nodal coordinates for a single element
            xyze = self.XYZ[self.CON[i,:],]
            J_sum = 0
                        
            for j in range(4):                     # loop over each element integration points
                xi  = Gauss[0,j]                   # natural coordinate - horizontal  
                eta = Gauss[1,j]                   # natural coordinate - vertical           
                J = utilities.Jacobian(xyze,xi,eta)[1]          
                J_sum = J_sum + J

            # effective elasticity matricx summation
            self.CV_eff1 = self.CV_eff1 + (C*J_sum)
            self.CV_eff2 = self.CV_eff2 + np.divide(J_sum,C)
                                            
            # total jacobian                                   
            self.J_tot = self.J_tot + J_sum
                                            
        # this method terminates WITHOUT returning a value.
        # it's sole effect is to modify the state of the mesh object.

    def plane_stress(self, E,nu):
    
        D=np.zeros((3*len(E),3))                                           #elasticity matrix - DIM: [3*element number X 3]
        
        for i in range(len(E)):
            C = E[i]/(1 - nu[i]**2)
            D[i*3][0] = C
            D[i*3][1] = C * nu[i]
            D[i*3 + 1][0] = C * nu[i]
            D[i*3 + 1][1] = C
            D[i*3 + 2][2] = C * (1 - nu[i])/2
        return D

    # Function which returns global DOFs given fixed nodes, and a list of internal left boundary nodes
    def BC_fun(self):
        #DOFs
        fnDOFs = np.r_[self.BCnodes*2,self.BCnodes*2+1]

        # Get array of DOFs after removing fixed nodes
        BCleft = np.arange(self.ndofs)
        self.BC = np.setdiff1d(BCleft,fnDOFs)

    def force_vector(self):
        rnodes = self.forcenodes
        f = np.zeros((self.ndofs, 1))
        f[2*rnodes] = self.force
        self.f = f[self.BC]

    def dispstrain_B(self, xyze,xi,eta):
        natcoord = np.array([[-1, 1, 1, -1],[-1, -1, 1, 1]])    #natural nodal coordinates of a quad element

        # derivatives of shape functions w.r.t. natural coordinates 
        dNdnat = np.zeros((2,4))
        a = 0.25
        for i in range(2):
            for j in range(4):
                if i == 0:
                    if j < 2:
                        dNdnat[i][j] = a*(1 - xi)*natcoord[i][j]
                    else: 
                        dNdnat[i][j] = a*(1 + xi)*natcoord[i][j]
                if i == 1:
                    if j == 0 or j == 3:
                        dNdnat[i][j] = a*(1 - eta)*natcoord[i][j]
                    else: 
                        dNdnat[i][j] = a*(1 + eta)*natcoord[i][j]

        # element Jacobian matrix
        Jmat = np.dot(dNdnat,xyze)
        J = np.linalg.det(Jmat)                          #determinant of the Jacobian
                    
        JmatInv = np.linalg.solve(Jmat, np.identity(len(Jmat)))                   #inverse of the Jacobian matrix
        dNdx = np.dot(JmatInv,dNdnat)                   #effectively: Jmat^-1 * dNdna
                    
        #displacement-strain matrix 
        #linear QUAD element
        dsB=np.zeros((3,8))                  #[3 strain components X 8 DOFs]
        
        for i in range(3):
            for j in range(8):
                if j % 2 + i == 0:
                    dsB[i][j] = dNdx[0][j // 2]
                if i + j % 2 == 2:
                    dsB[i][j] = dNdx[1][j // 2]
                if i == 2:
                    if j % 2 == 0:
                        dsB[i][j] = dsB[1][j + 1]
                    else:
                        dsB[i][j] = dsB[0][j - 1]

        return dsB, J

    def keval(self, xyze,De, th):
        ke = np.zeros((8,8))                                # create element stiffness matrix (array)                    

        a = 1/(np.sqrt(3))                                  # location of Gauss points (in natural coordinates)
        w = 1                                               # weights
        Gauss = np.array([[-a, a, a, -a],[-a, -a, a, a]])   # Gauss points matrix
            
        for i in range(4):                                # introduce natural coordinates 
            xi, eta = Gauss[:,i]                                 # natural coordinate - horizontal  
            #eta = Gauss[1,i]                                # natural coordinate - vertical 
            
            dsB, J = self.dispstrain_B(xyze,xi,eta);           # evaluate dsB matrix and Jacobian

            # YOUR CODE HERE - Update element stiffness matrix
            ke += th * w * w * dsB.T @ De @ dsB * J
            
        return ke 
    
    def K_matrix(self, D):                        #number of elements
        K = np.zeros((self.ndofs,self.ndofs))

        for i in range(self.nel):
            idi = self.DOF[i,:]                      # IDs of DOFs
            xyze = self.XYZ[self.CON[i,:], :]             # element coordinates
            De = D[3*i:3*i+3, 0:3]              # elasticity matrix

            ke = self.keval(xyze,De, self.th)             # call function evaluating element stiffness matrix
            
            K[np.ix_(idi, idi)] += ke

        self.K = K[np.ix_(self.BC,self.BC)] # Pin fixed nodes

    def displacement(self, Ea, nua):
        E = np.array([Ea[i] for i in self.IDS])
        nu = np.array([nua[i] for i in self.IDS])
        D = self.plane_stress(E, nu)
        self.BC_fun()
        self.force_vector()
        self.K_matrix(D)
        dm = np.linalg.solve(self.K, self.f)
        d = np.zeros((self.ndofs, 1))
        d[self.BC] = dm
        self.d = d
        return d*1000

    def plot_fun(self):
        ax, fig = plt.subplots()
        plt.plot(self.XYZ[:, 0], self.XYZ[:, 1], 'sk')
        for i in range(len(self.CON)):
            plt.fill(self.XYZ[self.CON[i, :], 0], self.XYZ[self.CON[i, :], 1], edgecolor='k', fill=False)
        plt.show()
        
    def deformation_plot(self, label, colour, ch, ax, lines, ls, D = np.array([1]), non = True):
        if len(D) == 1:
            D = self.d
        ccc1=np.array(self.XYZ[:,0])
        ccc2=np.array(D[0:len(D):2]).reshape(-1)
        ccc= np.array(ccc1+ccc2) 

        ddd1=np.array(self.XYZ[:,1])
        ddd2=np.array(D[1:len(D):2]).reshape(-1)
        ddd= np.array(ddd1+ddd2)

        #figure = plt.figure()
        if non == True:
            ax.plot(self.XYZ[:,0], self.XYZ[:, 1],'sk', markersize='6', zorder = 1, alpha = 0.6)
        ax.scatter(self.XYZ[:,0] + D[0:len(D):2].reshape(-1), self.XYZ[:,1] + D[1:len(D):2].reshape(-1), c = colour ,s=60, label = label, zorder = 5, alpha = ch)
        #plt.title(title)

        for i in range(len(self.CON)):
            if non == True:
                ax.fill(self.XYZ[self.CON[i, :], 0], self.XYZ[self.CON[i, :], 1], edgecolor='k', fill=False, zorder = 1, alpha = 0.6)
            ax.fill(self.XYZ[self.CON[i, :], 0] + ccc2[(self.CON[i, :])], self.XYZ[self.CON[i, :], 1] + ddd2[(self.CON[i, :])], edgecolor = colour, linestyle = ls, fill=False, zorder = 5, alpha = ch, linewidth = 3)

        ax.set_aspect('equal')

    def contour_plot(self, ver, f, a):
        ccc2=np.array(self.d[0:len(self.d):2]).reshape(-1) #deformation x

        ddd2=np.array(self.d[1:len(self.d):2]).reshape(-1) #deformation y

        X, Y = [], []
        j = self.XYZ[0][1]

        while j <= self.XYZ[-1][1]:
            x, y = [], []
            i = 0
            while i < len(self.XYZ):
                if self.XYZ[i][1] == j:
                    x.append(self.XYZ[i][0])
                    y.append(self.XYZ[i][1])
                elif i > 1 and self.XYZ[i - 1][1] == j:
                    break
                i += 1
            X.append(x)
            Y.append(y)
            if i == len(self.XYZ):
                break
            j = self.XYZ[i][1]
        X = np.array(X)
        Y = np.array(Y)

        a0 = a[0].contourf(X, Y, ccc2.reshape(np.shape(X)), 40)
        a1 = a[1].contourf(X, Y, ddd2.reshape(np.shape(X)), 40)

        f.colorbar(a0, ax=a[0], shrink=0.9)
        f.colorbar(a1, ax=a[1], shrink=0.9)

        a[0].set(title = f'{ver} Displacement in x direction', aspect="equal")
        a[1].set(title = f'{ver} Displacement in y direction', aspect="equal")

    def error_plot(self, Dis, f, a):
        
        EstimateX=np.array(self.d[0:len(self.d):2]).reshape(-1)*1 #deformation x

        EstimateY=np.array(self.d[1:len(self.d):2]).reshape(-1)*1 #deformation y

        TrueX = np.array(Dis[0:len(Dis):2]).reshape(-1)
        
        TrueY = np.array(Dis[1:len(Dis):2]).reshape(-1)

        DiffX = 100*abs(np.divide(TrueX - EstimateX, TrueX + 1e-20))

        DiffY = 100*abs(np.divide(TrueY - EstimateY, TrueY + 1e-20))

        X, Y = [], []
        j = self.XYZ[0][1]

        while j <= self.XYZ[-1][1]:
            x, y = [], []
            i = 0
            while i < len(self.XYZ):
                if self.XYZ[i][1] == j:
                    x.append(self.XYZ[i][0])
                    y.append(self.XYZ[i][1])
                elif i > 1 and self.XYZ[i - 1][1] == j:
                    break
                i += 1
            X.append(x)
            Y.append(y)
            if i == len(self.XYZ):
                break
            j = self.XYZ[i][1]
        X = np.array(X)
        Y = np.array(Y)

        a0 = a[0].contourf(X, Y, DiffX.reshape(np.shape(X)), 40)
        a1 = a[1].contourf(X, Y, DiffY.reshape(np.shape(X)), 40)

        f.colorbar(a0, ax=a[0], shrink=0.9)
        f.colorbar(a1, ax=a[1], shrink=0.9)

        a[0].set(title = f'Error in  Displacement in x direction', aspect="equal")
        a[1].set(title = f'Error in Displacement in y direction', aspect="equal")

