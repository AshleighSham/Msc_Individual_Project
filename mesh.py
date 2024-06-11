import numpy as np
import utilities

class Mesh:
    # the initialisation procedure
    def __init__(self, bot, top, left, right):
        self.bot = bot
        self.top = top
        self.left = left
        self.right = right
        
        # number of nodes and elements in the domain
        nnodesx = len(bot)                     # number of horizontal nodes
        nnodesy = len(left)                    # number of vertical nodes 
        nelx = nnodesx-1                       # number of horizontal elements
        nely = nnodesy-1                       # number of vertical elements
        nnodes = nnodesx*nnodesy               # total number of nodes    
        
        # dimensions of the domain
        lx = bot[nnodesx-1] - bot[0]           # length of the domain in x-direction (horizontal)
        ly = left[nnodesy-1] - left[0]         # length of the domain in y-direction (vertical)
        
        # GENERATE COORDINATES OF NODES 'XYZ'
        self.XYZ = np.zeros((nnodes,2))         # two-column array [nnodes x 2] containing all nodal coordinates  
        for i in range(nnodesy):                # loop over all nodes on the vertical sides 
            yl = left[i] - left[0]              # distance between node 'i' and left-bottom node '1'
            dy = right[i] - left[i]             # distance between the corresponing nodes j on top and bottom 
            for j in range(nnodesx):            # loop over all nodes on the horizontal sides
                xb = bot[j] - bot[0]            # distance between node 'j' and bottom-left node '1' 
                dx = top[j] - bot[j]            # distance between nodes 'j' on opposite sides (top and bottom)

                x = (dx*yl+xb*ly)/(ly-dx*dy/lx) # x-coordinate (horizontal) of a node in the interior of the domain
                y = dy/lx*x+yl                  # y-coordinate (vertical) of a node in the interior of the domain

                self.XYZ[j+i*nnodesx, 0] = x + bot[0]  # coordinate 'x' in the global coordinate system 
                self.XYZ[j+i*nnodesx, 1] = y + left[0] # coordinate 'y' in the global coordinate system

        # NODE NUMBERS FOR ELEMENTS 
        nel = nelx*nely                              # total number of elements in the domain
        self.CON = np.zeros((nel,4), dtype=int)           # [nel*4] array of node number for each element
        for i in range(nely):                        # loop over elements in the vertical direction 
            for j in range(nelx):                    # loop over elements in the horizontal direction 
                # element 'el' and corresponding node numbers
                self.CON[j+i*nelx, :] = [j+i*nnodesx, j+i*nnodesx+1,j+(i+1)*nnodesx+1, j+(i+1)*nnodesx] 

        # Global DOF for each element (4-node (linear) quadrilateral element)
        self.DOF = np.zeros((nel,2*4), dtype=int)
        for i in range(nel):
            # defines single row of DOF for each element 'i'
            self.DOF[i,:] = [self.CON[i,0]*2, self.CON[i,1]*2-1, self.CON[i,1]*2, \
                             self.CON[i,1]*2+1, self.CON[i,2]*2, self.CON[i,2]*2+1, \
                             self.CON[i,3]*2, self.CON[i,3]*2+1]
    
    def eff_el(self):
        # calculate the effective elasticity of the system
        nel=len(self.CON)
        a = 1/(np.sqrt(3))
        w = 1
        Gauss = np.array([[-a, a, a, -a], [-a, -a, a, a]])

        # these are the effective elasticity matrices for each element
        # I have initialised them as attributes, incase they need to be used in later code.
        self.J_tot = 0
        self.CV_eff1 = 0
        self.CV_eff2 = 0

        for i in range(nel):
            # get the plane strain matrix for a given element
            C = utilities.plane_strain(self.youngs[i], self.poisson[i])
            
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

                                           
    def voigt(self):
        # use the formulae defined lecture notes
        return self.CV_eff1/self.J_tot
                                            
    def reuss(self):
        # use the formulae defined lecture notes
        return self.J_tot/self.CV_eff2