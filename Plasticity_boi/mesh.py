import numpy as np
import utilities as utilities
import matplotlib.pyplot as plt

class Mesh():
    # the initialisation procedure
    def __init__(self, m):
        self.m = m
        quad, self.scfa, NC, ENN, NE, FM, FN, FBCN, EID, self.th = self.m
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
                    
            self.XYZ = self.XYZ*self.scfa
            # NODE NUMBERS FOR ELEMENTS 
            self.nel = nelx*nely                              # total number of elements in the domain
            self.CON = np.zeros((self.nel,4), dtype=int)           # [nel*4] array of node number for each element
            for i in range(nely):                        # loop over elements in the vertical direction 
                for j in range(nelx):                    # loop over elements in the horizontal direction 
                    # element 'el' and corresponding node numbers
                    self.CON[j+i*nelx, :] = [j+i*self.nnodesx, j+i*self.nnodesx+1,j+(i+1)*self.nnodesx+1, j+(i+1)*self.nnodesx] 

        else:
            self.XYZ = np.array(NC)*self.scfa
            self.CON = np.array(ENN)
            self.nel = NE
            self.DOF = np.zeros((self.nel,2*4), dtype=int)

        self.DOF = np.zeros((self.nel,2*4), dtype=int)
        for i in range(self.nel):
            # defines single row of DOF for each element 'i'
            self.DOF[i,:] = [self.CON[i,0]*2, self.CON[i,1]*2-1, self.CON[i,1]*2, \
                            self.CON[i,1]*2+1, self.CON[i,2]*2, self.CON[i,2]*2+1, \
                            self.CON[i,3]*2, self.CON[i,3]*2+1]

        self.DOF2 = np.zeros((self.nnodesy, 2*self.nnodesx), dtype=int)
        for i in range(self.nnodesy):
            self.DOF2[i,:] = 2*i*self.nnodesx + np.array(range(2*self.nnodesx))

        self.ndofs = 2 * len(self.XYZ)
        bcn = [self.DOF[0][0], self.DOF[0][1], self.DOF[0][3]]
        for i in range(len(self.DOF))[1:]:
            bcn.append(self.DOF[i][0])

        self.BCnodes = np.array(bcn)

        frn = []
        for i in range(len(self.DOF2[0])//2):
            frn.append(self.DOF2[-1][2*i + 1])
        self.forcenodes = np.array(frn)
        self.forces = 0.174*np.array([0.5, 1, 1, 1, 1, 0.5])

    def meshgrid(self):
        print(self.XYZ, self.CON + 1, self.scfa, self.th, self.BCnodes + 1, self.forcenodes + 1, self.forces) 
        return self.XYZ, self.CON + 1, self.scfa, self.th, self.BCnodes + 1, self.forcenodes + 1, self.forces
