import numpy as np

def forawrd_model(x):
    #e=4.259
    a = 1
    que = 12
    etr = 0.5
    l = 1
    nel = 1000
    d1,dl1,sig1,step1,nel1 = linelast(nel,x,a,que,etr,l)
    return d1

def linelast(nel,e,a,que,etr,l):
    ea = e*a

    ng = nel+1

    d = np.zeros(nel+1, 1)

    dl  = l/nel

    ke = ea/dl*np.array([[1, -1],[-1, 1]])

    q = np.zeros(ng, 1)
    kg = np.zeros(ng)

    for i in range(0, nel):
        x1 = (i-1)*dl
        x2 = i*dl
        for j in range(2):
            for k in range(2):
                kg[j+i-1, k+i-1] = kg[j+i-1, k+i-1] +ke[j, k]
            if j==1:
                q[i] = q[i] + (l*np.cos(2*np.pi*x1/l)/(2*np.pi) -l**2*np.sin(2*np.pi*x2/l)/(4*np.pi**2*dl) + l**2*np.sin(2*np.pi*x1/l)/(4*np.pi**2*dl))*que
            elif j == 2:
                q[i+1] = q[i+1]+(-l*np.cos(2*np.pi*x2/l)/(2*np.pi)+l**2*np.sin(2*np.pi*x2/l)/(4*np.pi**2*dl)-l**2*np.sin(2*np.pi*x1/l)/(4*np.pi**2*dl))*que
    
    kg[0,0] = 1
    kg[0,1:] = np.zeros(1, nel)
    kg[0, 0] = 1
    kg[1:, 0] = np.zeros(nel, 1)
    q[0] = 0

    q[ng] = q[ng] + etr*a

    d[0:ng, 0] = kg**-q

    sig = np.zeros(2*nel, 2*nel)

    step = np.zeros(2*nel)
    for i in range(nel):
        sig[2*i-1, :] = e/dl * (d[i+1]-d[i])
        sig[2*i,:] = e/dl*(d[i+1 - d[i]])
        step[2*i-1] = dl*(i-1)
        step[2*i] = dl*i 

    return d, dl, sig, step, nel