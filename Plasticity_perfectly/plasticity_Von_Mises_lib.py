# ======================== import system-specific parameters and functions ========================


# ========================================= import libraries ======================================
# import system
import sys

# import operating system
import os

# import numpy
import numpy as np

# import matlab plots
import matplotlib.pyplot as plt

# import math
import math

# ===================================== set up pecision ===========================================
# increase the precision
np.set_printoptions(precision=16)
#np.set_printoptions(precision=5, linewidth=200)
from decimal import Decimal, getcontext 
getcontext().prec = 16

# ===================================== functions =================================================
# ******************************** get domain geometry ********************************************
def get_domain_geo_1d(domain_bdies, n_elems):
    # geometry
    
    # get node coords
    node_coord_temp = np.linspace(domain_bdies[0], domain_bdies[1], n_elems + 1)
    node_coord = node_coord_temp.reshape(-1, 1)

    # element connectivity
    elem_conn = \
        np.array([np.array([i_elem_node + 1, i_elem_node + 2]) for i_elem_node in range(n_elems)])

    # number of nodes
    n_nodes = elem_conn[-1,-1]

    # number of degrees of freedom
    n_dofs = n_nodes

    # element degrees of freedom
    elem_dof = elem_conn

    # element spacing
    elem_spacing = node_coord[1,0] - node_coord[0,0]

    # number of spaces
    n_spaces = 1

    # get number of integration points per element
    n_integr_pts_per_elem = 2

    # get the number of integration points (Gauss points)
    n_integr_pts = n_integr_pts_per_elem * n_elems

    # get the number of strains per integration point
    n_strns_per_integr_pt = 1

    # get the number of stresses per integration point
    n_strs_per_integr_pt = 1

    # get the number of nodes per element
    n_nodes_per_elem = elem_conn.shape[1]

    # get the number of degrees of freedom per element
    n_dofs_per_elem = elem_dof.shape[1]

    # ge the number of degrees of freedom per node
    n_dofs_per_node = 1

    # get element node coords
    elem_node_coord = np.zeros((n_elems, n_nodes_per_elem, n_spaces))
    for i_elem in range(n_elems):
        # an element connectivity
        an_elem_conn = elem_conn[i_elem]
    
        # an element node coords
        an_elem_node_coord = node_coord[an_elem_conn-1, :]
        elem_node_coord[i_elem, :, :] = an_elem_node_coord


    # get the number of degrees of freedom
    n_dofs = n_nodes * n_dofs_per_node

    # get the dofs
    dof = np.arange(1, n_dofs+1)

    return node_coord, elem_conn, n_nodes, n_dofs, elem_dof, elem_spacing, n_spaces, \
    	n_integr_pts_per_elem, n_integr_pts, n_strns_per_integr_pt, n_strs_per_integr_pt, \
		n_nodes_per_elem, n_dofs_per_elem, n_dofs_per_node, elem_node_coord, n_dofs, dof


def get_domain_geo_2d(example_case, node_coord, elem_conn, elem_spacing):
    # geometry

    # get the number of nodes
    n_nodes = node_coord.shape[0]

    # get the number of elements
    n_elems = elem_conn.shape[0]

    # get the number of nodes per element
    n_nodes_per_elem = elem_conn.shape[1]

    # get the number of degrees of freedom per node
    n_dofs_per_node = 2

    # get the number of degrees of freedom per element
    n_dofs_per_elem = n_dofs_per_node * n_nodes_per_elem

    # number of degrees of freedom
    n_dofs = n_dofs_per_node * n_nodes

    # element degrees of freedom
    elem_dof = np.zeros((n_elems, n_dofs_per_elem), dtype=int)
    for i_elem in range(n_elems):
        for i_node_per_elem in range(n_nodes_per_elem):
            an_elem_node = elem_conn[i_elem, i_node_per_elem]
            for dof in range(n_dofs_per_node):
                elem_dof[i_elem, i_node_per_elem * n_dofs_per_node + dof] = \
                    (an_elem_node - 1) * n_dofs_per_node + dof + 1

    # get number of integration points per element
    n_integr_pts_per_elem = 4

    # get the number of integration points (Gauss points)
    n_integr_pts = n_integr_pts_per_elem * n_elems

    # get the number of strains and stresses per intergration point
    if example_case == "2d plane strain":
        # get the number of strains per integration point
        n_strns_per_integr_pt = 3

        # get the number of stresses per integration point
        n_strs_per_integr_pt = 4
    elif example_case == "2d plane stress":
        # get the number of strains per integration point
        n_strns_per_integr_pt = 4

        # get the number of stresses per integration point
        n_strs_per_integr_pt = 3

    # number of spaces
    n_spaces = 2

    # get element node coords
    elem_node_coord = np.zeros((n_elems, n_nodes_per_elem, n_spaces))
    for i_elem in range(n_elems):
        # an element connectivity
        an_elem_conn = elem_conn[i_elem]
    
        # an element node coords
        an_elem_node_coord = node_coord[an_elem_conn-1, :]
        elem_node_coord[i_elem, :, :] = an_elem_node_coord

    # get the dofs
    dof = np.arange(1, n_dofs+1)

    return node_coord, elem_conn, n_nodes, n_dofs, elem_dof, elem_spacing, n_spaces, \
    	n_integr_pts_per_elem, n_integr_pts, n_strns_per_integr_pt, n_strs_per_integr_pt, \
		n_nodes_per_elem, n_dofs_per_elem, n_dofs_per_node, elem_node_coord, n_dofs, dof, n_elems


# ***************************** get constitutive matrix *******************************************
def dplfun(x, x_pt, y_pt):
    """
    Derivative of the piecewise linear function 'plfun' defined by a set
    of npoint pairs {x, f(x)} stored in the matrix xfx (dimension 2*npoint).
    """
    r0 = 0.0
    npoint = len(x_pt)
    for i in range(npoint):
        if x >= x_pt[i]:
            continue
        else:
            if i == 0:
                # x < x1 --> f(x)=f(x1) --> df(x)/dx=0
                return r0
            else:
                # x(i-1) <= x < x(i)
                return (y_pt[i] - y_pt[i-1]) / (x_pt[i] - x_pt[i-1])
    # x >= x(npoint) --> f(x) = f(x(npoint)) --> df/dx=0
    return r0


def get_hardening_slope(Cauchy_strn_plc_eqn_tp1, \
                        hardening_pt_eqn_plc_strn, hardening_pt_yield_strs):
    # get the hardening slope
    hardening_slope = \
        dplfun(Cauchy_strn_plc_eqn_tp1, \
            hardening_pt_eqn_plc_strn, hardening_pt_yield_strs)
    
    # return hardening slope
    return hardening_slope


def get_const_mtrx_pst_2d(Cauchy_strn_plc_eqn_tp1, \
                plc_multiplier_ttp1_inc, \
                Cauchy_strs_tnr_tp1, \
                youngs_modulus, poisson_ratio, \
                hardening_pt_eqn_plc_strn, hardening_pt_yield_strs, \
                plc_const_mtrx_flag_tp1):
    # define numbers
    r1, r2, r3, r4 = 1.0, 2.0, 3.0, 4.0
    rp5 = 0.5
    # fourth-order symmetric identity tensor stored in array form
    foid = np.zeros((4,4))
    foid[0,0] = 1.0
    foid[1,1] = 1.0
    foid[2,2] = 0.5
    foid[3,3] = 1.0
    # second-order identity tensor stored in array form
    soid = np.array([1.0, 1.0, 0.0, 1.0])
    # get shear modulus
    shear_modulus = \
        youngs_modulus/(r2*(r1+poisson_ratio))
    # get bulk modulus
    bulk_modulus = \
        youngs_modulus/(r3*(r1-r2*poisson_ratio))
    # compute aux variables
    r2s = r2 * shear_modulus
    r1d3 = r1/r3
    r2d3 = r2*r1d3
    # check if the elastoplastic consistent tangent
    # constitutive matrix should be computed
    if plc_const_mtrx_flag_tp1 == True:
        # compute the elasto-plastic consistent
        # tangent constitutive matrix
        # get xi
        xi = r2d3 * \
            (Cauchy_strs_tnr_tp1[0] * Cauchy_strs_tnr_tp1[0] + \
            Cauchy_strs_tnr_tp1[1] * Cauchy_strs_tnr_tp1[1] - \
            Cauchy_strs_tnr_tp1[0] * Cauchy_strs_tnr_tp1[1]) + \
            r2 * Cauchy_strs_tnr_tp1[2] * Cauchy_strs_tnr_tp1[2]
        # get the hardening slope
        hardening_slope = \
            get_hardening_slope(Cauchy_strn_plc_eqn_tp1, \
                                hardening_pt_eqn_plc_strn, hardening_pt_yield_strs)
        # matrix E components
        estar1 = r3 * youngs_modulus / \
            (r3 * (r1 - poisson_ratio) + youngs_modulus * plc_multiplier_ttp1_inc)
        estar2 = r2s / (r1 + r2s * plc_multiplier_ttp1_inc)
        estar3 = shear_modulus / (r1 + r2s * plc_multiplier_ttp1_inc)
        e11 = rp5 * (estar1 + estar2)
        e22 = e11
        e12 = rp5 * (estar1 - estar2)
        e33 = estar3
        # components of the matrix product EP
        epsta1 = r1d3 * estar1
        epsta2 = estar2
        epsta3 = epsta2
        ep11 = rp5 * (epsta1 + epsta2)
        ep22 = ep11
        ep12 = rp5 * (epsta1 - epsta2)
        ep21 = ep12
        ep33 = epsta3
        # vector n
        vecn = np.zeros(3)
        vecn[0] = ep11 * Cauchy_strs_tnr_tp1[0] + ep12 * Cauchy_strs_tnr_tp1[1]
        vecn[1] = ep21 * Cauchy_strs_tnr_tp1[0] + ep22 * Cauchy_strs_tnr_tp1[1]
        vecn[2] = ep33 * Cauchy_strs_tnr_tp1[2]
        # scalar alpha
        denom1 = \
            Cauchy_strs_tnr_tp1[0] * (r2d3 * vecn[0] - r1d3 * vecn[1]) + \
            Cauchy_strs_tnr_tp1[1] * (r2d3 * vecn[1] - r1d3 * vecn[0]) + \
            Cauchy_strs_tnr_tp1[2] * r2 * vecn[2]
        denom2 = r2 * xi * \
            hardening_slope / (r3 - r2 * hardening_slope * plc_multiplier_ttp1_inc)
        alpha = r1 / (denom1 + denom2)
        # assemble elasto-plastic tangent
        const_mtrx_tp1 = np.zeros((3, 3))
        const_mtrx_tp1[0, 0] = e11 - alpha * vecn[0] * vecn[0]
        const_mtrx_tp1[0, 1] = e12 - alpha * vecn[0] * vecn[1]
        const_mtrx_tp1[0, 2] = -alpha * vecn[0] * vecn[2]
        const_mtrx_tp1[1, 0] = const_mtrx_tp1[0, 1]
        const_mtrx_tp1[1, 1] = e22 - alpha * vecn[1] * vecn[1]
        const_mtrx_tp1[1, 2] = -alpha * vecn[1] * vecn[2]
        const_mtrx_tp1[2, 0] = const_mtrx_tp1[0, 2]
        const_mtrx_tp1[2, 1] = const_mtrx_tp1[1, 2]
        const_mtrx_tp1[2, 2] = e33 - alpha * vecn[2] * vecn[2]
    else:
        # compute the elastic constitutive matrix
        # number of stresses
        n_strs = 3
        # compute aux variables
        r4sd3 = r4 * shear_modulus / r3
        factor = (bulk_modulus - r2s / r3) * \
            (r2s / (bulk_modulus + r4sd3))
        # initialise constitutive matrix
        const_mtrx_tp1 = np.zeros((n_strs, n_strs))
        # get constitutive matrix
        for i in range(n_strs):
            for j in range(i, n_strs):
                const_mtrx_tp1[i, j] = \
                    r2s * foid[i, j] + factor * soid[i] * soid[j]
        # fill lower triangle
        for j in range(n_strs - 1):
                for i in range(j + 1, n_strs):
                    const_mtrx_tp1[i, j] = \
                        const_mtrx_tp1[j, i]
    # return constitutive matrix
    return const_mtrx_tp1


# ****************************** get stiffness matrix *********************************************
def get_stiff_mtrx_IJ_tp1_2d(example_case, n_elems, elem_dof, n_dofs_per_elem, n_integr_pts_per_elem, \
                            integr_pt_strain_disp_mtrx, integr_pt_Cauchy_strn_plc_eqn_tp1, \
                            integr_pt_plc_multiplier_ttp1_inc, integr_pt_Cauchy_strs_tnr_tp1, \
                            integr_pt_plc_const_mtrx_flag_tp1, youngs_modulus, poisson_ratio, \
                            hardening_pt_eqn_plc_strn, hardening_pt_yield_strs, \
                            n_dofs, integr_pt_vol):
    # initialise stiffness matrix
    stiff_mtrx_IJ_tp1 = np.zeros((n_dofs, n_dofs))

    # initialise counter
    i_integr_pt_count = 0

    for i_elem in range(n_elems):
        # element connectivity
        an_elem_dof = elem_dof[i_elem,:]

        # initialise local matrices/ vectors
        stiff_mtrx_IJ_elem = np.zeros((n_dofs_per_elem, n_dofs_per_elem))
        for i_integr_pt in range(n_integr_pts_per_elem):
            # strain - displacement matrix
            strain_disp_mtrx = integr_pt_strain_disp_mtrx[i_integr_pt_count,:,:]

            # total Cauchy equivalent plastic strain at time t+1
            Cauchy_strn_plc_eqn_tp1 = integr_pt_Cauchy_strn_plc_eqn_tp1[i_integr_pt_count]

            # incremental plastic multiplier from time t to t+1
            plc_multiplier_ttp1_inc = integr_pt_plc_multiplier_ttp1_inc[i_integr_pt_count]

            # total Cauchy stress at time t+1
            Cauchy_strs_tnr_tp1 = integr_pt_Cauchy_strs_tnr_tp1[i_integr_pt_count,:]

            # elastic or plastic flag for the consitutive matrix
            plc_const_mtrx_flag_tp1 = integr_pt_plc_const_mtrx_flag_tp1[i_integr_pt_count]

            # constitutive matrix at time t+1
            if example_case == "2d plane strain":
                print("To be implemented in the future!")
            elif example_case == "2d plane stress":
                const_mtrx = get_const_mtrx_pst_2d(Cauchy_strn_plc_eqn_tp1, \
                                        plc_multiplier_ttp1_inc, \
                                        Cauchy_strs_tnr_tp1, \
                                        youngs_modulus, poisson_ratio, \
                                        hardening_pt_eqn_plc_strn, hardening_pt_yield_strs, \
                                        plc_const_mtrx_flag_tp1)

            # volume
            a_vol = integr_pt_vol[i_integr_pt_count]

            # local stiffness matrix
            term_1 = np.dot(strain_disp_mtrx.T, const_mtrx)
            term_2 = np.dot(term_1, strain_disp_mtrx)
            stiff_mtrx_IJ_elem += term_2 * a_vol

            # update integration point counter
            i_integr_pt_count = i_integr_pt_count + 1

        # assemble stiffness matrix
        stiff_mtrx_IJ_tp1[an_elem_dof[:, np.newaxis] - 1, an_elem_dof - 1] = \
            stiff_mtrx_IJ_tp1[an_elem_dof[:, np.newaxis] - 1, an_elem_dof - 1] + stiff_mtrx_IJ_elem
    
    # return
    return stiff_mtrx_IJ_tp1


# ******************************* get stress tensor ***********************************************
def get_yield_strs(Cauchy_strn_plc_eqn_tp1, \
                    hardening_pt_eqn_plc_strn, hardening_pt_yield_strs):
    # get the yield stress
    yield_strs = \
        np.interp(Cauchy_strn_plc_eqn_tp1, \
                hardening_pt_eqn_plc_strn, hardening_pt_yield_strs)
    
    # return
    return yield_strs


def get_Cauchy_strs_tnr_tp1_pst_2d(Cauchy_strn_elc_trl_tnr_tp1, Cauchy_strn_plc_tnr_t,
                            Cauchy_strn_plc_eqn_t, youngs_modulus, poisson_ratio,
                            hardening_pt_eqn_plc_strn, hardening_pt_yield_strs):
    # declare numbers
    r0, rp5, r1, r2, r3, r4, r6 = \
        0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0
    # Newton-Raphson parameters (for plasticity)
    # (tolerance)
    tol = 1.0e-7 # 1.0e-8
    # (max number of iterations)
    mxiter = 50
        
    # initialise plastic multiplier from time t to time t + 1
    # (incremental)
    plc_multiplier_ttp1_inc = r0
        
    # set the has yield flag as as false
    has_yielded_tp1 = False        
    # set state update failure flag as flase
    sufail_tp1 = False

    # set previously (equilibrium) converged accumulated plastic strain
    epbarn = Cauchy_strn_plc_eqn_t

    # get shear and bulk moduli and other necessary constants
    shear_modulus = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
    bulk_modulus = youngs_modulus / (3.0 * (1.0 - 2.0 * poisson_ratio))
    r2s = r2 * shear_modulus
    r4s = r4 * shear_modulus
    r1d3 = r1 / r3
    r1d6 = r1 / r6
    r2d3 = r2 * r1d3
    sqr2d3 = np.sqrt(r2d3)
    r4sd3 = r4s * r1d3

    # get the elastic trial volumetric strain
    factor = r2s / (bulk_modulus + r4sd3)
    eev = \
        (Cauchy_strn_elc_trl_tnr_tp1[0] + \
        Cauchy_strn_elc_trl_tnr_tp1[1]) * factor

    # get the elastic trial deviatoric strain
    eevd3 = eev / r3
    eet = np.zeros(3)
    eet[0] = Cauchy_strn_elc_trl_tnr_tp1[0] - eevd3
    eet[1] = Cauchy_strn_elc_trl_tnr_tp1[1] - eevd3
    eet[2] = Cauchy_strn_elc_trl_tnr_tp1[2] * rp5

    # get the elastic trial stress components
    pt = bulk_modulus * eev
    strest = np.zeros(3)
    strest[0] = r2s * eet[0] + pt
    strest[1] = r2s * eet[1] + pt
    strest[2] = r2s * eet[2]

    # compute yield function value at trial state
    a1 = (strest[0] + strest[1]) ** 2
    a2 = (strest[1] - strest[0]) ** 2
    a3 = strest[2] ** 2
    xi = r1d6 * a1 + rp5 * a2 + r2 * a3
    sigmay = get_yield_strs(epbarn, \
            hardening_pt_eqn_plc_strn, hardening_pt_yield_strs)

    # get yield function
    phi = rp5 * xi - (r1 / r3) * sigmay ** 2

    # check for plastic admissibility
    if phi / sigmay > tol:
        # Plastic step: Apply return mapping - use Newton-Raphson algorithm
        # to solve the plane stress-projected return mapping
        # equation for the plastic multiplier
            
        # set the has yield flag as as true
        has_yielded_tp1 = True

        # equivalent plastic strain at time t + 1
        epbar = epbarn
        sqrtxi = np.sqrt(xi)
        b1 = r1
        b2 = r1
        fmodu = youngs_modulus / (r3 * (r1 - poisson_ratio))

        for nriter in range(1, mxiter + 1):                
            # get the hardening slope
            hardening_slope = \
                get_hardening_slope(epbar, \
                            hardening_pt_eqn_plc_strn, hardening_pt_yield_strs)
                
            # compute residual derivative
            dxi = \
                -a1 * fmodu / (r3 * b1 ** 3) - r2s * (a2 + r4 * a3) / (b2 ** 3)
            hbar = \
                r2 * sigmay * hardening_slope * sqr2d3 * \
                (sqrtxi + plc_multiplier_ttp1_inc * dxi / (2 * sqrtxi))
            dphi = rp5 * dxi - r1d3 * hbar

            # compute Newton-Raphson increment and update equation variable dgama
            plc_multiplier_ttp1_inc = plc_multiplier_ttp1_inc - phi / dphi

            # compute new residual (yield function value)
            b1 = r1 + fmodu * plc_multiplier_ttp1_inc
            b2 = r1 + r2s * plc_multiplier_ttp1_inc
            xi = \
                r1d6 * a1 / (b1 * b1) + \
                (rp5 * a2 + r2 * a3)/(b2 * b2)
            sqrtxi = np.sqrt(xi)
            epbar = epbarn + plc_multiplier_ttp1_inc * sqr2d3 * sqrtxi
            sigmay = get_yield_strs(epbar, \
                hardening_pt_eqn_plc_strn, hardening_pt_yield_strs)
            phi = rp5 * xi - r1d3 * sigmay ** 2

            # check for convergence
            resnor = abs(phi / sigmay)
            if resnor <= tol:
                # update accumulated plastic strain
                Cauchy_strn_plc_eqn_tp1 = epbar

                # update stress components:   sigma := A sigma^trial
                astar1 = \
                    r3 * (r1 - poisson_ratio) / \
                    (r3 * (r1 - poisson_ratio) + youngs_modulus * plc_multiplier_ttp1_inc)
                astar2 = r1 / (r1 + r2s * plc_multiplier_ttp1_inc)
                a11 = rp5 * (astar1 + astar2)
                a22 = a11
                a12 = rp5 * (astar1 - astar2)
                a21 = a12
                a33 = astar2
                Cauchy_strs_tnr_tp1 = np.zeros(3)
                Cauchy_strs_tnr_tp1[0] = a11 * strest[0] + a12 * strest[1]
                Cauchy_strs_tnr_tp1[1] = a21 * strest[0] + a22 * strest[1]
                Cauchy_strs_tnr_tp1[2] = a33 * strest[2]

                # compute corresponding elastic (engineering) strain components
                factg = r1 / r2s
                p = r1d3 * (Cauchy_strs_tnr_tp1[0] + Cauchy_strs_tnr_tp1[1])
                eev = p / bulk_modulus
                eevd3 = r1d3 * eev

                Cauchy_strn_elc_tnr_tp1 = np.zeros(4)
                Cauchy_strn_elc_tnr_tp1[0] = \
                    factg*(r2d3 * Cauchy_strs_tnr_tp1[0] - \
                               r1d3 * Cauchy_strs_tnr_tp1[1]) + eevd3
                Cauchy_strn_elc_tnr_tp1[1] = \
                    factg*(r2d3 * Cauchy_strs_tnr_tp1[1] - \
                               r1d3 * Cauchy_strs_tnr_tp1[0]) + eevd3
                Cauchy_strn_elc_tnr_tp1[2] = \
                        factg * Cauchy_strs_tnr_tp1[2]*r2
                Cauchy_strn_elc_tnr_tp1[3] =\
                        -poisson_ratio/(r1-poisson_ratio) * \
                        (Cauchy_strn_elc_tnr_tp1[0] + \
                        Cauchy_strn_elc_tnr_tp1[1])
                
                # Cauchy equivalent stress at time t+1. It is also equal to Von Mises stresses since that 
                # criterion is used.
                Cauchy_strs_eqn_tp1 = sigmay

                # incremental plastic strains from time t to t+1
                # (in-plane)
                Cauchy_strn_plc_tnr_ttp1_inc = np.zeros(4)
                Pi = (1.0/3.0) * np.array([[2.0, -1.0, 0.0],
                                           [-1.0, 2.0, 0.0],
                                           [0.0, 0.0, 6.0]])
                Cauchy_strn_plc_tnr_ttp1_inc[0:3] = plc_multiplier_ttp1_inc * np.dot(Pi,Cauchy_strs_tnr_tp1)
                # (out-of-plane) Eq. 9.38
                Cauchy_strn_plc_tnr_ttp1_inc[3] = -1.0 * (Cauchy_strn_plc_tnr_ttp1_inc[0] + Cauchy_strn_plc_tnr_ttp1_inc[1])

                # total plastic stain
                Cauchy_strn_plc_tnr_tp1 = Cauchy_strn_plc_tnr_t + Cauchy_strn_plc_tnr_ttp1_inc

                # total strain
                Cauchy_strn_tnr_tp1 = Cauchy_strn_elc_tnr_tp1 + Cauchy_strn_plc_tnr_tp1

                # break
                break
        else:
            # reset failure flag and print warning message if N-R algorithm fails
            sufail_tp1 = True
    else:
        # equivalent plastic strain remain the same as in the
        # previous time step
        Cauchy_strn_plc_eqn_tp1 = Cauchy_strn_plc_eqn_t
        # Elastic step: Update stress using linear elastic law
        Cauchy_strs_tnr_tp1 = np.zeros(3)
        Cauchy_strs_tnr_tp1[0] = strest[0]
        Cauchy_strs_tnr_tp1[1] = strest[1]
        Cauchy_strs_tnr_tp1[2] = strest[2]
        
        # elastic engineering strain
        Cauchy_strn_elc_tnr_tp1 = np.zeros(4)
        Cauchy_strn_elc_tnr_tp1[0] = Cauchy_strn_elc_trl_tnr_tp1[0]
        Cauchy_strn_elc_tnr_tp1[1] = Cauchy_strn_elc_trl_tnr_tp1[1]
        Cauchy_strn_elc_tnr_tp1[2] = Cauchy_strn_elc_trl_tnr_tp1[2]
        Cauchy_strn_elc_tnr_tp1[3] = \
            -poisson_ratio / (r1 - poisson_ratio) * (Cauchy_strn_elc_trl_tnr_tp1[0] + \
                                                     Cauchy_strn_elc_trl_tnr_tp1[1])

        # plastic strain equal to the previous step
        Cauchy_strn_plc_tnr_tp1 = Cauchy_strn_plc_tnr_t
        
        # total strain
        Cauchy_strn_tnr_tp1 = Cauchy_strn_elc_tnr_tp1 + Cauchy_strn_plc_tnr_tp1

        # Cauchy equivalent stress at time t+1. It is also equal to Von Mises stresses since that 
        # criterion is used.
        strs_xx_minus_strs_yy_square = (strest[0]-strest[1])**2
        strs_yy_minus_strs_zz_square = (strest[1])**2
        strs_zz_minus_strs_xx_square = (-strest[0])**2
        sum_norm_strs = strs_xx_minus_strs_yy_square + strs_yy_minus_strs_zz_square + strs_zz_minus_strs_xx_square
        strs_xy_square = strest[2]**2
        sum_norm_shr_strs = sum_norm_strs + 6.0*strs_xy_square
        Cauchy_strs_eqn_tp1 = np.sqrt(0.5*sum_norm_shr_strs)
    # return variables
    return Cauchy_strn_tnr_tp1, Cauchy_strn_elc_tnr_tp1, Cauchy_strn_plc_tnr_tp1, \
        Cauchy_strn_plc_eqn_tp1, plc_multiplier_ttp1_inc, Cauchy_strs_tnr_tp1, \
        Cauchy_strs_eqn_tp1, has_yielded_tp1, sufail_tp1


# ****************************** get internal force ***********************************************
def get_force_int_I_ttp1_inc_2d(n_dofs, n_integr_pts, n_strns_per_integr_pt, n_strs_per_integr_pt, n_elems, 
                            elem_dof, disp_I_ttp1_inc, n_dofs_per_elem,
                            n_integr_pts_per_elem, integr_pt_strain_disp_mtrx, integr_pt_Cauchy_strn_elc_tnr_t,
                            integr_pt_Cauchy_strn_plc_tnr_t, integr_pt_Cauchy_strn_plc_eqn_t, youngs_modulus, poisson_ratio,
                            hardening_pt_eqn_plc_strn, hardening_pt_yield_strs, integr_pt_Cauchy_strs_tnr_t,
                            integr_pt_vol):

    # initialise force internal matrix
    force_int_I_ttp1_inc = np.zeros(n_dofs)

    # initialise integration points' quantities
    integr_pt_Cauchy_strn_tnr_tp1 = np.zeros((n_integr_pts, n_strns_per_integr_pt))
    integr_pt_Cauchy_strn_elc_tnr_tp1 = np.zeros((n_integr_pts, n_strns_per_integr_pt))
    integr_pt_Cauchy_strn_plc_tnr_tp1 = np.zeros((n_integr_pts, n_strns_per_integr_pt))
    integr_pt_Cauchy_strn_plc_eqn_tp1 = np.zeros(n_integr_pts)
    integr_pt_plc_multiplier_ttp1_inc = np.zeros(n_integr_pts)
    integr_pt_Cauchy_strs_tnr_tp1 = np.zeros((n_integr_pts, n_strs_per_integr_pt))
    integr_pt_Cauchy_strs_eqn_tp1 = np.zeros(n_integr_pts)
    integr_pt_has_yielded_tp1 = np.full(n_integr_pts, False, dtype=bool)
    integr_pt_sufail_tp1 = np.full(n_integr_pts, False, dtype=bool)

    # initialise counter
    i_integr_pt_count = 0

    for i_elem in range(n_elems):
        # element connectivity
        an_elem_dof = elem_dof[i_elem, :]

        # get incremental displacement field at element
        disp_I_ttp1_inc_elem = disp_I_ttp1_inc[an_elem_dof - 1]

        # initialise local matrices/ vectors
        force_int_I_ttp1_inc_elem = np.zeros(n_dofs_per_elem)

        for i_integr_pt in range(n_integr_pts_per_elem):
            # strain - displacement matrix
            strain_disp_mtrx = integr_pt_strain_disp_mtrx[i_integr_pt_count, :, :]

            # get incremental Cauchy elastic trial strain tensor from time t to t + 1
            Cauchy_strn_elc_trl_tnr_ttp1_inc = np.zeros(4)
            Cauchy_strn_elc_trl_tnr_ttp1_inc[0:3] = np.dot(strain_disp_mtrx, disp_I_ttp1_inc_elem)
            # get the out of plane strain
            coef = -1.0 * (poisson_ratio / (1.0 - poisson_ratio))
            Cauchy_strn_elc_trl_tnr_ttp1_inc[3] = \
                coef * (Cauchy_strn_elc_trl_tnr_ttp1_inc[0] + \
                        Cauchy_strn_elc_trl_tnr_ttp1_inc[1])

            # get total Cauchy elastic trial strain tensor at time t+1
            Cauchy_strn_elc_tnr_t = integr_pt_Cauchy_strn_elc_tnr_t[i_integr_pt_count]
            Cauchy_strn_elc_trl_tnr_tp1 = Cauchy_strn_elc_tnr_t + Cauchy_strn_elc_trl_tnr_ttp1_inc

            # equivalent plastic Cauchy strain at time t
            Cauchy_strn_plc_eqn_t = integr_pt_Cauchy_strn_plc_eqn_t[i_integr_pt_count]

            # Cauchy plastic strain at time t
            Cauchy_strn_plc_tnr_t = integr_pt_Cauchy_strn_plc_tnr_t[i_integr_pt_count]

            # get total Cauchy strain and stress tensor at time t+1
            Cauchy_strn_tnr_tp1, Cauchy_strn_elc_tnr_tp1, Cauchy_strn_plc_tnr_tp1, \
            Cauchy_strn_plc_eqn_tp1, plc_multiplier_ttp1_inc, Cauchy_strs_tnr_tp1, \
            Cauchy_strs_eqn_tp1, has_yielded_tp1, sufail_tp1 = \
                get_Cauchy_strs_tnr_tp1_pst_2d(Cauchy_strn_elc_trl_tnr_tp1, Cauchy_strn_plc_tnr_t,
                            Cauchy_strn_plc_eqn_t, youngs_modulus, poisson_ratio,
                            hardening_pt_eqn_plc_strn, hardening_pt_yield_strs)

            integr_pt_Cauchy_strn_tnr_tp1[i_integr_pt_count,:] = Cauchy_strn_tnr_tp1
            integr_pt_Cauchy_strn_elc_tnr_tp1[i_integr_pt_count,:] = Cauchy_strn_elc_tnr_tp1
            integr_pt_Cauchy_strn_plc_tnr_tp1[i_integr_pt_count,:] = Cauchy_strn_plc_tnr_tp1
            integr_pt_Cauchy_strn_plc_eqn_tp1[i_integr_pt_count] = Cauchy_strn_plc_eqn_tp1
            integr_pt_plc_multiplier_ttp1_inc[i_integr_pt_count] = plc_multiplier_ttp1_inc
            integr_pt_Cauchy_strs_tnr_tp1[i_integr_pt_count,:] = Cauchy_strs_tnr_tp1
            integr_pt_Cauchy_strs_eqn_tp1[i_integr_pt_count] = Cauchy_strs_eqn_tp1
            integr_pt_has_yielded_tp1[i_integr_pt_count] = has_yielded_tp1
            integr_pt_sufail_tp1[i_integr_pt_count] = sufail_tp1

            # get incremental Cauchy stress tensor from time t to t+1
            Cauchy_strs_tnr_t = integr_pt_Cauchy_strs_tnr_t[i_integr_pt_count,:]
            Cauchy_strs_tnr_ttp1_inc = Cauchy_strs_tnr_tp1 - Cauchy_strs_tnr_t

            # volume
            a_vol = integr_pt_vol[i_integr_pt_count]

            # get vector
            force_int_I_ttp1_inc_elem += np.dot(strain_disp_mtrx.T, Cauchy_strs_tnr_ttp1_inc[0:3]) * a_vol

            # update integration point counter
            i_integr_pt_count = i_integr_pt_count + 1

        # assemble internal force
        force_int_I_ttp1_inc[an_elem_dof - 1] = \
            force_int_I_ttp1_inc[an_elem_dof - 1] + force_int_I_ttp1_inc_elem

    # return
    return integr_pt_Cauchy_strn_tnr_tp1, integr_pt_Cauchy_strn_elc_tnr_tp1, integr_pt_Cauchy_strn_plc_tnr_tp1, \
        integr_pt_Cauchy_strn_plc_eqn_tp1, integr_pt_plc_multiplier_ttp1_inc, integr_pt_Cauchy_strs_tnr_tp1, \
        integr_pt_Cauchy_strs_eqn_tp1, \
        integr_pt_has_yielded_tp1, integr_pt_sufail_tp1, force_int_I_ttp1_inc


# ***************** get field and integration points' variables for storing ***********************
def get_fld_integr_pt_strg(n_time_steps, n_strns_per_integr_pt, \
                        n_strs_per_integr_pt):
    # field variables for storing
    disp_I_strg = np.zeros(n_time_steps)
    force_ext_strg = np.zeros(n_time_steps)

    # integration point variables for storing
    Cauchy_strn_tnr_strg = np.zeros((n_time_steps, n_strns_per_integr_pt))
    Cauchy_strn_elc_tnr_strg = np.zeros((n_time_steps, n_strns_per_integr_pt))
    Cauchy_strn_plc_tnr_strg = np.zeros((n_time_steps, n_strns_per_integr_pt))
    Cauchy_strn_plc_eqn_strg = np.zeros(n_time_steps)
    Cauchy_strs_tnr_strg = np.zeros((n_time_steps, n_strs_per_integr_pt))
    Cauchy_strs_eqn_strg = np.zeros(n_time_steps)
    has_yielded_strg = np.zeros(n_time_steps)
    sufail_strg = np.zeros(n_time_steps)

    # return
    return disp_I_strg, force_ext_strg, Cauchy_strn_tnr_strg, \
        Cauchy_strn_elc_tnr_strg, Cauchy_strn_plc_tnr_strg, \
        Cauchy_strn_plc_eqn_strg, Cauchy_strs_tnr_strg, \
        Cauchy_strs_eqn_strg, has_yielded_strg, sufail_strg

# **************** store field and integration points' variables for storing **********************
def store_fld_integr_pt_strg(i_time_step, dof_id, integr_pt_id, \
        disp_I_strg, force_ext_I_strg, Cauchy_strn_tnr_strg, Cauchy_strn_elc_tnr_strg, \
        Cauchy_strn_plc_tnr_strg, Cauchy_strn_plc_eqn_strg, Cauchy_strs_tnr_strg, \
        Cauchy_strs_eqn_strg, has_yielded_strg, sufail_strg, \
        disp_I_tp1, force_ext_I_tp1, integr_pt_Cauchy_strn_tnr_tp1, \
        integr_pt_Cauchy_strn_elc_tnr_tp1, integr_pt_Cauchy_strn_plc_tnr_tp1, \
        integr_pt_Cauchy_strn_plc_eqn_tp1, integr_pt_Cauchy_strs_tnr_tp1, \
        integr_pt_Cauchy_strs_eqn_tp1, integr_pt_has_yielded_tp1, \
        integr_pt_sufail_tp1):

    # store field variables
    disp_I_strg[i_time_step - 1] = disp_I_tp1[dof_id-1]
    force_ext_I_strg[i_time_step - 1] = force_ext_I_tp1[dof_id-1]

    # store integration point variables
    Cauchy_strn_tnr_strg[i_time_step - 1,:] = integr_pt_Cauchy_strn_tnr_tp1[integr_pt_id-1,:]
    Cauchy_strn_elc_tnr_strg[i_time_step - 1,:] = integr_pt_Cauchy_strn_elc_tnr_tp1[integr_pt_id-1,:]
    Cauchy_strn_plc_tnr_strg[i_time_step - 1,:] = integr_pt_Cauchy_strn_plc_tnr_tp1[integr_pt_id-1,:]
    Cauchy_strn_plc_eqn_strg[i_time_step - 1] = integr_pt_Cauchy_strn_plc_eqn_tp1[integr_pt_id-1]
    Cauchy_strs_tnr_strg[i_time_step - 1,:] = integr_pt_Cauchy_strs_tnr_tp1[integr_pt_id-1,:]
    Cauchy_strs_eqn_strg[i_time_step - 1] = integr_pt_Cauchy_strs_eqn_tp1[integr_pt_id-1]
    has_yielded_strg[i_time_step - 1] = integr_pt_has_yielded_tp1[integr_pt_id-1]
    sufail_strg[i_time_step - 1] = integr_pt_sufail_tp1[integr_pt_id-1]

    return disp_I_strg, force_ext_I_strg, Cauchy_strn_tnr_strg, Cauchy_strn_elc_tnr_strg, \
        Cauchy_strn_plc_tnr_strg, Cauchy_strn_plc_eqn_strg, Cauchy_strs_tnr_strg, \
        Cauchy_strs_eqn_strg, has_yielded_strg, sufail_strg


# **************************** initialise field variables *****************************************
def init_fld_vars(n_dofs, n_nodes):
    # initialise field variables
    # (total displacement at time t+1)
    disp_I_tp1 = np.zeros(n_dofs)
    # (total external force at time t+1)
    force_ext_I_tp1 = np.zeros(n_dofs)

    # return
    return disp_I_tp1, force_ext_I_tp1

# ************************ initialise integration points variables ********************************
def init_integr_pt_vars(n_integr_pts, n_strns_per_integr_pt, \
                        n_strs_per_integr_pt):
    # initialise integration points' quantities
    # (Cauchy elastic strain tensor at time t+1)
    integr_pt_Cauchy_strn_elc_tnr_tp1 = np.zeros((n_integr_pts, n_strns_per_integr_pt))
    # (Cauchy plastic strain tensor at time t+1)
    integr_pt_Cauchy_strn_plc_tnr_tp1 = np.zeros((n_integr_pts, n_strns_per_integr_pt))
    # (Cauchy equivalent plastic strain at time t+1)
    integr_pt_Cauchy_strn_plc_eqn_tp1 = np.zeros(n_integr_pts)
    # (Cauchy stress tensor at time t+1)
    integr_pt_Cauchy_strs_tnr_tp1 = np.zeros((n_integr_pts, n_strs_per_integr_pt))
    # (incremental plastic multiplier from time t to t+1)
    integr_pt_plc_multiplier_ttp1_inc = np.zeros(n_integr_pts)
    # (has yielded flag at time t+1)
    integr_pt_has_yielded_tp1 = np.full(n_integr_pts, False, dtype=bool)
    # (has failed flag at time t+1)
    integr_pt_sufail_tp1 = np.full(n_integr_pts, False, dtype=bool)

    # return
    return integr_pt_Cauchy_strn_elc_tnr_tp1, integr_pt_Cauchy_strn_plc_tnr_tp1, \
        integr_pt_Cauchy_strn_plc_eqn_tp1, integr_pt_Cauchy_strs_tnr_tp1, \
        integr_pt_plc_multiplier_ttp1_inc, integr_pt_has_yielded_tp1, \
        integr_pt_sufail_tp1

# ****************************** update field variables *******************************************
def update_fld_vars(disp_I_tp1, force_ext_I_tp1, phase_fld_I_tp1):
    # update field variables
    disp_I_t = disp_I_tp1
    force_ext_I_t = force_ext_I_tp1
    phase_fld_I_t = phase_fld_I_tp1

    # return
    return disp_I_t, force_ext_I_t, phase_fld_I_t

# ************************* update integration points variables ***********************************
def update_integr_pt_vars(integr_pt_Cauchy_strn_tnr_tp1, integr_pt_Cauchy_strn_tnr_plus_tp1, \
                        integr_pt_Cauchy_strn_tnr_minus_tp1, integr_pt_Cauchy_strs_tnr_plus_tp1, \
                        integr_pt_Cauchy_strs_tnr_minus_tp1, integr_pt_Cauchy_strs_tnr_tp1, \
                        integr_pt_elc_eng_dens_plus_tp1, integr_pt_elc_eng_dens_minus_tp1, \
                        integr_pt_elc_eng_dens_tp1, integr_pt_hist_fld_tp1, integr_pt_degrad_func_tp1, \
                        integr_pt_phase_fld_tp1, integr_pt_der1_phase_fld_tp1):
    # update variables
    # (Cauchy strain tensor at time t)
    integr_pt_Cauchy_strn_tnr_t = integr_pt_Cauchy_strn_tnr_tp1
    # (Cauchy strain tensor plus at time t)
    integr_pt_Cauchy_strn_tnr_plus_t = integr_pt_Cauchy_strn_tnr_plus_tp1
    # (Cauchy strain tensor minus at time t)
    integr_pt_Cauchy_strn_tnr_minus_t = integr_pt_Cauchy_strn_tnr_minus_tp1
    # (Cauchy stress tensor plus at time t)
    integr_pt_Cauchy_strs_tnr_plus_t = integr_pt_Cauchy_strs_tnr_plus_tp1
    # (Cauchy stress tensor minus at time t)
    integr_pt_Cauchy_strs_tnr_minus_t = integr_pt_Cauchy_strs_tnr_minus_tp1
    # (Cauchy stress tensor at time t)
    integr_pt_Cauchy_strs_tnr_t = integr_pt_Cauchy_strs_tnr_tp1
    # (elastic energy density plus at time t)
    integr_pt_elc_eng_dens_plus_t = integr_pt_elc_eng_dens_plus_tp1
    # (elastic energy density minus at time t)
    integr_pt_elc_eng_dens_minus_t = integr_pt_elc_eng_dens_minus_tp1
    # (elastic energy density at time t)
    integr_pt_elc_eng_dens_t = integr_pt_elc_eng_dens_tp1
    # (history field at time t)
    integr_pt_hist_fld_t = integr_pt_hist_fld_tp1
    # (degradation function at time t)
    integr_pt_degrad_func_t = integr_pt_degrad_func_tp1
    # (phase field at time t)
    integr_pt_phase_fld_t = integr_pt_phase_fld_tp1
    # (first spatial derivatives of phase field at time t+1)
    integr_pt_der1_phase_fld_t = integr_pt_der1_phase_fld_tp1

    # return
    return integr_pt_Cauchy_strn_tnr_t, integr_pt_Cauchy_strn_tnr_plus_t, \
        integr_pt_Cauchy_strn_tnr_minus_t, integr_pt_Cauchy_strs_tnr_plus_t, \
        integr_pt_Cauchy_strs_tnr_minus_t, integr_pt_Cauchy_strs_tnr_t, \
        integr_pt_elc_eng_dens_plus_t, integr_pt_elc_eng_dens_minus_t, \
        integr_pt_elc_eng_dens_t, integr_pt_hist_fld_t, integr_pt_degrad_func_t, \
        integr_pt_phase_fld_t, integr_pt_der1_phase_fld_t


# =================================================================================================