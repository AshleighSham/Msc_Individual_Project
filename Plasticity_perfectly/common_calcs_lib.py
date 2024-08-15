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

# ===================================== set up pecision ===========================================
# increase the precision
np.set_printoptions(precision=16)
#np.set_printoptions(precision=5, linewidth=200)
from decimal import Decimal, getcontext 
getcontext().prec = 16

# ===================================== functions =================================================
# *********************************** get Lame constants ******************************************
def get_lame_constants(youngs_modulus, poisson_ratio):
    # first Lame constant
    lambda_lame = \
        poisson_ratio * youngs_modulus/((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))

    # second Lame constant
    mu_lame = youngs_modulus/(2.0 * (1.0 + poisson_ratio))

    # return
    return lambda_lame, mu_lame

# *********************************** get Lame constants ******************************************
# ************************ get Gauss points natural coords and weights ****************************
def Gauss_pt_nat_coords_weights_1d():
    # Gauss points' natural coords
    nat_coord = np.array([[-0.577350269189626], 
                        [ 0.577350269189626]])

    # Gauss points' weights
    weight = np.array([1.0, 1.0])

    return nat_coord, weight

def Gauss_pt_nat_coords_weights_2d():
    # Gauss points' natural coords
    nat_coord = np.array([[-0.577350269189626, -0.577350269189626],
                        [ 0.577350269189626, -0.577350269189626],
                        [ 0.577350269189626,  0.577350269189626],
                        [-0.577350269189626,  0.577350269189626]])

    # Gauss points' weights
    weight = np.array([1.0, 1.0, 1.0, 1.0])

    return nat_coord, weight

# ******************** get integration points natural coords and weights **************************
def get_integr_pt_nat_coords_weights(n_elems, n_integr_pts_per_elem, n_spaces, \
                                 integr_pt_nat_coords_elem, integr_pt_weight_elem):
    # get integration points' natural coords and volumes
    integr_pt_nat_coords = np.zeros((n_elems * n_integr_pts_per_elem, n_spaces))
    integr_pt_weight = np.zeros(n_elems * n_integr_pts_per_elem)
    for i_elem in range(n_elems):
        start_idx = i_elem * n_integr_pts_per_elem
        end_idx = (i_elem + 1) * n_integr_pts_per_elem
        integr_pt_nat_coords[start_idx:end_idx] = integr_pt_nat_coords_elem
        integr_pt_weight[start_idx:end_idx] = integr_pt_weight_elem

    # return
    return integr_pt_nat_coords, integr_pt_weight

# ******************************** get shape functions ********************************************
def get_shape_func_1d(xi):
    # shape function for the 1d bar
    shape_func = np.array([
        0.5 * (1.0 - xi),
        0.5 * (1.0 + xi)
    ])

    # return
    return shape_func

def get_shape_func_2d(xi, eta):
    # shape function for the 2d quad
    shape_func = np.array([
        (1.0 - xi) * (1.0 - eta) / 4.0,
        (1.0 + xi) * (1.0 - eta) / 4.0,
        (1.0 + xi) * (1.0 + eta) / 4.0,
        (1.0 - xi) * (1.0 + eta) / 4.0
    ])

    # return
    return shape_func

# *************** get first spatial derivatives of shape functions in natural space ***************
def get_der1_nat_shape_func_1d():
    # first spatial derivatives of the shape functions for the 1d bar
    # (natural space)
    der1_nat_shape_func = np.array([
        [-0.5, 0.5]
    ])

    # return
    return der1_nat_shape_func

def get_der1_nat_shape_func_2d(xi, eta):
    # first spatial derivatives of the shape functions for the 2d quad
    # (natural space)
    der1_nat_shape_func = 0.25 * np.array([
        [-(1.0 - eta), 1.0 - eta, 1.0 + eta, -(1.0 + eta)],
        [-(1.0 - xi), -(1.0 + xi), 1.0 + xi, 1.0 - xi]
    ])

    # return
    return der1_nat_shape_func

# ************************************ get Jacobian ***********************************************
def get_Jacobian(der1_nat_shape_func, elem_node_coord):
    # get Jacobian
    Jacob = np.dot(der1_nat_shape_func, elem_node_coord)

    # return
    return Jacob

# *************************************************************************************************
# get integration points shape functions, their spatial derivatives in natural and physical space
# *************************************************************************************************
def get_integr_pt_shape_func_1d(n_integr_pts, n_nodes_per_elem, n_spaces, n_elems, \
                            elem_node_coord, n_integr_pts_per_elem, integr_pt_nat_coords, \
                            integr_pt_weight):
    # integration points' shape functions
    integr_pt_shape_func = np.zeros((n_integr_pts, n_nodes_per_elem))

    # integration points' first spatial derivatives of shape functions in natural space
    integr_pt_der1_nat_shape_func = np.zeros((n_integr_pts, n_spaces, n_nodes_per_elem))

    # integration points' Jacobian (for mapping from natural to physical space)
    integr_pt_Jacob = np.zeros((n_integr_pts, n_spaces, n_spaces))

    # integration points' inverse of Jacobian
    integr_pt_inv_Jacob = np.zeros((n_integr_pts, n_spaces, n_spaces))

    # integration points' determinant of Jacobian
    integr_pt_det_Jacob = np.zeros(n_integr_pts)

    # integration points' volumes
    integr_pt_vol = np.zeros(n_integr_pts)

    # integration points' first spatial derivatives of shape functions in physcial space
    integr_pt_der1_phys_shape_func = np.zeros((n_integr_pts, n_spaces, n_nodes_per_elem))

    # initialise counter
    i_integr_pt_count = 0

    # loop for all elements
    for i_elem in range(n_elems):
        # an element node coord
        an_elem_node_coord = elem_node_coord[i_elem,:,:]

        for i_integr_pt in range(n_integr_pts_per_elem):
            # natural coords, shape functions and first spatial derivatives
            # of the shape function in natural space
            # natural coords
            xi = integr_pt_nat_coords[i_integr_pt_count, 0]
            
            # get shape function
            a_shape_func = get_shape_func_1d(xi)
            integr_pt_shape_func[i_integr_pt_count, :] = a_shape_func
            
            # get first spatial derivatives of the shape functions in natural space
            a_der1_nat_shape_func = get_der1_nat_shape_func_1d()
            integr_pt_der1_nat_shape_func[i_integr_pt_count, :, :] = a_der1_nat_shape_func

            # get Jacobian
            a_Jacob = get_Jacobian(a_der1_nat_shape_func, an_elem_node_coord)
            integr_pt_Jacob[i_integr_pt_count, :, :] = a_Jacob
        
            # get inverse of Jacobian
            an_inv_Jacob = np.linalg.inv(a_Jacob)
            integr_pt_inv_Jacob[i_integr_pt_count, :, :] = an_inv_Jacob

            # get determinant of Jacobian
            a_det_Jacob = np.linalg.det(a_Jacob)
            integr_pt_det_Jacob[i_integr_pt_count] = a_det_Jacob

            # get volume
            a_weight = integr_pt_weight[i_integr_pt_count]
            a_vol = a_det_Jacob * a_weight
            integr_pt_vol[i_integr_pt_count] = a_vol

            # get first spatial derivatives of the shape functions in physical space
            a_der1_phys_shape_func = np.dot(an_inv_Jacob, a_der1_nat_shape_func)
            integr_pt_der1_phys_shape_func[i_integr_pt_count] = a_der1_phys_shape_func

            # update counter
            i_integr_pt_count = i_integr_pt_count + 1

    # return
    return integr_pt_shape_func, integr_pt_der1_nat_shape_func, integr_pt_Jacob, \
        integr_pt_inv_Jacob, integr_pt_det_Jacob, integr_pt_vol, integr_pt_der1_phys_shape_func


def get_integr_pt_shape_func_2d(n_integr_pts, n_nodes_per_elem, n_spaces, n_elems, \
                            elem_node_coord, n_integr_pts_per_elem, integr_pt_nat_coords, \
                            integr_pt_weight, thickness):
    # integration points' shape functions
    integr_pt_shape_func = np.zeros((n_integr_pts, n_nodes_per_elem))

    # integration points' first spatial derivatives of shape functions in natural space
    integr_pt_der1_nat_shape_func = np.zeros((n_integr_pts, n_spaces, n_nodes_per_elem))

    # integration points' Jacobian (for mapping from natural to physical space)
    integr_pt_Jacob = np.zeros((n_integr_pts, n_spaces, n_spaces))

    # integration points' inverse of Jacobian
    integr_pt_inv_Jacob = np.zeros((n_integr_pts, n_spaces, n_spaces))

    # integration points' determinant of Jacobian
    integr_pt_det_Jacob = np.zeros(n_integr_pts)

    # integration points' volumes
    integr_pt_vol = np.zeros(n_integr_pts)

    # integration points' first spatial derivatives of shape functions in physcial space
    integr_pt_der1_phys_shape_func = np.zeros((n_integr_pts, n_spaces, n_nodes_per_elem))

    # initialise counter
    i_integr_pt_count = 0

    # loop for all elements
    for i_elem in range(n_elems):
        # an element node coord
        an_elem_node_coord = elem_node_coord[i_elem,:,:]

        for i_integr_pt in range(n_integr_pts_per_elem):
            # natural coords, shape functions and first spatial derivatives
            # of the shape function in natural space
            # natural coords
            xi = integr_pt_nat_coords[i_integr_pt_count, 0]
            eta = integr_pt_nat_coords[i_integr_pt_count, 1]

            # get shape function            
            a_shape_func = get_shape_func_2d(xi, eta)
            integr_pt_shape_func[i_integr_pt_count, :] = a_shape_func

            # get first spatial derivatives of the shape functions in natural space
            a_der1_nat_shape_func = get_der1_nat_shape_func_2d(xi, eta)
            integr_pt_der1_nat_shape_func[i_integr_pt_count, :, :] = a_der1_nat_shape_func

            # get Jacobian
            a_Jacob = get_Jacobian(a_der1_nat_shape_func, an_elem_node_coord)
            integr_pt_Jacob[i_integr_pt_count, :, :] = a_Jacob
        
            # get inverse of Jacobian
            an_inv_Jacob = np.linalg.inv(a_Jacob)
            integr_pt_inv_Jacob[i_integr_pt_count, :, :] = an_inv_Jacob

            # get determinant of Jacobian
            a_det_Jacob = np.linalg.det(a_Jacob)
            integr_pt_det_Jacob[i_integr_pt_count] = a_det_Jacob

            # get area
            a_weight = integr_pt_weight[i_integr_pt_count]
            an_area = a_det_Jacob * a_weight

            # get volume
            a_vol = an_area * thickness
            integr_pt_vol[i_integr_pt_count] = a_vol

            # get first spatial derivatives of the shape functions in physical space
            a_der1_phys_shape_func = np.dot(an_inv_Jacob, a_der1_nat_shape_func)
            integr_pt_der1_phys_shape_func[i_integr_pt_count] = a_der1_phys_shape_func

            # update counter
            i_integr_pt_count = i_integr_pt_count + 1

    # return
    return integr_pt_shape_func, integr_pt_der1_nat_shape_func, integr_pt_Jacob, \
        integr_pt_inv_Jacob, integr_pt_det_Jacob, integr_pt_vol, integr_pt_der1_phys_shape_func

# **************************** get strain displacement matrix *************************************
def get_strain_disp_mtrx_1d(n_spaces, n_dofs_per_node, n_dofs_per_elem, der1_phys_shape_func, \
                            n_nodes_per_elem):
    # strain - displacement matrix

    strain_disp_mtrx = np.zeros((n_spaces, n_dofs_per_elem))

    i_dof_per_node = 1
    for i_space in range(1, n_spaces + 1):
        strain_disp_mtrx[i_space - 1, i_dof_per_node - 1::n_dofs_per_node] = \
            der1_phys_shape_func[i_space - 1, :n_nodes_per_elem]
        
        # update
        i_dof_per_node = i_dof_per_node + 1    

    return strain_disp_mtrx


def get_strain_disp_mtrx_2d(n_spaces, n_dofs_per_node, n_dofs_per_elem, \
                            der1_phys_shape_func, n_nodes_per_elem):
    # strain - displacement matrix

    strain_disp_mtrx = np.zeros((n_spaces + 1, n_dofs_per_elem))

    i_dof_per_node = 1
    for i_space in range(1, n_spaces + 1):
        strain_disp_mtrx[i_space - 1, i_dof_per_node - 1::n_dofs_per_node] = \
            der1_phys_shape_func[i_space - 1, :n_nodes_per_elem]
        
        # update 
        i_dof_per_node = i_dof_per_node + 1
    
    strain_disp_mtrx[n_spaces, ::n_dofs_per_node] = \
	    der1_phys_shape_func[1, :n_nodes_per_elem]
    strain_disp_mtrx[n_spaces, 1::n_dofs_per_node] = \
	    der1_phys_shape_func[0, :n_nodes_per_elem]
    
    return strain_disp_mtrx

# ******************** get integration points' strain - displacement matrix ***********************
# integration points' strain - displacement matrix
def get_integr_pt_strain_disp_mtrx_1d(n_integr_pts, n_spaces, n_dofs_per_elem, \
                integr_pt_der1_phys_shape_func, n_dofs_per_node, n_nodes_per_elem):
    integr_pt_strain_disp_mtrx = np.zeros((n_integr_pts, n_spaces, n_dofs_per_elem))

    for i_integr_pt in range(n_integr_pts):
        a_der1_phys_shape_func = integr_pt_der1_phys_shape_func[i_integr_pt,:,:]
        a_strain_disp_mtrx = \
            get_strain_disp_mtrx_1d(n_spaces, n_dofs_per_node, n_dofs_per_elem, a_der1_phys_shape_func, n_nodes_per_elem)
        integr_pt_strain_disp_mtrx[i_integr_pt,:,:] = a_strain_disp_mtrx 

    # return
    return integr_pt_strain_disp_mtrx


# integration points' strain - displacement matrix
def get_integr_pt_strain_disp_mtrx_2d(n_integr_pts, n_spaces, n_dofs_per_elem, \
                integr_pt_der1_phys_shape_func, n_dofs_per_node, n_nodes_per_elem):
    integr_pt_strain_disp_mtrx = np.zeros((n_integr_pts, n_spaces+1, n_dofs_per_elem))

    for i_integr_pt in range(n_integr_pts):
        a_der1_phys_shape_func = integr_pt_der1_phys_shape_func[i_integr_pt,:,:]
        a_strain_disp_mtrx = \
            get_strain_disp_mtrx_2d(n_spaces, n_dofs_per_node, n_dofs_per_elem, \
                            a_der1_phys_shape_func, n_nodes_per_elem)
        integr_pt_strain_disp_mtrx[i_integr_pt,:,:] = a_strain_disp_mtrx 

    # return
    return integr_pt_strain_disp_mtrx

# *************************************************************************************************
# *************************************************************************************************
# =================================================================================================