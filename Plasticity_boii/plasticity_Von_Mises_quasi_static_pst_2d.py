# ========================================= import libraries ======================================
# import system
import sys

#speed?
import jax

# import operating system
import os

# import mesh class
from mesh import Mesh

# import numpy
import numpy as np

# import common library
from common_calcs_lib import *

# import plasticity Von Mises library
from plasticity_Von_Mises_lib import *

def forward_model(params, mesh):

    youngs_modulus, poisson_ratio, yield_strs = params[:,0]

    hardening_modulus = 0

    # ===================================== set up pecision ===========================================
    # increase the precision
    np.set_printoptions(precision=16)
    #np.set_printoptions(precision=5, linewidth=200)
    from decimal import Decimal, getcontext 
    getcontext().prec = 16

    # ======================================== inputs =================================================
    # example case
    #example_case = "2d plane strain"
    example_case = "2d plane stress"
    my_mesh = Mesh(mesh)
    node_coord, elem_conn, elem_spacing, thickness, restrained_dof, force_ext_dof, force_ext_mag = my_mesh.meshgrid()
    # geometry

    # node coords
    # node_coord = np.array([[0.00, 0.00],
    #                         [1.00, 0.00],
    #                         [2.00, 0.00],
    #                         [0.00, 1.00],
    #                         [1.00, 1.00],
    #                         [2.00, 1.00]])

    # node_coord = np.array([[ 0.,  0.],
    #    [ 1.,  0.],
    #    [ 2.,  0.],
    #    [ 3.,  0.],
    #    [ 4.,  0.],
    #    [ 5.,  0.],
    #    [ 0.,  1.],
    #    [ 1.,  1.],
    #    [ 2.,  1.],
    #    [ 3.,  1.],
    #    [ 4.,  1.],
    #    [ 5.,  1.],
    #    [ 0.,  2.],
    #    [ 1.,  2.],
    #    [ 2.,  2.],
    #    [ 3.,  2.],
    #    [ 4.,  2.],
    #    [ 5.,  2.],
    #    [ 0.,  3.],
    #    [ 1.,  3.],
    #    [ 2.,  3.],
    #    [ 3.,  3.],
    #    [ 4.,  3.],
    #    [ 5.,  3.],
    #    [ 0.,  4.],
    #    [ 1.,  4.],
    #    [ 2.,  4.],
    #    [ 3.,  4.],
    #    [ 4.,  4.],
    #    [ 5.,  4.],
    #    [ 0.,  5.],
    #    [ 1.,  5.],
    #    [ 2.,  5.],
    #    [ 3.,  5.],
    #    [ 4.,  5.],
    #    [ 5.,  5.],
    #    [ 0.,  6.],
    #    [ 1.,  6.],
    #    [ 2.,  6.],
    #    [ 3.,  6.],
    #    [ 4.,  6.],
    #    [ 5.,  6.],
    #    [ 0.,  7.],
    #    [ 1.,  7.],
    #    [ 2.,  7.],
    #    [ 3.,  7.],
    #    [ 4.,  7.],
    #    [ 5.,  7.],
    #    [ 0.,  8.],
    #    [ 1.,  8.],
    #    [ 2.,  8.],
    #    [ 3.,  8.],
    #    [ 4.,  8.],
    #    [ 5.,  8.],
    #    [ 0.,  9.],
    #    [ 1.,  9.],
    #    [ 2.,  9.],
    #    [ 3.,  9.],
    #    [ 4.,  9.],
    #    [ 5.,  9.],
    #    [ 0., 10.],
    #    [ 1., 10.],
    #    [ 2., 10.],
    #    [ 3., 10.],
    #    [ 4., 10.],
    #    [ 5., 10.],
    #    [ 0., 11.],
    #    [ 1., 11.],
    #    [ 2., 11.],
    #    [ 3., 11.],
    #    [ 4., 11.],
    #    [ 5., 11.],
    #    [ 0., 12.],
    #    [ 1., 12.],
    #    [ 2., 12.],
    #    [ 3., 12.],
    #    [ 4., 12.],
    #    [ 5., 12.],
    #    [ 0., 13.],
    #    [ 1., 13.],
    #    [ 2., 13.],
    #    [ 3., 13.],
    #    [ 4., 13.],
    #    [ 5., 13.],
    #    [ 0., 14.],
    #    [ 1., 14.],
    #    [ 2., 14.],
    #    [ 3., 14.],
    #    [ 4., 14.],
    #    [ 5., 14.],
    #    [ 0., 15.],
    #    [ 1., 15.],
    #    [ 2., 15.],
    #    [ 3., 15.],
    #    [ 4., 15.],
    #    [ 5., 15.]])

    # # element connectivity
    # # elem_conn = np.array([[1, 2, 5, 4],
    # #                     [2, 3, 6, 5]])
    # elem_conn = np.array([[ 1,  2,  8,  7],
    #    [ 2,  3,  9,  8],
    #    [ 3,  4, 10,  9],
    #    [ 4,  5, 11, 10],
    #    [ 5,  6, 12, 11],
    #    [ 7,  8, 14, 13],
    #    [ 8,  9, 15, 14],
    #    [ 9, 10, 16, 15],
    #    [10, 11, 17, 16],
    #    [11, 12, 18, 17],
    #    [13, 14, 20, 19],
    #    [14, 15, 21, 20],
    #    [15, 16, 22, 21],
    #    [16, 17, 23, 22],
    #    [17, 18, 24, 23],
    #    [19, 20, 26, 25],
    #    [20, 21, 27, 26],
    #    [21, 22, 28, 27],
    #    [22, 23, 29, 28],
    #    [23, 24, 30, 29],
    #    [25, 26, 32, 31],
    #    [26, 27, 33, 32],
    #    [27, 28, 34, 33],
    #    [28, 29, 35, 34],
    #    [29, 30, 36, 35],
    #    [31, 32, 38, 37],
    #    [32, 33, 39, 38],
    #    [33, 34, 40, 39],
    #    [34, 35, 41, 40],
    #    [35, 36, 42, 41],
    #    [37, 38, 44, 43],
    #    [38, 39, 45, 44],
    #    [39, 40, 46, 45],
    #    [40, 41, 47, 46],
    #    [41, 42, 48, 47],
    #    [43, 44, 50, 49],
    #    [44, 45, 51, 50],
    #    [45, 46, 52, 51],
    #    [46, 47, 53, 52],
    #    [47, 48, 54, 53],
    #    [49, 50, 56, 55],
    #    [50, 51, 57, 56],
    #    [51, 52, 58, 57],
    #    [52, 53, 59, 58],
    #    [53, 54, 60, 59],
    #    [55, 56, 62, 61],
    #    [56, 57, 63, 62],
    #    [57, 58, 64, 63],
    #    [58, 59, 65, 64],
    #    [59, 60, 66, 65],
    #    [61, 62, 68, 67],
    #    [62, 63, 69, 68],
    #    [63, 64, 70, 69],
    #    [64, 65, 71, 70],
    #    [65, 66, 72, 71],
    #    [67, 68, 74, 73],
    #    [68, 69, 75, 74],
    #    [69, 70, 76, 75],
    #    [70, 71, 77, 76],
    #    [71, 72, 78, 77],
    #    [73, 74, 80, 79],
    #    [74, 75, 81, 80],
    #    [75, 76, 82, 81],
    #    [76, 77, 83, 82],
    #    [77, 78, 84, 83],
    #    [79, 80, 86, 85],
    #    [80, 81, 87, 86],
    #    [81, 82, 88, 87],
    #    [82, 83, 89, 88],
    #    [83, 84, 90, 89],
    #    [85, 86, 92, 91],
    #    [86, 87, 93, 92],
    #    [87, 88, 94, 93],
    #    [88, 89, 95, 94],
    #    [89, 90, 96, 95]])
    # elem_spacing = 1 
    # thickness = 1
    # restrained_dof = np.array([  1,   2,   4,   3,   5,   7,   9,  13,  15,  17,  19,  21,  25,
    #     27,  29,  31,  33,  37,  39,  41,  43,  45,  49,  51,  53,  55,
    #     57,  61,  63,  65,  67,  69,  73,  75,  77,  79,  81,  85,  87,
    #     89,  91,  93,  97,  99, 101, 103, 105, 109, 111, 113, 115, 117,
    #    121, 123, 125, 127, 129, 133, 135, 137, 139, 141, 145, 147, 149,
    #    151, 153, 157, 159, 161, 163, 165, 169, 171, 173, 175, 177])
    # force_ext_dof = np.array([182, 184, 186, 188, 190, 192])
    # force_ext_mag = np.array([0.0005, 0.001 , 0.001 , 0.001 , 0.001 , 0.0005])

    # number of spaces
    n_spaces = 2

    # element spacing    
    # elem_spacing = 1.0


    # get domain geometry
    # 2d quad
    node_coord, elem_conn, n_nodes, n_dofs, elem_dof, elem_spacing, n_spaces, \
        n_integr_pts_per_elem, n_integr_pts, n_strns_per_integr_pt, n_strs_per_integr_pt, \
        n_nodes_per_elem, n_dofs_per_elem, n_dofs_per_node, elem_node_coord, n_dofs, dof, n_elems = \
            get_domain_geo_2d(example_case, node_coord, elem_conn, elem_spacing)


    # material properties

    # elasticity parameters
    # (Youngs_modulus)
    #youngs_modulus

    # (Poisson ratio)
    #poisson_ratio = 0.3


    # plasticity parameters
    # (number of hardening points)
    n_hardening_pts = 2

    # (equivalent plastic strain - hardening point)
    hardening_pt_eqn_plc_strn = np.array([0.0, 1.0])

    # (yield stress - hardening point)
    hardening_pt_yield_strs = np.array([yield_strs, yield_strs + hardening_modulus])

    # hardening modulus 
    # hardening_modulus = (500.0-200.0)/(1.0-0.0) = 300.0
    # yield_strs = hardening_pt_yield_strs[0] = 200.0

    #
    # parameters for identification
    # 1. youngs modulus (youngs_modulus): 210000.0
    # 2. hardening modulus (hardening_modulus): 300.0
    # 3. yield stress when plastic strain has just started to evolve (yield_strs): 200.0



    # section properties

    # thickness
    # if example_case == "2d plane strain":
    #     thickness = 1.0
    # elif example_case == "2d plane stress":
    #     thickness = 0.1

    # get restraints
    # restrainted degrees of freedom
    # restrained_dof = np.array([1, 2, 7, 8])

    # get external force
    force_ext_case = "concentrated forces"

    # concentrated forces
    # force external magnitude
    #force_ext_mag = np.array([0.5, 1, 1, 1, 0.5])
    # degrees of freedom which the external force is applied
    # force_ext_dof = np.array([12])


    # get solution algorithm case. "load control" or "displacement control"
    #sol_alg_case = "load control"
    sol_alg_case = "displacement control"
    if sol_alg_case == "load control":
        # load factor (load control only)
        # number of time steps per loading
        n_time_steps_per_loading = 1000 # 10 #1000
        # (loading only)
        load_fact = np.linspace(0, 1.0, n_time_steps_per_loading+1)

    elif sol_alg_case == "displacement control":
        # controlled displacement (displacement control only)
        disp_trgt_dof_id = force_ext_dof[-1]
        # number of time steps per loading
        n_time_steps_per_loading = 50
        # max displacement target
        disp_trgt_max = 0.1
        #disp_trgt_max = -0.02
        # (loading only)
        disp_trgt = np.linspace(0, disp_trgt_max, n_time_steps_per_loading+1)    
        # (loading - unloading)
        #array1 = np.linspace(0.0, disp_trgt_max, n_time_steps_per_loading+1)
        #array2 = np.linspace(disp_trgt_max-(disp_trgt_max/n_time_steps_per_loading), 0.0, n_time_steps_per_loading)
        #disp_trgt = np.concatenate((array1, array2))
        # (loop)
        #array1 = np.linspace(0.0, disp_trgt_max, n_time_steps_per_loading+1)
        #array2 = np.linspace(disp_trgt_max-(disp_trgt_max/n_time_steps_per_loading), 0.0, n_time_steps_per_loading)
        #array3 = np.linspace(0.0-(disp_trgt_max/n_time_steps_per_loading), -disp_trgt_max, n_time_steps_per_loading)
        #array4 = np.linspace(-disp_trgt_max+(disp_trgt_max/n_time_steps_per_loading), disp_trgt_max, 2*n_time_steps_per_loading)
        #disp_trgt = np.concatenate((array1, array2, array3, array4))



    # get Newton - Raphson parameters
    # (max number of Newton-Raphson iterations)
    n_max_NR_iters = 20

    # (Newton-Raphson tolerance)
    tol_NR = 10**(-6)


    # (for plotting)
    # store degree of freedom id
    strg_dof_id = disp_trgt_dof_id
    # store node id
    strg_node_id = n_nodes

    # store integration point id
    strg_integr_pt_id = 1

    # ======================================= outputs =================================================
    # ******************************* set up domain & parameters **************************************
    # get external force load factor at time t
    if sol_alg_case == "load control":
        disp_trgt = np.zeros(load_fact.shape[0])
    elif sol_alg_case == "displacement control":
        load_fact = np.zeros(disp_trgt.shape[0])

    # get number of time steps
    n_time_steps = load_fact.shape[0]-1

    # get the number of restrained dofs
    n_restrained_dofs = restrained_dof.shape[0]

    # get the free dofs
    free_dof = np.setdiff1d(dof,restrained_dof)

    # get the number of free dofs
    n_free_dofs = free_dof.shape[0]

    # get the free dof id of the target displacement (displacement control only)
    disp_trgt_free_dof_id = \
        (np.where(disp_trgt_dof_id == free_dof)[0][0]+1) if sol_alg_case == "displacement control" else None

    # get Gauss points natural coords and weights
    integr_pt_nat_coords_elem, integr_pt_weight_elem = Gauss_pt_nat_coords_weights_2d()

    # get integration points natural coords and weights
    integr_pt_nat_coords, integr_pt_weight = \
        get_integr_pt_nat_coords_weights(n_elems, n_integr_pts_per_elem, n_spaces, \
                                    integr_pt_nat_coords_elem, integr_pt_weight_elem)

    # get integration points shape functions, their spatial derivatives in natural and physical space
    integr_pt_shape_func, integr_pt_der1_nat_shape_func, integr_pt_Jacob, \
        integr_pt_inv_Jacob, integr_pt_det_Jacob, integr_pt_vol, integr_pt_der1_phys_shape_func = \
            get_integr_pt_shape_func_2d(n_integr_pts, n_nodes_per_elem, n_spaces, n_elems, \
                                elem_node_coord, n_integr_pts_per_elem, integr_pt_nat_coords, \
                                integr_pt_weight, thickness)


    # integration points' strain - displacement matrix
    integr_pt_strain_disp_mtrx = \
        get_integr_pt_strain_disp_mtrx_2d(n_integr_pts, n_spaces, n_dofs_per_elem, \
                integr_pt_der1_phys_shape_func, n_dofs_per_node, n_nodes_per_elem)


    # unfactored external load
    force_ext_I_unfact = np.zeros(n_dofs)
    force_ext_I_unfact[force_ext_dof-1] = force_ext_mag

    # free unfactored external load
    force_ext_I_f_unfact = force_ext_I_unfact[free_dof-1]

    # get restrained unfactored external load
    force_ext_I_r_unfact = force_ext_I_unfact[restrained_dof-1]

    # ******************************** initialise variables *******************************************
    # get field and integration points' variables for storing
    disp_I_strg, force_ext_I_strg, Cauchy_strn_tnr_strg, Cauchy_strn_elc_tnr_strg, \
    Cauchy_strn_plc_tnr_strg, Cauchy_strn_plc_eqn_strg, Cauchy_strs_tnr_strg, \
    Cauchy_strs_eqn_strg, has_yielded_strg, sufail_strg = \
        get_fld_integr_pt_strg(n_time_steps, n_strns_per_integr_pt, \
                                n_strs_per_integr_pt)

    # initialise field quantities
    disp_I_tp1, force_ext_I_tp1 = init_fld_vars(n_dofs, n_nodes)

    # initialise integration points quantities
    integr_pt_Cauchy_strn_elc_tnr_tp1, integr_pt_Cauchy_strn_plc_tnr_tp1, \
            integr_pt_Cauchy_strn_plc_eqn_tp1, integr_pt_Cauchy_strs_tnr_tp1, \
            integr_pt_plc_multiplier_ttp1_inc, integr_pt_has_yielded_tp1, \
            integr_pt_sufail_tp1 = \
            init_integr_pt_vars(n_integr_pts, n_strns_per_integr_pt, \
                                n_strs_per_integr_pt)

    # ********************************* time integration **********************************************

    # time step
    time_step = 1.0

    # current time: t = 0
    time_tp1 = 0.0

    # time step counter
    i_time_step = 0

    # loop over all time steps
    while i_time_step < n_time_steps:
        # find equilibrium
        #   at time t + 1 (i_time_step+1)
        #   from time t (i_time_step)

        # update time step counter
        i_time_step = i_time_step + 1

        # time: t
        time_t = time_tp1

        # time: t+1
        time_tp1 = time_t + time_step

        # print time increment
        #print("\nTime increment:",time_t,"to",time_tp1)

        # update variables
        # Cauchy elastic strain tensor at time t 
        integr_pt_Cauchy_strn_elc_tnr_t = integr_pt_Cauchy_strn_elc_tnr_tp1

        # Cauchy plastic strain tensor at time t+1
        integr_pt_Cauchy_strn_plc_tnr_t = integr_pt_Cauchy_strn_plc_tnr_tp1

        # Cauchy equivalent plastic strain at time t
        integr_pt_Cauchy_strn_plc_eqn_t = integr_pt_Cauchy_strn_plc_eqn_tp1
        
        # Cauchy stress tensor at time t
        integr_pt_Cauchy_strs_tnr_t = integr_pt_Cauchy_strs_tnr_tp1

        # incremental plastic multiplier from time t to t+1
        integr_pt_plc_multiplier_ttp1_inc = np.zeros(n_integr_pts)

        # has yielded flag at time t)
        integr_pt_has_yielded_t = integr_pt_has_yielded_tp1
        
        # (has failed flag at time t)
        integr_pt_sufail_t = integr_pt_sufail_tp1

        # displacement at time t
        disp_I_t = disp_I_tp1

        # initialise the incremental displacement field from time t to t+1
        disp_I_ttp1_inc = np.zeros(n_dofs)

        # external force at time t
        force_ext_I_t = force_ext_I_tp1
        
        # initialise the free incremental internal force from time t to t+1
        force_int_I_f_ttp1_inc = np.zeros(n_free_dofs)

        if sol_alg_case == "load control": # load control
            # external force load factor at time t 
            load_fact_t = load_fact[i_time_step-1]

            # external force load factor at time t+1
            load_fact_tp1 = load_fact[i_time_step]

            # incremental external force load factor from time t to t+1
            load_fact_ttp1_inc = load_fact_tp1 - load_fact_t

            # get loading/ unloading flag
            is_unloading_tp1 = False
            if i_time_step != 1:
                # external force load factor at time t-1
                load_fact_tm1 = load_fact[i_time_step-2]

                # incremental external force load factor from time t-1 to t
                load_fact_tm1t_inc = load_fact_t - load_fact_tm1

                # get the loading/ unloading flag
                if ((load_fact_tm1t_inc*load_fact_ttp1_inc) < 0.0):
                    is_unloading_tp1 = True

            # get the external force from time t to t + 1
            force_ext_I_ttp1_inc = load_fact_ttp1_inc * force_ext_I_unfact

            # get the free external force from time t to t + 1
            force_ext_I_f_ttp1_inc = load_fact_ttp1_inc * force_ext_I_f_unfact

            # set an initial guess of the iterative free residual force from time t to t+1
            force_res_I_f_ttp1_iter = force_int_I_f_ttp1_inc - force_ext_I_f_ttp1_inc

        elif sol_alg_case == "displacement control": # displacement control       
            # external force load factor at time t+1 (initial guess set as zero)
            load_fact_tp1 = 0.0

            # incremental external force load factor from time t to t+1
            load_fact_ttp1_inc = 0.0

            # target displacement at time t
            disp_trgt_t = disp_trgt[i_time_step-1]

            # target displacement at time t+1
            disp_trgt_tp1 = disp_trgt[i_time_step]

            # incremental target displacement from time t to t+1
            disp_trgt_ttp1_inc = disp_trgt_tp1 - disp_trgt_t

            # get loading/ unloading flag
            is_unloading_tp1 = False
            if i_time_step != 1:
                # target displacement at time t-1
                disp_trgt_tm1 = disp_trgt[i_time_step-2]

                # incremental target displacement from time t-1 to t
                disp_trgt_tm1t_inc = disp_trgt_t - disp_trgt_tm1

                # get the loading/ unloading flag
                if ((disp_trgt_tm1t_inc*disp_trgt_ttp1_inc) < 0.0):
                    is_unloading_tp1 = True


        # Newton-Raphson iterations in this time step
        
        # initialise iteration index
        i_NR_iter = 0
        
        # initialise convergence & divergence flag as false
        has_convrged = False
        has_diverged = False

        # initialise relative & maximum residual norm
        ratold = 0.0
        remold = 0.0

        # loop until converge or reach the max number of iterations
        while i_NR_iter < n_max_NR_iters and has_convrged == False:
            # update Newton-Raphson counter
            i_NR_iter = i_NR_iter + 1
            
            # print current / max iterations
            #print("Current iteration / Max iterations:", i_NR_iter, "/", n_max_NR_iters)

            # get loading/ unloading flag (reset)
            is_unloading_tp1 = False if i_NR_iter > 1 else is_unloading_tp1

            # elastic or plastic constitutive matrix flag
            cond_flag = (~integr_pt_has_yielded_tp1) | (is_unloading_tp1 == 1)
            integr_pt_plc_const_mtrx_flag_tp1 = np.where(cond_flag, False, True)

            # get stiffness matrix at time t+1
            stiff_mtrx_IJ_tp1 = get_stiff_mtrx_IJ_tp1_2d(example_case, n_elems, elem_dof, n_dofs_per_elem, n_integr_pts_per_elem, \
                                        integr_pt_strain_disp_mtrx, integr_pt_Cauchy_strn_plc_eqn_tp1, \
                                        integr_pt_plc_multiplier_ttp1_inc, integr_pt_Cauchy_strs_tnr_tp1, \
                                        integr_pt_plc_const_mtrx_flag_tp1, youngs_modulus, poisson_ratio, \
                                        hardening_pt_eqn_plc_strn, hardening_pt_yield_strs, \
                                        n_dofs, integr_pt_vol)

            # get the free-free stiffness matrix at time t+1
            stiff_mtrx_IJ_ff_tp1 = stiff_mtrx_IJ_tp1[free_dof-1][:, free_dof-1]

            # get the free iterative displacement field from time t to t+1
            if sol_alg_case == "load control": # load control
                # get the free iterative displacement field from time t to t+1
                disp_I_f_ttp1_iter = np.linalg.solve(stiff_mtrx_IJ_ff_tp1, -1.0 * force_res_I_f_ttp1_iter)
            elif sol_alg_case == "displacement control": # displacement control
                # get the free unfactored iterative displacement field from time t to t+1
                disp_I_f_unfact_ttp1_iter = np.linalg.solve(stiff_mtrx_IJ_ff_tp1, 1.0 * force_ext_I_f_unfact)

                # get the unfactored iterative displacement at the target dof from time t to t+1
                disp_unfact_ttp1_iter = disp_I_f_unfact_ttp1_iter[disp_trgt_free_dof_id-1]

                # get the free iterative displacement field from time t to t+1
                if i_NR_iter == 1:
                    # iterative external force load factor from time t to t+1
                    load_fact_ttp1_iter = disp_trgt_ttp1_inc/disp_unfact_ttp1_iter

                    # incremental external force load factor from time t to t+1
                    load_fact_ttp1_inc = load_fact_ttp1_inc + load_fact_ttp1_iter

                    # incremental free external force from time t to t+1
                    force_ext_I_f_ttp1_inc = load_fact_ttp1_inc*force_ext_I_f_unfact

                    # incremental restrained force from time t to t+1
                    force_ext_I_r_ttp1_inc = load_fact_ttp1_inc*force_ext_I_r_unfact

                    # incremental external force from time t to t+1
                    force_ext_I_ttp1_inc = np.zeros(n_dofs)
                    force_ext_I_ttp1_inc[free_dof-1] = force_ext_I_f_ttp1_inc
                    force_ext_I_ttp1_inc[restrained_dof-1] = force_ext_I_r_ttp1_inc

                    # get the free iterative displacement field from time t to t+1
                    disp_I_f_ttp1_iter = load_fact_ttp1_inc*disp_I_f_unfact_ttp1_iter

                else:
                    # get the free residual iterative displacement field from time t to t+1
                    disp_I_f_res_ttp1_iter = np.linalg.solve(stiff_mtrx_IJ_ff_tp1, -1.0 * force_res_I_f_ttp1_iter)

                    # get the residual iterative displacement at the target dof from time t to t+1
                    disp_res_ttp1_iter = disp_I_f_res_ttp1_iter[disp_trgt_free_dof_id-1]

                    # iterative external force load factor from time t to t+1
                    load_fact_ttp1_iter = -1.0 * disp_res_ttp1_iter/disp_unfact_ttp1_iter

                    # incremental external force load factor from time t to t+1
                    load_fact_ttp1_inc = load_fact_ttp1_inc + load_fact_ttp1_iter

                    # iterative external force load factor from time t to t+
                    force_ext_I_f_ttp1_iter = load_fact_ttp1_iter*force_ext_I_f_unfact

                    # incremental free external force from time t to t+1
                    force_ext_I_f_ttp1_inc = load_fact_ttp1_inc*force_ext_I_f_unfact

                    # incremental restrained force from time t to t+1
                    force_ext_I_r_ttp1_inc = load_fact_ttp1_inc*force_ext_I_r_unfact

                    # incremental external force from time t to t+1
                    force_ext_I_ttp1_inc = np.zeros(n_dofs)
                    force_ext_I_ttp1_inc[free_dof-1] = force_ext_I_f_ttp1_inc
                    force_ext_I_ttp1_inc[restrained_dof-1] = force_ext_I_r_ttp1_inc

                    # get the free iterative new residual force from time t to t+1
                    force_res_ext_I_f_ttp1_iter = \
                        (-1.0 * force_res_I_f_ttp1_iter + force_ext_I_f_ttp1_iter)
                    
                    # get the free iterative displacement field from time t to t+1
                    disp_I_f_ttp1_iter = np.linalg.solve(stiff_mtrx_IJ_ff_tp1, force_res_ext_I_f_ttp1_iter)
                
        
            # get the restrained iterative displacement field from time t to t+1
            disp_I_r_ttp1_iter = np.zeros(n_restrained_dofs)

            # get the iterative displacement field from time t to t+1
            disp_I_ttp1_iter = np.zeros(n_dofs)
            disp_I_ttp1_iter[free_dof-1] = disp_I_f_ttp1_iter
            disp_I_ttp1_iter[restrained_dof-1] = disp_I_r_ttp1_iter

            # get the incremental displacement field from time t to t+1
            # (add the extra iterative displacement)
            disp_I_ttp1_inc = disp_I_ttp1_inc + disp_I_ttp1_iter

            # get the incremental internal force from time t to t+1
            integr_pt_Cauchy_strn_tnr_tp1, integr_pt_Cauchy_strn_elc_tnr_tp1, integr_pt_Cauchy_strn_plc_tnr_tp1, \
            integr_pt_Cauchy_strn_plc_eqn_tp1, integr_pt_plc_multiplier_ttp1_inc, integr_pt_Cauchy_strs_tnr_tp1, \
            integr_pt_Cauchy_strs_eqn_tp1, integr_pt_has_yielded_tp1, integr_pt_sufail_tp1, force_int_I_ttp1_inc = \
                    get_force_int_I_ttp1_inc_2d(n_dofs, n_integr_pts, n_strns_per_integr_pt, n_strs_per_integr_pt, n_elems,
                    elem_dof, disp_I_ttp1_inc, n_dofs_per_elem,
                    n_integr_pts_per_elem, integr_pt_strain_disp_mtrx, integr_pt_Cauchy_strn_elc_tnr_t,
                    integr_pt_Cauchy_strn_plc_tnr_t, integr_pt_Cauchy_strn_plc_eqn_t, youngs_modulus, poisson_ratio,
                    hardening_pt_eqn_plc_strn, hardening_pt_yield_strs, integr_pt_Cauchy_strs_tnr_t,
                    integr_pt_vol)

            # get the free incremental free internal force vector from time t to t+1
            force_int_I_f_ttp1_inc = force_int_I_ttp1_inc[free_dof-1]

            # get the free incremental restrainted internal force vector from time t to t+1
            force_int_I_r_ttp1_inc = force_int_I_ttp1_inc[restrained_dof-1]

            # get the iterative free residual force vector from time t to t+1
            force_res_I_f_ttp1_iter = force_int_I_f_ttp1_inc - force_ext_I_f_ttp1_inc

            # get the reaction & residual forces
            if sol_alg_case == "load control": # load control
                # get the iterative reaction forces from time t to t+1
                force_rec_I_ttp1_inc = np.zeros(n_dofs)
                force_rec_I_ttp1_inc[free_dof-1] = force_ext_I_f_ttp1_inc
                force_rec_I_ttp1_inc[restrained_dof-1] = force_int_I_r_ttp1_inc
                
                # get the iterative residual forces from time t to t+1
                # (no contribution of reactions)
                force_res_no_rec_I_ttp1_iter = force_rec_I_ttp1_inc - force_int_I_ttp1_inc
            elif sol_alg_case == "displacement control": # displacement control
                # get the iterative reaction forces from time t to t+1
                force_rec_I_ttp1_inc = np.zeros(n_dofs)
                #force_rec_I_ttp1_inc[disp_trgt_dof_id-1] = force_int_I_f_ttp1_inc[disp_trgt_free_dof_id-1]
                force_rec_I_ttp1_inc[free_dof-1] = force_ext_I_f_ttp1_inc
                force_rec_I_ttp1_inc[restrained_dof-1] = force_int_I_r_ttp1_inc

                # get the iterative residual forces from time t to t+1
                # (no contribution of reactions)
                force_res_no_rec_I_ttp1_iter = force_rec_I_ttp1_inc - force_int_I_ttp1_inc

            # get the residual force norm
            resid = np.sum(force_res_no_rec_I_ttp1_iter ** 2)
            retot = \
                np.sum(force_rec_I_ttp1_inc * force_rec_I_ttp1_inc)

            # get the maximum nodal residual
            agash = np.abs(force_res_no_rec_I_ttp1_iter)
            remax = np.max(agash)
            
            # get the Euclidean norm of residual
            resid = np.sqrt(resid)
            
            # get the Euclidean norm of external force
            retot = np.sqrt(retot)

            # compute relative residual norm
            r100 = 100.0
            r0 = 0.0
            if retot == r0:
                ratio = r0
            else:
                ratio = \
                    r100 * resid / retot

            # set convergence/divergence flags
            r1000 = 1000.0
            r20 = 20.0
            if ratio <= tol_NR or np.abs(remax) <= (tol_NR / r1000):
                has_convrged = True
            if i_NR_iter != 1 and ((ratio > r20 * ratold) or (remax > r20 * remold)):
                has_diverged = True
            # relative residual norm
            ratold = ratio
            # maximum residual norm
            remold = remax

            # print relative & maximum residual norm
            #print("Relative residual norm:", ratold, \", Maximum residual norm:", remold)

            # convergence check
            if has_convrged == True:
                # print has converged message
                #print("Has converged:", has_convrged)
                break

        # update variables
        if has_convrged == True:
            # update total displacement field at time  t+1
            disp_I_tp1 = disp_I_t + disp_I_ttp1_inc
                    
            # update total external force at time t+1
            force_ext_I_tp1 = force_ext_I_t + force_ext_I_ttp1_inc

            # get the external force load factor (displacement control only)
            if sol_alg_case == "displacement control": # displacement control
                # get external force load factor at time t
                load_fact_t = load_fact[i_time_step-1]
                            
                # get external force load factor at time t+1
                load_fact_tp1 = load_fact_t + load_fact_ttp1_inc
                load_fact[i_time_step] = load_fact_tp1

            # store field and integration point variables
            disp_I_strg, force_ext_I_strg, Cauchy_strn_tnr_strg, Cauchy_strn_elc_tnr_strg, \
            Cauchy_strn_plc_tnr_strg, Cauchy_strn_plc_eqn_strg, Cauchy_strs_tnr_strg, \
            Cauchy_strs_eqn_strg, has_yielded_strg, sufail_strg = \
                store_fld_integr_pt_strg(i_time_step, strg_dof_id, strg_integr_pt_id, \
                    disp_I_strg, force_ext_I_strg, Cauchy_strn_tnr_strg, Cauchy_strn_elc_tnr_strg, \
                    Cauchy_strn_plc_tnr_strg, Cauchy_strn_plc_eqn_strg, Cauchy_strs_tnr_strg, \
                    Cauchy_strs_eqn_strg, has_yielded_strg, sufail_strg, \
                    disp_I_tp1, force_ext_I_tp1, integr_pt_Cauchy_strn_tnr_tp1, \
                    integr_pt_Cauchy_strn_elc_tnr_tp1, integr_pt_Cauchy_strn_plc_tnr_tp1, \
                    integr_pt_Cauchy_strn_plc_eqn_tp1, integr_pt_Cauchy_strs_tnr_tp1, \
                    integr_pt_Cauchy_strs_eqn_tp1, integr_pt_has_yielded_tp1, \
                    integr_pt_sufail_tp1)
    
    # epsilon_xx = Cauchy_strn_tnr_strg[:, 1]
    # sigma_xx = Cauchy_strs_tnr_strg[:, 1]

    line = []
    for i in range(len(disp_I_strg)):
        line.append([1000*disp_I_strg[i]])
        line.append([1000*force_ext_I_strg[i]])

    # normalised graph
    # for i in range(len(epsilon_xx)):
    #     line.append([100*epsilon_xx[i]])
    #     line.append([100*sigma_xx[i]])
    
    #return disp_I_tp1
    return np.array(line)
# ************************************* plots ****************************************************
# plots

# Plot the data with triangular markers and blue color
#plt.plot(disp_I_strg, force_ext_I_strg, marker='^', color='blue', linestyle='None')

# Add labels and title
#plt.xlabel('Displacement, [mm]')
#plt.ylabel('External Force, [N]')
#plt.title('Plot of Displacement vs External Force')

# Show grid
#plt.grid(True)

# Show legend
#plt.legend()

# Show the plot
#plt.show()


# plots

#epsilon_xx = Cauchy_strn_tnr_strg[:, 0]
#sigma_xx = Cauchy_strs_tnr_strg[:, 0]

# Plot the data
#plt.plot(epsilon_xx, sigma_xx, marker='<', color='red', linestyle='None')

# Add labels and title
#plt.xlabel('Total strain xx, [-]')
#plt.ylabel('Total stress xx, [N/mm^2]')
#plt.title('Plot of Strain vs Stress')

# Show grid
#plt.grid(True)

# Show legend
#plt.legend()

# Show the plot
# plt.show()

# =================================================================================================