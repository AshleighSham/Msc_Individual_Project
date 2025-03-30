## Table of contents

- [Table of contents](#table-of-contents)
- [Project Overview](#project-overview)
- [Abstract](#abstract)
- [Repository Structure](#repository-structure)
- [Relevant References](#relevant-references)

## Project Overview

This project was for my Master's in Predictive Modeling nads Scientific Computing degree, it studies the application of bayesian inference into the subject of material parameter estimation.

## Abstract

The Bayesian Inference approach, incorporating Markov Chain Monte Carlo (MCMC) sampling methods and the Ensemble Kalman Filter (EnKF), is applied to identify various material parameters in linear and nonlinear solid mechanics problems, encompassing homogeneous and heterogeneous models. The Finite Element Method (FEM) is utilised in the forward models, employing quadrilateral element mesh grids. Observation data, synthetically generated to enable a rigorous assessment of the algorithms' effectiveness, are further perturbed with additive Gaussian noise to simulate the model uncertainty typically encountered in real-world scenarios. The study implements the Ensemble Kalman Filter and the Metropolis-Hastings algorithm, with a particular focus on proposal kernel adaptation and delayed rejection variants. The computational time and accuracy of these algorithms are assessed to evaluate their overall performance.

## Repository Structure

This GitHub Repository is first split into the different solid mechanics models studied within this project:

- Elastcity_1D: simple 1-dimensional linear elasticity problem (1 material parameter)
- Elasticity_2D: Extension of the previous model into a 2-dimensional space under the plane stress assumption (2 material parameters)
- Elasticty_2D_Beam: 2-dimensional model for assigned beam structure of a composite material (4 material parameters)
- Elasticity_2D_RVE: 2-dimensional model for a 3x3 RVE of a composite material (4 material parameters)
- Plasticity_RVE: 2-dimensional plasticity hardening model on a composite RVE (8 material parameters)
- Plasticty_hardening: 2-dimensional plasticity hardening model (4 material parameters)
- Plasticty_Perfectly: 2-dimensional perfectly plasticity model (3 material parameters)

Within in each folder their contains a

- config.json file which allows for centralised input of new matrial parameters, chosen sampling method and corresponding mcmc variable values
- config.py which import the config.json file
- A main file that houses the running of the simulation and estimation which is labelled elasticity_2d.py for the elastcity problems and plastcity_2d.py for the plasticty ones.
- mesh.py that build the matrials mesh and runs the elasticty equations, for the plastcity model due to its complexitty it occurs across 3 extra files named common_calcs_lib.py, platicty_Von_Mises_lib.py and plasticity_von_mises_quasi_static_pst_2d.py
- utilities.py containing functions used across the files
- Sensitivity.py for evaluating the sensitivities of the material parameters within the models.
- MH.py, AMH.py, MH_DR.py, DRAM.py, EnKF.py each conating the corresponding sampling method

## Relevant References

The plasiticty model in this code is a Python conversion of that outlined in Computational Methods for Plasticity: Theory and Applications.
