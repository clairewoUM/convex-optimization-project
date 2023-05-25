# IOE 511/MATH 562, University of Michigan
# Code written by: Katherine Ahn & Seonho Woo

# Function that runs a chosen algorithm on a chosen problem
#           Inputs: problem, method, options (structs)
#           Outputs: final iterate (x) and final function value (f)


import numpy as np
import algorithms 
import logging 
import time

def optSolver_SUU511(problem,method,options):
    start = time.time()
    logging.debug(f'Problem:{problem.name}, Method: {method.name}')

    # compute initial function/gradient/Hessian
    x = problem.x0
    f = problem.compute_f(x) 
    g = problem.compute_g(x)
    H = problem.compute_H(x)

    norm_0 = np.linalg.norm(g,ord=np.inf)   # Norm of the initial gradient
    norm_g = np.linalg.norm(g,ord=np.inf)   # norm of the gradient

    # Create storage for functional values, norm of the gradient
    fk = np.array(f)
    gk = np.array(norm_0)

    # Storage for s, y(LBFGS)
    s = []
    y = []

    # set initial iteration counter
    k = 0

    while  k < (options.max_iterations):
        # Termination condition
        if norm_g <= options.term_tol * max(norm_0, 1):
            break

        # Algorithms
        if method.name.startswith('GradientDescent'):
            x_new, f_new, g_new, d, alpha = algorithms.GDStep(x, f, g, problem, method, options)
        
        elif method.name.startswith('Newton'):
            x_new, f_new, g_new, d, alpha = algorithms.NewtonStep(x, f, g, problem, method, options)

        elif method.name.startswith("BFGS"):
            # Initialize h matrix at the first iterate
            # here, H is an approximate inverted Hessian matrix
            if k==0:
                if method.hess_mode == 0 or method.hess_mode == 'T':
                    H = np.identity(problem.n) 
                else:
                    H = np.identity(problem.n) * method.hess_mode

            # update
            x_new, f_new, g_new, H_new, d, alpha = algorithms.BFGS(x, f, g, H, k, problem, method, options)

            # Update inverted hessian matrix
            H = H_new

        elif method.name.startswith("LBFGS"):
            m = method.m    # memory length
            # Initialize h matrix at the first iterate
            if k==0:
                if method.hess_mode == 0 or method.hess_mode == 'T':
                    h0 = np.identity(problem.n) 
                else:
                    h0 = np.identity(problem.n) * method.hess_mode

            x_new, f_new, g_new, d, s, y, alpha = algorithms.LBFGS(x, f, g, h0, k, m, s, y, problem, method, options)

        elif method.name.startswith("DFP"):
            # Initialize H matrix at the first iterate
            # Here, H is an approximate inverted Hessian matrix
            if k==0:
                H = np.identity(problem.n)
            # update
            x_new, f_new, g_new, H_new, d, alpha = algorithms.DFP(x, f, g, H, k, problem, method, options)

            # Update inverted hessian matrix
            H = H_new

        elif method.name == 'TRNewtonCG':
            # initialize trust region radius
            if k==0:
                radius = 1
            x_new, f_new, g_new, H_new, radius, rho = algorithms.trust_region_newton_cg(f, g, H, x, radius, problem, method, options)

            # Update hessian matrix
            H = H_new

        elif method.name == 'TRSR1CG': 
            # Initialize Hessian approximation for SR1 and trust region radius
            if k == 0: 
                H = np.identity(problem.n)
                radius = 1
            x_new,f_new, g_new, H_new, radius, rho =  algorithms.trust_region_quasi_newton_cg(f, g, H, x, radius, problem, method, options, eta = 1e-6, r = 1e-6) 

            # Update Hessian matrrix
            H = H_new
        else:
            print('Warning: method is not implemented yet')

        # update old and new function values
        x_old = x; f_old = f; g_old = g; norm_g_old = norm_g
        x = x_new; f = f_new; g = g_new; norm_g = np.linalg.norm(g,ord=np.inf)

        # Print out intermediate results
        if method.name == 'TRNewtonCG' or method.name == 'TRSR1CG':
            logging.debug(f'iteration:{k:3}, function value: {round(f.item(),3):6}, gradient norm: {round(norm_g.item(),3):6}, radius={radius}')
        else:
            logging.debug(f'iteration:{k:3}, function value: {round(f.item(),3):6}, gradient norm: {round(norm_g.item(),3):6}, alpha={alpha}')

        # increment iteration counter
        k = k + 1

        # store the results
        fk = np.append(fk, f.item())
        gk = np.append(gk, norm_g)
        t = time.time() - start
        
    # return x, f, norm_g, fk, gk, k, round(t, 5)
    return x, f