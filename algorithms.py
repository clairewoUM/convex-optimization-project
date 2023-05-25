# IOE 511/MATH 562, University of Michigan
# Code written by: Katherine Ahn & Seonho Woo

# Compute the next step for all iterative optimization algorithms given current solution x:
import numpy as np
import logging

def Armijo(x, f, g, d, problem, options):
    '''
    Armijo Backtracking
    '''
    # Set up the parameters
    alpha = 1 ; c1 = options.c_1_ls
    tau = 0.5
    x_new = x + alpha * d

    # Finding the alpha
    while problem.compute_f(x_new) > f + c1 * alpha * g.T @ d:
        alpha = alpha * tau
        x_new = x + alpha * d

    return alpha, x_new

def Wolfe(x, f, g, d, problem, options):
    '''
    Wolfe line search
    '''
    # Set up parameters
    c1 = options.c_1_ls
    c2 = options.c_2_ls

    alpha = 1
    alpha_l = 0
    alpha_h = 1000
    c = 0.5

    while True:
        x_new = x + alpha * d
        # if Armijo condition holds
        if problem.compute_f(x_new) <= f + c1 * alpha * g.T @ d:

            # if curvature condition holds
            if problem.compute_g(x_new).T @ d >= c2 * g.T @ d:
                return alpha, x_new
            else:
                alpha_l = alpha
        else:
            alpha_h = alpha
        alpha = c * alpha_l + (1-c) * alpha_h 

def twoloop_recursion(g, h0, s, y, sl, m):
    '''
    LBFGS two-loop recursion
    '''
    q = g
    iter = sl if sl < m else m

    alpha = {}
    for i in np.flip(np.arange(iter)):
        rho = 1/(s[i].T @ y[i])
        alpha[i] = rho * s[i].T @ q
        q = q - alpha[i] * y[i]

    r = h0 @ q
    for i in np.arange(iter):
        rho = 1/(s[i].T @ y[i])
        beta = rho * y[i].T @ r
        r = r + s[i] * (alpha[i]-beta)

    return r


# (1) Gradient Descent
def GDStep(x,f,g,problem,method,options):
    # Set the search direction d to be -g
    d = -g

    if method.name == 'GradientDescent':
        alpha, x_new = Armijo(x, f, g, d, problem, options)
    
    elif method.name == 'GradientDescentW':
        alpha, x_new = Wolfe(x, f, g, d, problem, options)

    # Compute the new functional value, gradient
    f_new = problem.compute_f(x_new)
    g_new = problem.compute_g(x_new)

    return x_new,f_new,g_new,d,alpha

# (2) Newton`s Method
def NewtonStep(x, f, g, problem, method, options):

    # Modified Newton subroutine: Make the Hessian PD
    A = problem.compute_H(x)
    beta = 1e-3
    A_min = np.diagonal(A).min()
    eta = 0 if A_min>0 else -A_min + beta

    while True:
        try:
            L = np.linalg.cholesky(A + eta * np.identity(problem.n))
            break
        except np.linalg.LinAlgError as e:
            if 'not positive definite' in str(e):
                logging.debug('Factorization not successful')
                eta = max(2*eta, beta)

    # Search direction
    d = np.linalg.solve(L @ L.T, -g)
        
    if method.name == 'Newton':
        alpha, x_new = Armijo(x, f, g, d, problem, options)

    else:
        alpha, x_new = Wolfe(x, f, g, d, problem, options)

    # Compute the new functional value, gradient
    f_new = problem.compute_f(x_new)
    g_new = problem.compute_g(x_new)

    return x_new, f_new, g_new, d, alpha


# (3) BFGS
def BFGS(x, f, g, H, k, problem, method, options):
    # search direction where H is the inverse Hessian approx.
    d = - H @ g

    # step size computation: Armijo backtracking
    if method.name == 'BFGS':
        alpha, x_new = Armijo(x, f, g, d, problem, options)

    # step size computation: Wolfe line search
    else:
        alpha, x_new = Wolfe(x, f, g, d, problem, options)

    # Compute the new functional value, gradient
    f_new = problem.compute_f(x_new)
    g_new = problem.compute_g(x_new)

    # Updating H
    s = x_new - x
    y = g_new - g

    sTy = s.T @ y
    # logging.debug(f'iteration:{k:3}, sTy = {sTy: .2e}, g = {np.linalg.norm(g, ord=np.inf).round(3):6}, ||dk||:{np.linalg.norm(d): .3e}, H = {np.linalg.norm(H): .3e}')

    # if sTy is not sufficiently large, do not update H
    if sTy < options.term_tol * np.linalg.norm(s) * np.linalg.norm(y):
        logging.debug('skip updating H')
        return x_new, f_new, g_new, H, d, alpha
    
    # update H
    else:
        I = np.identity(problem.n)
        rho = 1 / sTy
        V = I - rho * np.outer(y, s)
        H_new = V.T @ H @ V + rho * np.outer(s, s)

        # Hessian update (6.20) from 'Numerical Optimization textbook'
        if k==0 and method.hess_mode == 'T':
            yTy = y.T @ y
            H = sTy/yTy * I
            H_new = V.T @ H @ V + rho * np.outer(s, s)
        return x_new, f_new, g_new, H_new, d, alpha


# (4) LBFGS
def LBFGS(x, f, g, h0, k, m, s, y, problem, method, options):
    '''
    arguments:
    -----
    m: memory length
    '''
    sl = len(s)

    # The first iteration(no curvature pairs)
    if sl == 0:
        d = - h0 @ g
        
        # Do line search with Armijo backtracking
        if method.name == 'LBFGS':
            alpha, x_new = Armijo(x, f, g, d, problem, options)

        # step size computation: Wolfe line search
        elif method.name == 'LBFGSW':
            alpha, x_new = Wolfe(x, f, g, d, problem, options)

        # Compute the new functional value, gradient
        f_new = problem.compute_f(x_new)
        g_new = problem.compute_g(x_new)

        # compute s, y
        sk = x_new - x
        yk = g_new - g

        sTy = sk.T @ yk
        # logging.debug(f'iteration:{k:3}, sTy = {sTy: .2e}, g = {np.linalg.norm(g, ord=np.inf).round(3):6}, ||dk||:{np.linalg.norm(d): .3e}')

        # Update s, y if sTy is sufficiently large
        if sTy >= options.term_tol * np.linalg.norm(s) * np.linalg.norm(y):
            # store new vectors
            s.append(sk)
            y.append(yk)

    # 2loop recursion
    else:
        # Hessian update (7.20) from 'Numerical Optimization textbook'
        if method.hess_mode == 'T':
            sk = s[-1]
            yk = y[-1]
            h0 = (sk.T @ yk) / (yk.T @ yk) * np.identity(problem.n)
        r = twoloop_recursion(g, h0, s, y, sl, m)

        # Search direction
        d = -r

        # step size computation: Armijo Backtracking
        if method.name == 'LBFGS':
            alpha, x_new = Armijo(x, f, g, d, problem, options)

        # step size computation: Wolfe line search
        elif method.name == 'LBFGSW':
            alpha, x_new = Wolfe(x, f, g, d, problem, options)


        # update f, g
        f_new = problem.compute_f(x_new)
        g_new = problem.compute_g(x_new)

        # compute s, y
        s_k = x_new - x
        y_k = g_new - g

        sTy = s_k.T @ y_k
        # logging.debug(f'iteration:{k:3}, sTy = {sTy: .2e}, g = {np.linalg.norm(g, ord=np.inf).round(3):6}, ||dk||:{np.linalg.norm(d): .3e}')

        # Update s, y
        # Update s, y if sTy is sufficiently large
        if sTy >= options.term_tol * np.linalg.norm(s_k) * np.linalg.norm(y_k):
            # store new vectors
            s.append(s_k)
            y.append(y_k)

            # Discard 0-th curvature pair
            if sl == m:
                s.pop(0)
                y.pop(0)

        else:
            logging.debug('skip update')


    return x_new, f_new, g_new, d, s, y, alpha

# (5) DFP
def DFP(x, f, g, H, k, problem, method, options):
    # search direction where H is the inverse Hessian approx.
    d = - H @ g

    # step size computation: Armijo backtracking
    if method.name == 'DFP':
        alpha, x_new = Armijo(x, f, g, d, problem, options)

    # step size computation: Wolfe line search
    elif method.name == 'DFPW':
        alpha, x_new = Wolfe(x, f, g, d, problem, options)

    # Compute the new functional value, gradient
    f_new = problem.compute_f(x_new)
    g_new = problem.compute_g(x_new)

    # Update H
    s = x_new - x
    y = g_new - g

    sTy = s.T @ y
    # logging.debug(f'iteration:{k:3}, sTy = {sTy: .2e}, g = {np.linalg.norm(g, ord=np.inf).round(3):6}, ||dk||:{np.linalg.norm(d): .3e}, H = {np.linalg.norm(H): .3e}')

    # if sTy is not sufficiently large, do not update H
    if sTy < options.term_tol * np.linalg.norm(s) * np.linalg.norm(y):
        logging.debug('skip updating H')

        return x_new, f_new, g_new, H, d, alpha

    # update H
    else:
        rho = 1 / sTy
        H_new = H - H @ np.outer(y, y) @ H /(y.T @ H @ y) + rho * np.outer(s, s)

        return x_new, f_new, g_new, H_new, d, alpha


def CG_steihaug(f, g, B, x, delta, options):
    """
    Conjugate Gradient Steihaug subproblem solver
    (Equation H p = -g subject to the trust region constraint ||p|| <= Delta)

    Parameters:
        - f: the objective function
        - grad: the gradient of f
        - B: the approximate Hessian matrix of f
        - x: initial point
        - delta: TR radius

    Returns:
        - search direction
    """
    z = np.zeros_like(x)
    r = g
    p = -r
    
    if np.linalg.norm(r) < options.term_tol_CG:
        return z

    for j in np.arange(options.max_iterations_CG):
        # if p.T @ B @ p <=0
        if p.T @ B @ p <= 0:
            ztp = np.dot(z.T, p)
            ptp = np.dot(p.T, p)

            tau1 = (-ztp + np.sqrt(ztp**2 - ptp * (np.dot(z.T,z)- delta**2)))/ptp
            tau2 = (-ztp - np.sqrt(ztp**2 - ptp * (np.dot(z.T,z)- delta**2)))/ptp

            d1 = z + tau1 * p
            d2 = z + tau2 * p

            if f + np.dot(g.T, d1) + 0.5 * d1.T @ B @ d1 < f + np.dot(g.T, d2) + 0.5 * d2.T @ B @ d2:
                return d1
            else:
                return d2

        alpha = (r.T @ r) / (p.T @ B @ p)
        alpha = alpha.item()

        z_new = z + alpha*p

        if np.linalg.norm(z_new) >= delta:
            ztp = np.dot(z.T, p)
            ptp = np.dot(p.T, p)

            tau1 = (-ztp + np.sqrt(ztp**2 - ptp * (np.dot(z.T,z)- delta**2)))/ptp
            tau2 = (-ztp - np.sqrt(ztp**2 - ptp * (np.dot(z.T,z)- delta**2)))/ptp

            tau1 = tau1.item()
            tau2 = tau2.item()
            d1 = z + tau1 * p
            d2 = z + tau2 * p
            if f + np.dot(g.T, d1) + 0.5 * d1.T @ B @ d1 < f + np.dot(g.T, d2) + 0.5 * d2.T @ B @ d2:
                return d1
            else:
                return d2

        r_new = r + alpha * np.matmul(B, p)

        if np.linalg.norm(r_new) <= options.term_tol_CG:
            return z_new

        beta = (r_new.T @ r_new) / (r.T @ r)
        beta = beta.item()
        p_new = -r_new + beta * p

        # update r, p, z
        r, p, z = r_new, p_new, z_new


# (6) trust_region_newton_cg
def trust_region_newton_cg(f, g, H, x, radius, problem, method, options):
    # Compute search direction
    d = CG_steihaug(f, g, H, x, radius, options)

    # calculate the predicted reduction and the actual reduction
    actual_reduction = f - problem.compute_f(x + d)
    predicted_reduction = (-1) * np.dot(g.T, d) + (-0.5)* np.dot(d.T.dot(H), d)

    # calculate the ratio of actual to predicted reduction
    rho = (actual_reduction / predicted_reduction) if predicted_reduction != 0 else 0.0

    # update the trust region radius based on the ratio rho
    if rho > options.c_1_tr:
        x = x + d
        if rho > options.c_2_tr:
            radius *= 2
    else:
        radius /= 2

    f_new = problem.compute_f(x)
    g_new = problem.compute_g(x)
    H_new = problem.compute_H(x)

    return x, f_new, g_new, H_new, radius, rho


def trust_region_quasi_newton_cg(f, g, B, x, delta, problem, method, options, eta, r):
    '''
    Algorithm 6.2 from 'Numerical Optimization' textbook
    -----
    B: Hessian Approximate
    delta: initial radius
    eta: idk
    tol_cg: tolerance for CG subproblems
    r: any small number between 0 and 1
    '''

    # compute search direction
    s = CG_steihaug(f, g, B, x, delta, options)

    # calculate yk, the predicted reduction, and the actual reduction
    yk = problem.compute_g(x + s) - g
    actual_reduction = f - problem.compute_f(x + s)
    predicted_reduction = (-1) * (np.dot(g.T, s) + 0.5* np.dot(s.T.dot(B), s) )

    # calculate the ratio of actual to predicted reduction
    rho = (actual_reduction / predicted_reduction) if predicted_reduction != 0 else 0
    # update the trust region radius based on the ratio rho
    if rho > eta:
        x = x + s

    if rho > options.c_2_tr:
        if np.linalg.norm(s) > 0.8*delta:
            delta *= 2

    elif options.c_1_tr <= rho and rho <= options.c_2_tr:
        delta = delta

    else:
        delta /= 2

    # Update Hessian matrix(SR1)
    if abs(s.T.dot(yk - B.dot(s))) >= r * np.linalg.norm(s) * np.linalg.norm(yk - B.dot(s)):
        B = B + np.outer((yk - B.dot(s)), (yk - B.dot(s)))  / np.dot((yk - B.dot(s)).T, s) 

    # Compute f, g at the updated x
    f = problem.compute_f(x)
    g = problem.compute_g(x)

    return x, f, g, B, delta, rho