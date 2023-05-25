# IOE 511/MATH 562, University of Michigan
# Code written by: Albert S. Berahas & Jiahao Shi

import numpy as np
import scipy.stats as stats
import scipy.sparse as sparse
import scipy.io
# Define all the functions and calculate their gradients and Hessians, those functions include:
# (1)(2)(3)(4) Quadractic function
# (5)(6) Quartic function
# (7)(8) Rosenbrock function 
# (9) Data fit
# (10)(11) Exponential

 
# Problem Number: 1
# Problem Name: quad_10_10
# Problem Description: A randomly generated convex quadratic function; the 
#                      random seed is set so that the results are 
#                      reproducable. Dimension n = 10; Condition number
#                      kappa = 10

# function that computes the function value of the quad_10_10 function

def quad_10_10_func(x):
    # set raondom seed
    np.random.seed(0)
    # Generate random data
    q = np.random.normal(size=(10,1))

    mat = scipy.io.loadmat('quad_10_10_Q.mat')
    Q = np.array(mat['Q'].todense())
    
    # compute function value
    return (1/2*x.T@Q@x + q.T@x)[0]

def quad_10_10_grad(x):
    # set raondom seed
    np.random.seed(0)
    # Generate random data
    q = np.random.normal(size=(10,1))
    mat = scipy.io.loadmat('quad_10_10_Q.mat')
    Q = np.array(mat['Q'].todense())
    
    return Q @ x + q   
    

def quad_10_10_Hess(x):
    # set raondom seed
    np.random.seed(0)
    # Generate random data
    q = np.random.normal(size=(10,1))
    mat = scipy.io.loadmat('quad_10_10_Q.mat')
    Q = np.array(mat['Q'].todense())
    
    return Q

# Problem Number: 2
# Problem Name: quad_10_1000
# Problem Description: A randomly generated convex quadratic function; the 
#                      random seed is set so that the results are 
#                      reproducable. Dimension n = 10; Condition number
#                      kappa = 1000

# function that computes the function value of the quad_10_1000 function

 

def quad_10_1000_func(x):
    # set raondom seed
    np.random.seed(0)
    # Generate random data
    q = np.random.normal(size=(10,1))

    mat = scipy.io.loadmat('quad_10_1000_Q.mat')
    Q = np.array(mat['Q'].todense())
    
    # compute function value
    return (1/2*x.T@Q@x + q.T@x)[0]

def quad_10_1000_grad(x):
    # set raondom seed
    np.random.seed(0)
    # Generate random data
    q = np.random.normal(size=(10,1))
    mat = scipy.io.loadmat('quad_10_1000_Q.mat')
    Q = np.array(mat['Q'].todense())
    
    return Q@x + q   

def quad_10_1000_Hess(x):
    mat = scipy.io.loadmat('quad_10_1000_Q.mat')
    Q = mat['Q'].todense()
    
    return Q

# Problem Number: 3
# Problem Name: quad_1000_10
# Problem Description: A randomly generated convex quadratic function; the 
#                      random seed is set so that the results are 
#                      reproducable. Dimension n = 1000; Condition number
#                      kappa = 10

# function that computes the function value of the quad_1000_10 function

def quad_1000_10_func(x):
    # set raondom seed
    np.random.seed(0)
    # Generate random data
    q = np.random.normal(size=(1000,1))

    mat = scipy.io.loadmat('quad_1000_10_Q.mat')
    Q = np.array(mat['Q'].todense())
    
    # compute function value
    return (1/2*x.T@Q@x + q.T@x)[0]

def quad_1000_10_grad(x):
    # set raondom seed
    np.random.seed(0)
    # Generate random data
    q = np.random.normal(size=(1000,1))
    mat = scipy.io.loadmat('quad_1000_10_Q.mat')
    Q = np.array(mat['Q'].todense())
    
    return Q@x + q   
    

def quad_1000_10_Hess(x):
    mat = scipy.io.loadmat('quad_1000_10_Q.mat')
    Q = np.array(mat['Q'].todense())
    
    return Q

# Problem Number: 4
# Problem Name: quad_1000_1000
# Problem Description: A randomly generated convex quadratic function; the 
#                      random seed is set so that the results are 
#                      reproducable. Dimension n = 1000; Condition number
#                      kappa = 1000

# function that computes the function value of the quad_10_10 function

def quad_1000_1000_func(x):
    # set raondom seed
    np.random.seed(0)
    # Generate random data
    q = np.random.normal(size=(1000,1))
    
    mat = scipy.io.loadmat('quad_1000_1000_Q.mat')
    Q = np.array(mat['Q'].todense())
    print(Q)
    
    # compute function value
    return (1/2*x.T@Q@x + q.T@x)[0]

def quad_1000_1000_grad(x):
    # set raondom seed
    np.random.seed(0)
    # Generate random data
    q = np.random.normal(size=(1000,1))
    mat = scipy.io.loadmat('quad_1000_1000_Q.mat')
    Q = np.array(mat['Q'].todense())
    
    return Q@x + q   
    

def quad_1000_1000_Hess(x):
    mat = scipy.io.loadmat('quad_1000_1000_Q.mat')
    Q = np.array(mat['Q'].todense())
    
    return Q


# Problem Number: 5
# Problem Name: quartic_1
# Problem Description: A quartic function. Dimension n = 4

# function that computes the function value of the quartic_1 function

def quartic_1_func(x):
    Q = np.array([[5,1,0,0.5],
     [1,4,0.5,0],
     [0,0.5,3,0],
     [0.5,0,0,2]])
    sigma = 1e-4
    
    return 1/2*(x.T @x) + sigma/4*(x.T@Q@x)**2

def quartic_1_grad(x):
    Q = np.array([[5,1,0,0.5],
     [1,4,0.5,0],
     [0,0.5,3,0],
     [0.5,0,0,2]])
    sigma = 1e-4

    return x + sigma * (x.T @ Q @ x) * (Q @ x)

def quartic_1_Hess(x):
    Q = np.array([[5,1,0,0.5],
     [1,4,0.5,0],
     [0,0.5,3,0],
     [0.5,0,0,2]])
    sigma = 1e-4

    Qx = Q @ x
    return np.identity(4) + sigma * (np.outer(Qx, Qx) + (x.T @ Q @ x) * Q)



# Problem Number: 6
# Problem Name: quartic_2
# Problem Description: A quartic function. Dimension n = 4

# function that computes the function value of the quartic_2 function


def quartic_2_func(x):
    Q = np.array([[5,1,0,0.5],
     [1,4,0.5,0],
     [0,0.5,3,0],
     [0.5,0,0,2]])
    sigma = 1e4
    
    return 1/2*(x.T@x) + sigma/4*(x.T@Q@x)**2

def quartic_2_grad(x):
    Q = np.array([[5,1,0,0.5],
     [1,4,0.5,0],
     [0,0.5,3,0],
     [0.5,0,0,2]])
    sigma = 1e4

    return x + sigma * (x.T @ Q @ x) * (Q @ x)

def quartic_2_Hess(x):
    Q = np.array([[5,1,0,0.5],
     [1,4,0.5,0],
     [0,0.5,3,0],
     [0.5,0,0,2]])
    sigma = 1e4

    Qx = Q @ x
    return np.identity(4) + sigma * (np.outer(Qx, Qx) + (x.T @ Q @ x) * Q)


# Problem Number: 7
# Problem Name: rosenbrock_2
def rosenbrock_2_func(x):
    return (1-x[0])**2 + 100*(x[1] - x[0]**2)**2


def rosenbrock_2_grad(x):
    w, z = x[0], x[1]
    return np.array([-2*(1-w) - 400*w*(z-w**2), 200*(z-w**2)])


def rosenbrock_2_Hess(x):
    w, z = x[0], x[1]
    return np.array([[2-400*(z-3*w**2), -400*w], [-400*w, 200]])

# Problem Number: 8
# Problem Name: rosenbrock_100
def rosenbrock_100_func(x):
    f = 0
    for i in np.arange(100-1):
        x1, x2 = x[i], x[i+1]
        f += (1-x1)**2 + 100*(x2 - x1**2)**2
    return f


def rosenbrock_100_grad(x):
    g = np.zeros(100)
    for i in np.arange(100-1):
        x1, x2 = x[i], x[i+1]
        g[i] = g[i] + -2*(1-x1) - 400*x1*(x2-x1**2)
        g[i+1] = g[i+1] + 200*(x2-x1**2)
    return g


def rosenbrock_100_Hess(x):
    H = np.zeros((100, 100))
    for i in np.arange(100-1):
        x1, x2 = x[i], x[i+1]
        H[i, i] = H[i, i] + 2-400*(x2-3*x1**2)
        H[i, i+1] = H[i, i+1] - 400 * x1
        H[i+1, i] = H[i+1, i] - 400 * x1
        H[i+1, i+1] = H[i+1, i+1] + 200
    return H


# Problem Number: 9
# Problem Name: datafit_2
def datafit_2_func(x):
    w, z = x[0], x[1]
    y1, y2, y3 = 1.5, 2.25, 2.625
    t = np.array([y1 - w * (1-z), y2 - w * (1-z**2), y3 - w*(1-z**3)])
    return np.power(t, 2).sum()

def datafit_2_grad(x):
    w, z = x[0], x[1]
    y1, y2, y3 = 1.5, 2.25, 2.625

    t1 = np.array([y1 - w * (1-z), y2 - w * (1-z**2), y3 - w*(1-z**3)])
    t2 = np.array([1-z, 1-z**2, 1-z**3])
    t3 = np.array([w, 2 * w * z, 3 * w * z**2])
    return np.array([-2 * sum(t1 * t2), 2 * sum(t1 * t3)])

def datafit_2_Hess(x):
    w, z = x[0], x[1]
    y1, y2, y3 = 1.5, 2.25, 2.625

    a1 = np.array([1-z, 1-z**2, 1-z**3])
    a2 = np.array([1, 2*z, 3 * z**2])

    t1 = np.array([y1-w*(1-z), y2-w*(1-z**2), y3-w*(1-z**3)])

    h11 = 2 * np.sum(np.power(a1, 2))
    h12 = 2 * np.sum(-a1 * (w*a2) + (t1 * a2))
    h22 = 2 * (w**2 + 6 * np.power(w, 2) * np.power(z, 2) + 15 * np.power(w, 2) * np.power(z, 4))

    return np.array([[h11, h12], [h12, h22]])


# Problem Number: 10
# Problem Name: exponential_10
def exponential_10_func(x):
    z1 = x[0]
    return (np.exp(z1)-1)/(np.exp(z1)+1) + 0.1 * np.exp(-z1) + np.sum(np.power(x[1:]-1, 4))

def exponential_10_grad(x):
    z1 = x[0]
    grad1 = 2 * np.exp(z1)/(np.exp(z1)+1)**2 - 0.1 * np.exp(-z1)
    grad2 = 4 * np.power(x[1:]-1, 3)
    return np.append(grad1, grad2)

def exponential_10_Hess(x):
    z1 = x[0]
    hess1 = -2 * np.exp(z1) * (np.exp(z1)-1)/(np.exp(z1)+1)**3 + 0.1 * np.exp(-z1)
    hess2 = 12 * (x[1:]-1)**2
    return np.diag(np.append(hess1, hess2))


# Problem Number: 11
# Problem Name: exponential_1000
def exponential_1000_func(x):
    z1 = x[0]
    return (np.exp(z1)-1)/(np.exp(z1)+1) + 0.1 * np.exp(-z1) + np.sum(np.power(x[1:]-1, 4))

def exponential_1000_grad(x):
    z1 = x[0]
    grad1 = 2 * np.exp(z1)/(np.exp(z1)+1)**2 - 0.1 * np.exp(-z1)
    grad2 = 4 * np.power(x[1:]-1, 3)
    return np.append(grad1, grad2)

def exponential_1000_Hess(x):
    z1 = x[0]
    hess1 = -2 * np.exp(z1) * (np.exp(z1)-1)/(np.exp(z1)+1)**3 + 0.1 * np.exp(-z1)
    hess2 = 12 * (x[1:]-1)**2
    return np.diag(np.append(hess1, hess2))


# Problem Number: 12
# Problem Name: genhumps_5
# Problem Description: A multi-dimensional problem with a lot of humps.
#                      This problem is from the well-known CUTEr test set.

# function that computes the function value of the genhumps_5 function
def genhumps_5_func(x):
    f = 0
    for i in range(4):
        f = f + np.sin(2*x[i])**2*np.sin(2*x[i+1])**2 + 0.05*(x[i]**2 + x[i+1]**2)
    return f

# function that computes the gradient of the genhumps_5 function

def genhumps_5_grad(x):
    g = [4*np.sin(2*x[0])*np.cos(2*x[0])* np.sin(2*x[1])**2                  + 0.1*x[0],
         4*np.sin(2*x[1])*np.cos(2*x[1])*(np.sin(2*x[0])**2 + np.sin(2*x[2])**2) + 0.2*x[1],
         4*np.sin(2*x[2])*np.cos(2*x[2])*(np.sin(2*x[1])**2 + np.sin(2*x[3])**2) + 0.2*x[2],
         4*np.sin(2*x[3])*np.cos(2*x[3])*(np.sin(2*x[2])**2 + np.sin(2*x[4])**2) + 0.2*x[3],
         4*np.sin(2*x[4])*np.cos(2*x[4])* np.sin(2*x[3])**2                  + 0.1*x[4]]
    
    return np.array(g)

# function that computes the Hessian of the genhumps_5 function
def genhumps_5_Hess(x):
    H = np.zeros((5,5))
    H[0,0] =  8* np.sin(2*x[1])**2*(np.cos(2*x[0])**2 - np.sin(2*x[0])**2) + 0.1
    H[0,1] = 16* np.sin(2*x[0])*np.cos(2*x[0])*np.sin(2*x[1])*np.cos(2*x[1])
    H[1,1] =  8*(np.sin(2*x[0])**2 + np.sin(2*x[2])**2)*(np.cos(2*x[1])**2 - np.sin(2*x[1])**2) + 0.2
    H[1,2] = 16* np.sin(2*x[1])*np.cos(2*x[1])*np.sin(2*x[2])*np.cos(2*x[2])
    H[2,2] =  8*(np.sin(2*x[1])**2 + np.sin(2*x[3])**2)*(np.cos(2*x[2])**2 - np.sin(2*x[2])**2) + 0.2
    H[2,3] = 16* np.sin(2*x[2])*np.cos(2*x[2])*np.sin(2*x[3])*np.cos(2*x[3])
    H[3,3] =  8*(np.sin(2*x[2])**2 + np.sin(2*x[4])**2)*(np.cos(2*x[3])**2 - np.sin(2*x[3])**2) + 0.2
    H[3,4] = 16* np.sin(2*x[3])*np.cos(2*x[3])*np.sin(2*x[4])*np.cos(2*x[4])
    H[4,4] =  8* np.sin(2*x[3])**2*(np.cos(2*x[4])**2 - np.sin(2*x[4])**2) + 0.1
    H[1,0] = H[0,1]
    H[2,1] = H[1,2]
    H[3,2] = H[2,3]
    H[4,3] = H[3,4]
    return H