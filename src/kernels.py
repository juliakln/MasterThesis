"""
Define kernels for Gaussian Process Regression
"""

import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np


def kernel_rbf(x, y, param):
    """ Radial Basis Function Kernel 
    
    Parameters:
    x : numpy array with N dimensions of 1 element
        First input vector of kernel
    y : numpy array with N dimensions of 1 element
        Second input vector of kernel
    param : dictionary
        Contains scale factor variance, and lengthscale ell
        
    Returns:
        Covariance matrix of each pairwise combination of set of points
    """
    variance = param['var']
    lengthscale = param['ell']
    # Euclidean distance between points
    eucdist = np.sum(x**2,1).reshape(-1,1) + np.sum(y**2,1) - 2*np.dot(x, y.T)
    return variance * np.exp(-0.5 * eucdist * 1/(lengthscale**2))


def kernel_linear(x, y, param):
    """ Linear Kernel
    """
    variance = param['var']
    variance_b = param['var_b']
    offset = param['off']
    return variance_b + variance * np.dot((x-offset), (y-offset).T)


def kernel_periodic(x, y, param):
    """ Periodic Kernel
    """
    variance = param['var']
    lengthscale = param['ell']
    period = param['per']
    return variance * np.exp(-(2*np.sin((np.pi * (x - y.T))/period)**2)/ (lengthscale**2))


def kernel_mult_r_l(x, y, param):
    """ Multiply RBF and Linear Kernel
    """
    return kernel_rbf(x, y, param) * kernel_linear(x, y, param)

def kernel_mult_p_l(x, y, param):
    """ Multiply Periodic and Linear Kernel
    """
    return kernel_periodic(x, y, param) * kernel_linear(x, y, param)

def kernel_mult_p_r(x, y, param):
    """ Multiply Periodic and RBF Kernel
    """
    return kernel_periodic(x, y, param) * kernel_rbf(x, y, param)

def kernel_add_r_l(x, y, param):
    """ Add RBF and Linear Kernel
    """
    return kernel_rbf(x, y, param) + kernel_linear(x, y, param)

def kernel_add_p_l(x, y, param):
    """ Multiply Periodic and Linear Kernel
    """
    return kernel_periodic(x, y, param) + kernel_linear(x, y, param)

def kernel_add_p_r(x, y, param):
    """ Multiply Periodic and RBF Kernel
    """
    return kernel_periodic(x, y, param) + kernel_rbf(x, y, param)


def kernel_rbf_ard(x, y, param):
    """ Radial Basis Function Kernel with Automatic Relevance Determination
        for 2-dimensional case
    Args:
        x: First input vector of kernel (N,2)
        y: Second input vector of kernel (N,2)
        param: Hyperparameter of kernel: scale factor variance and 2 lengthscales ell
        
    Returns:
        Covariance matrix of each pairwise combination of set of points
    """
    variance = param['var']
    lengthscale = param['ell_dim']
    eucdist = []
    # for each dimension
    for d in np.arange(2):
        eucdist.append((np.sum(x[:,d].reshape(-1,1)**2,1).reshape(-1,1) + 
                 np.sum(y[:,d].reshape(-1,1)**2,1) - 
                 2*np.dot(x[:,d].reshape(-1,1), y[:,d].reshape(-1,1).T))/(lengthscale[d]**2))
    return variance * np.exp(-0.5 * np.sum(eucdist,0))