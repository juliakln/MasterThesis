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
    
    Parameters:
    x : numpy array with N dimensions of 1 element
        First input vector of kernel
    y : numpy array with N dimensions of 1 element
        Second input vector of kernel
    param : dictionary
        Contains scale factor variance, variance_b, and offset off
        
    Returns: 
        Covariance matrix of each pairwise combination of set of points
    """
    variance = param['var']
    variance_b = param['var_b']
    offset = param['off']
    return variance_b + variance * np.dot((x-offset), (y-offset).T)


def kernel_periodic(x, y, param):
    """ Periodic Kernel
    
    Parameters:
    x : numpy array with N dimensions of 1 element
        First input vector of kernel
    y : numpy array with N dimensions of 1 element
        Second input vector of kernel
    param : dictionary
        Contains scale factor variance, lengthscale, and period (distance between repetitions)
        
    Returns: 
        Covariance matrix of each pairwise combination of set of points
    """
    variance = param['var']
    lengthscale = param['ell']
    period = param['per']
    #return variance * np.exp((-(2*np.sin(np.dot(np.pi/period, np.linalg.norm((x.reshape(-1,1)-y.reshape(-1,1)), ord=1)))) /(lengthscale))**2)
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

