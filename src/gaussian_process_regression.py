"""
Gaussian Process Regression for single value output
"""

import sys
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np
from create_data import *
from kernels import *
from scipy.optimize import minimize



# define all hyperparameters for all kernels
params = {'var':   0.5,
          'ell':   2,        # larger l -> less wiggles
          'var_b': 2,
          'off':   0.5,
          'per':   2}

# define all kernels
#all_kernels = [kernel_linear, kernel_rbf, kernel_add, kernel_mult]
all_kernels = [kernel_periodic, kernel_mult2]

def create_test_data(N_s: int = 50, start: int = 0, end: int = 15):
    """ Create test set as N_s points from start to end 
    """
    x_s = np.linspace(start, end, N_s).reshape(-1,1)
    return x_s

def get_prior(kernel = kernel_rbf, x_s = np.array):
    """ Compute mean and covariance of test set using kernel to define Prior
    """
    mu_prior = np.zeros(x_s.shape)
    cov_prior = kernel(x_s, x_s, params)
    return mu_prior, cov_prior

def plot_prior(kernel = kernel_rbf, x_s = np.array):
    """ Sample 5 normals from GP prior and plot
    """
    mu_prior, cov_prior = get_prior(kernel, x_s)
    f_prior = np.random.multivariate_normal(mu_prior.ravel(), cov_prior, 5)
    plt.figure(figsize=(12,6))
    for sample in f_prior:
        plt.plot(x_s, sample, lw=1.5, ls='-')
        plt.title(f'Single output - Prior using {kernel.__name__}')
    plt.xlabel('x')
    plt.ylabel('f')
    plt.tight_layout()        
    plt.savefig(f'figures/results/single_output_gpr_prior_{kernel.__name__}.png')

def get_posterior(x, x_s, f, kernel = kernel_rbf, params = params, noise = 1e-2):
    """ Derive Posterior Distribution using Equation (1)
    
    Parameters:
    x : numpy array with N dimensions of 1 element
        Training Data inputs 
    x_s : numpy array with N dimensions of 1 element
        Test Data inputs
    f : numpy array with N dimensions of 1 element
        Training Data outputs
    kernel : function (default: rbf kernel)
        Kernel function to compute covariance matrix
    params : dictionary
        Hyperparameters for kernel functions
    noise : float
        Noise to ensure that the matrix is not singular
        
    Returns:
    mu_s : numpy array with N dimensions of 1 element
        Mean vector of posterior distribution
    sigma_s : numpy array with N x N dimensions
        Covariance matrix of posterior distribution
    """
    N = len(x)
    N_s = len(x_s)
    cov = kernel(x, x, params) + noise**2 * np.eye(N)
    cov_s = kernel(x, x_s, params) + 1e-5 * np.eye(N,N_s)
    cov_ss = kernel(x_s, x_s, params) + 1e-5 * np.eye(N_s)
    
    mu_s = cov_s.T.dot(np.linalg.inv(cov)).dot(f)
    sigma_s = cov_ss - cov_s.T.dot(np.linalg.inv(cov)).dot(cov_s)
    
    return mu_s, sigma_s

def plot_posterior(x, x_s, f, kernel=kernel_rbf, noise = 1e-2):
    """ Plot GP with mean function & uncertainty
    Adapted from: https://github.com/krasserm/bayesian-machine-learning/blob/dev/gaussian-processes/gaussian_processes_util.py
    
    Parameters:
    mu : numpy array with N dimensions of 1 element
        Mean vector of (posterior) distribution
    cov : numpy array with N x N dimensions
        Covariance matrix of posterior distribution
    X : numpy array with N dimensions of 1 element
        Test Data inputs
    X_train : numpy array with N dimensions of 1 element
        Training Data inputs
    Y_train : numpy array with N dimensions of 1 element
        Training Data outputs
    samples : numpy array with 5 x N dimensions
        Drawn samples from GP
    ax0 : plot
        Plot to draw GP on
    pred : int (default: 7)
        Specifies a test point with its prediction
    """
    mu, cov = get_posterior(x, x_s, f, kernel, params, noise)
    x_s = x_s.ravel()
    mu = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov))

    samples = np.random.multivariate_normal(mu, cov, 5)
        
    plt.figure(figsize=(12, 6))
    plt.fill_between(x_s, mu + uncertainty, mu - uncertainty, alpha=0.2)
    plt.plot(x_s, mu, '--', color='darkblue', lw=3, label='Mean')
    for i, sample in enumerate(samples):
        plt.plot(x_s, sample, lw=1.5)
    if x is not None:
        plt.plot(x, f, 'o', ms=8, color='darkblue')
    plt.legend()
    plt.title(f'Single output - Posterior using {kernel.__name__}')
    plt.xlabel('x')
    plt.ylabel('f')
    plt.tight_layout()        
    plt.savefig(f'figures/results/single_output_gpr_posterior_noisy_{kernel.__name__}.png')


def nll(x, f, kernel=kernel_rbf, noise=1e-2):
    """ Compute negative log marginal likelihood, naive implementation of Equation (2)
    to optimize hyperparameters of kernels
    
    Parameters:
    X : numpy array with N x 1 dimensions
        Training data input
    f : numpy array N x 1
        Training data output
    kernel : function (default: rbf kernel)
        Kernel function for which the parameters should be optimized
    noise : float
        Ensure that the matrix calculations work
        
    Returns:
    nll_naive 
    
    """
    
    f = f.ravel()
    N = len(x)
    def nll_naive(theta):
        params = {'var': theta[0],
                  'ell': theta[1],        
                  'var_b': theta[2],
                  'off': theta[3]}
        cov_y = kernel(x, x, params) + noise**2 * np.eye(N)
        return 0.5 * f.dot(np.linalg.inv(cov_y).dot(f)) + \
               0.5 * np.log(np.linalg.det(cov_y)) + \
               0.5 * N * np.log(2*np.pi)
    
    return nll_naive

def optimized_posterior(x, f, kernel, noise, x_s):
    res = minimize(nll(x, f, kernel=kernel, noise=noise), [1, 1, 1, 1],
                   bounds=((1e-2, None), (1e-2, None), (1e-2, None), (1e-2, None)),
                   method='L-BFGS-B')
    # compute posterior with optimized parameters
    var_opt, ell_opt, var_b_opt, off_opt = res.x
    params = {'var': var_opt,
              'ell': ell_opt,        
              'var_b': var_b_opt,
              'off': off_opt}
    mu_s, cov_s = get_posterior(x, x_s, f, kernel, params=params, noise=noise)
    return mu_s, cov_s


def loocv(x, x_s, f, kernel=kernel_rbf, noise=1e-2):
    """ Leave-one-out cross-validation to compare model performances
    For each input data point, remove it from training data, then:
        Optimize hyperparameters of specified kernel 
        Derive posterior distribution with optimized parameters
        Obtain prediction for left-out data point
    
    Parameters:
    X : numpy array N x 1
        Training data input
    f : numpy array N x 1
        Training data output
    kernel : function (default: rbf kernel)
        Kernel function that specifies respective model
        
    Returns:
    predictions : list
        List of predictions of left-out training data points    
    """
    predictions = []
    N = len(x) - 1
    for leave_out in range(N+1):
        X_new = np.delete(x, leave_out).reshape(-1,1)
        f_new = np.delete(f, leave_out).reshape(-1,1)

        mu_s, cov_s = optimized_posterior(X_new, f_new, kernel, noise, x_s)
        
        # prediction for left out data point
        idx_pred = np.absolute(x_s-x[leave_out]).argmin()
        predictions.append(mu_s.item(idx_pred))

    # L2 distance
    l2dist = np.linalg.norm(f.reshape(1,-1) - predictions)

    return predictions, l2dist


def plot_all(x, x_s, y, noise):
    """ Plot prior distributions and posterior distributions (without + with noise) 
    and save in figures/results
    Use all available kernels
    """
    for kernel in all_kernels:
        plot_prior(kernel, x_s)
        plot_posterior(x, x_s, y, kernel)
        plot_posterior(x, x_s, y, kernel, noise)


    

def main():
    # import data
    [x_true,y_true], [x_data,y_data] = create_single_output()
    #x_s = x_true.reshape(-1,1)
    x_s = create_test_data(N_s = 100, start = 0, end = 10).reshape(-1,1)
    x = x_data.reshape(-1,1)
    y = y_data.reshape(-1,1)
    
    # assume noise
    noise = 0.3

    # plot priors, posteriors, and noisy posteriors for all kernels
    plot_all(x, x_s, y, noise)

    # compute LOOCV for all kernels (with optimized parameters and assumed noise)
    #for kernel in all_kernels:
    #    pred, l2 = loocv(x, x_s, y, kernel, noise)
    #    print(f'\nPredictions for {kernel.__name__}:', pred, '\nL2 distance: ', l2)
    


if __name__ == "__main__":
    sys.exit(main())