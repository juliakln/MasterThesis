"""
Gaussian Process Regression for single value output

1) Predict values from random function

2) Predict mean from histogram

3) Predict variance from histogram

"""

import sys
import warnings
from numpy.core.fromnumeric import reshape
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np
from create_data import *
from kernels import *
from scipy.optimize import minimize
#import itertools


""" Global Parameters """

# define all hyperparameters for all kernels
params = {'var':   0.5,
          'ell':   2,        # larger l -> less wiggles
          'var_b': 2,
          'off':   0.5,
          'per':   2}

# define all kernels
all_kernels = [kernel_linear, kernel_rbf, kernel_periodic, kernel_add_p_r, kernel_add_r_l,
              kernel_mult_p_l, kernel_mult_p_r, kernel_mult_r_l]


""" Helper Functions """

def create_test_data(N_s: int = 50, start: int = 0, end: int = 15):
    """ Create test set as N_s points from start to end 
    """
    x_s = np.linspace(start, end, N_s).reshape(-1,1)
    return x_s

def comp_mean(freq, obs):
    """ Compute mean value of hisogtram with frequencies freq, and observations obs
    Frequencies must be fractions, not integers!
    """
    return np.sum(freq * obs)

def comp_sd(freq, obs, mu):
    """ Compute standard deviation of histogram with frequencies freq, observations obs, and mean value mu
    Frequencies must be fractions, not integers!
    """
    return np.sqrt(np.sum(freq * (obs - mu)**2))

def comp_margin(t, sd, N):
    """ Compute margin of error for mean of histogram with t-value t, standard deviation sd, and sample size N
    """
    return t * (sd / np.sqrt(N))



""" Analyses """

def prior(kernel = kernel_rbf, x_s = np.array, plot = True, name = 'single'):
    """ Compute mean and covariance of test set to define Prior

    Parameters:
    kernel : function
        Kernel function to compute covariance matrix
    x_s : array N x 1
        Test data input
    plot : boolean
        If true, then sample 5 normals from GP prior and save figures
    name : string
        Name under which figures are saved

    Returns mean and covariance
    """
    mu_prior = np.zeros(x_s.shape)
    cov_prior = kernel(x_s, x_s, params)    
    if plot:
        f_prior = np.random.multivariate_normal(mu_prior.ravel(), cov_prior, 5)
        plt.figure(figsize=(12,6))
        for sample in f_prior:
            plt.plot(x_s, sample, lw=1.5, ls='-')
        plt.title(f'{name} output - Prior using {kernel.__name__}')
        plt.xlabel('x')
        plt.ylabel('f')
        plt.tight_layout()        
        plt.savefig(f'figures/results/{name}_output_gpr_prior_{kernel.__name__}.png')
    return mu_prior, cov_prior


def posterior(x, x_s, f, kernel = kernel_rbf, params = params, noise = 1e-15, optimized = True, plot = True, name = "single"):
    """ Derive Posterior Distribution using Equation (1)
    
    Parameters:
    x : array  N x 1 
        Training Data inputs 
    x_s : array N x 1 
        Test Data inputs
    f : array N x 1
        Training Data outputs
    kernel : function (default: rbf kernel)
        Kernel function to compute covariance matrix
    params : dictionary
        Hyperparameters for kernel functions
    noise : float (default 1e-15)
        Noise that is added to data points (margins of error)
    optimized : boolean
        If true, optimize hyperparameters of kernel before deriving the posterior distribution
    plot : boolean
        If true, sample 5 normals from GP and plot with mean function and uncertainty
    name : string
        Name under which figures are saved
        
    Returns: mean vector and covariance matrix of posterior distribution
    """

    if optimized:
        res = minimize(nll(x, f, kernel=kernel, noise=noise), [1, 1, 1, 1, 1],
                    bounds=((1e-2, None), (1e-2, None), (1e-2, None), (1e-2, None), (1e-2, None)),
                    method='L-BFGS-B')
        # take optimized parameters for deriving posterior distribution
        var_opt, ell_opt, var_b_opt, off_opt, per_opt = res.x
        params = {'var': var_opt,
                'ell': ell_opt,        
                'var_b': var_b_opt,
                'off': off_opt,
                'per': per_opt}
        
    N, N_s = len(x), len(x_s)
    cov = kernel(x, x, params) + noise * np.eye(N)
    cov_s = kernel(x, x_s, params) 
    cov_ss = kernel(x_s, x_s, params) 
    
    # Cholesky decomposition for numerical stability of matrix inversion
    L = np.linalg.cholesky(cov + 1e-5 * np.eye(N))
    alpha_1 = np.linalg.solve(L, f)
    alpha = np.linalg.solve(L.T, alpha_1)
    mu_s = cov_s.T.dot(alpha)
    
    v = np.linalg.solve(L, cov_s)
    sigma_s = cov_ss - v.T.dot(v)


    if plot:
        x_s = x_s.ravel()
        mu = mu_s.ravel()
        cov = sigma_s
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
        plt.title(f'{name} output - Posterior using {kernel.__name__}')
        plt.xlabel('x')
        plt.ylabel('f')
        plt.tight_layout()        
        plt.savefig(f'figures/results/{name}_output_gpr_posterior_{kernel.__name__}.png')

    return mu_s, sigma_s



def nll(x, f, kernel=kernel_rbf, noise=1e-2):
    """ Compute negative log marginal likelihood, naive implementation of Equation (2) or
    stable implementation using Cholesky decomposition
    to optimize hyperparameters of kernels
    
    Parameters:
    x : array N x 1 
        Training data input
    f : array N x 1
        Training data output
    kernel : function (default: rbf kernel)
        Kernel function for which the parameters should be optimized
    noise : float (default: 1e-2)
        Ensure that the matrix calculations work
        
    Returns:
    nll_naive or nll_stable function to be minimized
    """
    f = f.ravel()
    N = len(x)
    def nll_naive(theta):
        params = {'var': theta[0],
                  'ell': theta[1],        
                  'var_b': theta[2],
                  'off': theta[3],
                  'per': theta[4]}
        cov_y = kernel(x, x, params) + noise**2 * np.eye(N)
        return 0.5 * f.dot(np.linalg.inv(cov_y).dot(f)) + \
               0.5 * np.log(np.linalg.det(cov_y)) + \
               0.5 * N * np.log(2*np.pi)

    def nll_stable(theta):
        params = {'var': theta[0],
                  'ell': theta[1],        
                  'var_b': theta[2],
                  'off': theta[3],
                  'per': theta[4]}
        cov_y = kernel(x, x, params)     # weniger noise wenn ich noise**2 nehm statt 1e-5
        L = np.linalg.cholesky(cov_y + 1e-8 * np.eye(N) )
        alpha_1 = np.linalg.solve(L, f)
        alpha = np.linalg.solve(L.T, alpha_1)
        
        return 0.5 * f.T.dot(alpha) + \
               np.sum(np.log(np.diagonal(L))) + \
               0.5 * N * np.log(2*np.pi)
    
    return nll_stable


def loocv(x, x_s, f, kernel=kernel_rbf, noise=1e-2):
    """ Leave-one-out cross-validation to compare model performances
    For each input data point, remove it from training data, then:
        Optimize hyperparameters of specified kernel 
        Derive posterior distribution with optimized parameters
        Obtain prediction for left-out data point
    
    Parameters:
    x : array N x 1
        Training data input
    x_s : array N x 1
        Test data input 
    f : array N x 1
        Training data output
    kernel : function (default: rbf kernel)
        Kernel function that specifies respective model
    noise : float (default: 1e-2)
        
    Returns:
    predictions : list
        List of predictions of left-out training data points    
    l2dist : float
        Distance of predictions to true training data output
    """
    predictions = []
    N = len(x) - 1
    for leave_out in range(N+1):
        X_new = np.delete(x, leave_out).reshape(-1,1)
        f_new = np.delete(f, leave_out).reshape(-1,1)
        if len(noise)>1:
            noise_new = np.delete(noise, leave_out).reshape(-1,1)

        mu_s, cov_s = posterior(X_new, x_s, f_new, kernel, params, noise_new, True, False, 'single')
        
        # prediction for left out data point
        idx_pred = np.absolute(x_s-x[leave_out]).argmin()
        predictions.append(mu_s.item(idx_pred))

    l2dist = np.linalg.norm(f.reshape(1,-1) - predictions)

    return predictions, l2dist





def analyse_single():
     # import single output data
    [x_true,y_true], [x_data,y_data] = create_single_output()
    x_s = x_true.reshape(-1,1)
    #x_s = create_test_data(N_s = 100, start = 0, end = 10).reshape(-1,1)
    x = x_data.reshape(-1,1)
    y = y_data.reshape(-1,1)
    
    # assume noise
    noise = 0.2

    # derive posteriors with optimized hyperparameters and plot them for all kernels
    for kernel in all_kernels:
        posterior(x, x_s, y, kernel, params, noise, True, True, 'single')

    # compute LOOCV for all kernels (with optimized parameters and assumed noise)
    for kernel in all_kernels:
        pred, l2 = loocv(x, x_s, y, kernel)
        print(f'\nPredictions for {kernel.__name__}:', pred, '\nL2 distance: ', l2)


def analyse_hist():
    # create normally distributed histogram data for colonies of size 2,3,4,5,7,10, N = sample size
    colony_sizes, outputs = create_hist_output(N=1000)
    x = colony_sizes.reshape(-1,1)
    x_s = create_test_data(N_s = 100, start = 0, end = 15).reshape(-1,1)

    # calculate mean number of histogram for each colony size 
    y = np.array([comp_mean(out, np.arange(len(out))) for out in outputs]).reshape(-1,1)
    # noise = margin of error for mean
    noise_y = np.array([comp_margin(t=1.962, sd=comp_sd(obs, np.arange(len(obs)), y[i]), N=100) for (obs,i) \
                   in zip(outputs, np.arange(len(outputs)))])


    # calculate variance of each histogram as second output
    y_v = np.array([sum(((np.arange(len(out)) * out) - y[i])**2) for (out,i) in zip(outputs,np.arange(len(outputs)))]).reshape(-1,1)
    y2 = np.array([comp_sd(obs, np.arange(len(obs)), y[i])**2 for (obs,i) \
                   in zip(outputs, np.arange(len(outputs)))])


    # MEAN: derive posteriors with optimized hyperparameters and plot them for all kernels
    for kernel in all_kernels:
        posterior(x, x_s, y, kernel, params, noise_y, True, True, 'hist_mean') 
    # compute LOOCV
    for kernel in all_kernels:
        pred, l2 = loocv(x, x_s, y, kernel, noise_y)
        print(f'\nPredictions for {kernel.__name__}:', pred, '\nL2 distance: ', l2)


    # VARIANCE: derive posteriors with optimized hyperparameters and plot them for all kernels
    #for kernel in all_kernels:
    #    posterior(x, x_s, y_v, kernel, params, noise, True, True, 'hist_var') 
    # VAR: compute LOOCV for all kernels (with optimized parameters and assumed noise)
    #print('\nVARIANCE')
    #for kernel in all_kernels:
    #    pred_v, l2_v = loocv(x, x_s, y_v, kernel)
    #    print(f'\nPredictions for {kernel.__name__}:', pred_v, '\nL2 distance: ', l2_v)

    # TODO: mean + var in 1 plot? 
    # TODO: noise around mean
    # TODO: kernel_add_p_l matrix not positive definite

    

def main():
   
    #analyse_single()
    analyse_hist()



if __name__ == "__main__":
    sys.exit(main())