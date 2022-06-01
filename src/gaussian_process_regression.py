"""
Gaussian Process Regression 

to predict the collective response (mean and variance of histogram)
for varying population sizes

"""

import sys
import warnings

from sympy import rotations
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import numpy as np
from read_data import *
from kernels import *
from scipy.optimize import minimize


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
    """ Compute mean value of histogram with frequencies freq, and observations obs
    Frequencies must be fractions, not integers!
    """
    return np.sum(freq * obs)

def comp_surv(freq, obs):
    """ Compute percentage of survivors of histogram with frequencies freq, and observations obs
    Frequencies must be fractions, not integers!
    """
    freq = list(reversed(freq))
    obs = obs / np.max(obs)
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
        plt.title(f'{name} output - prior using {kernel.__name__.split("_",1)[1]} kernel')
        plt.xlabel('x')
        plt.ylabel('f')
        plt.tight_layout()        
        plt.savefig(f'../figures/results/gpr/{name}_prior_{kernel.__name__.split("_",1)[1]}.png')
    return mu_prior, cov_prior


def posterior(x, x_s, f, kernel = kernel_rbf, params = params, noise = 1e-15, optimized = True, initial_guess = [1, 1, 1, 1, 1], plot = True, name = "single", prediction = None):
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
        try:
            res = minimize(nll(x, f, kernel=kernel, noise=noise), initial_guess,
                    bounds=((1e-7, None), (1e-7, None), (1e-7, None), (1e-7, None), (1, 10)),
                    method='L-BFGS-B')
            # take optimized parameters for deriving posterior distribution
            var_opt, ell_opt, var_b_opt, off_opt, per_opt = res.x
            params = {'var': var_opt,
                    'ell': ell_opt,   
                    'var_b': var_b_opt,
                    'off': off_opt,
                    'per': per_opt}
            print(params)
        except (np.linalg.linalg.LinAlgError, ValueError) as err:
            params = params
            
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

    cov = sigma_s
    uncertainty = 1.96 * np.sqrt(np.diag(cov))

    if plot:
        x_s, mu = x_s.ravel(), mu_s.ravel()

        samples = np.random.multivariate_normal(mu, cov, 5)
            
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.fill_between(x_s, mu + uncertainty, mu - uncertainty, alpha=0.2)
        ax.plot(x_s, mu, '--', color='darkblue', lw=3, label='Mean')
        # plot 5 samples:
        #for i, sample in enumerate(samples):
        #    plt.plot(x_s, sample, lw=1.5)
        if x is not None:
            ax.plot(x, f, 'o', ms=8, color='darkblue')
        #plt.legend()
        #plt.title(f'GPR posterior using {kernel.__name__.split("_",1)[1]} kernel')
        at = AnchoredText(f'{kernel.__name__}', prop=dict(size=22), frameon=True, loc='upper left')
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.add_artist(at)
        ax.set_xlabel('$X_*$', fontsize=22)
        ax.set_ylabel('$f_*$', fontsize=22, rotation = 0)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()        
        fig.savefig(f'../figures/results/gpr/{name}_posterior_{kernel.__name__.split("_",1)[1]}.png', bbox_inches = "tight")
        # dpi = 350, 

    if prediction is not None:
        idx = np.absolute(x_s - prediction).argmin()
        p = mu_s.item(idx)
        ci = uncertainty.item(idx)
        #print(f'Prediction for x_* = {prediction}({name}, {kernel.__name__}): f_* = {p} +- {ci}')


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
        L = np.linalg.cholesky(cov_y + noise**2 * np.eye(N) )
        alpha_1 = np.linalg.solve(L, f)
        alpha = np.linalg.solve(L.T, alpha_1)
        
        return 0.5 * f.T.dot(alpha) + \
               np.sum(np.log(np.diagonal(L))) + \
               0.5 * N * np.log(2*np.pi)
    
    return nll_stable


def loocv(x, x_s, f, kernel=kernel_rbf, noise=1e-2, initial_guess = [1, 1, 1, 1, 1]):
    """ Leave-one-out cross-validation to compare model performances
    For each input data point, remove it from training data, then:
        Optimize hyperparameters of specified kernel 
        Derive posterior distribution with optimized parameters
        Obtain prediction for left-out data point and calculate MSE
    Total score is average of all MSEs
    
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
        if hasattr(noise, "__len__"):
            noise_new = np.delete(noise, leave_out).reshape(-1,1)
        else:
            noise_new = noise

        mu_s, cov_s = posterior(X_new, x_s, f_new, kernel, params, noise_new, True, initial_guess, False, 'single')
        
        # prediction for left out data point
        #idx_pred = np.absolute(x_s-x[leave_out]).argmin()
        #predictions.append(mu_s.item(idx_pred))
        predictions.append(predict(x, x_s, mu_s, leave_out))

    #l2dist = np.linalg.norm(f.reshape(1,-1) - predictions)
    mse = np.square(np.subtract(f.reshape(1,-1), predictions)).mean()

    return predictions, mse


def predict(x, x_s, mu_s, pred):
    """ Return prediction for data point pred
    Equals value of mean function of posterior at this point
    """
    idx = np.absolute(x_s - x[pred]).argmin()
    return mu_s.item(idx)




""" Run complete program """

def analyse_hist(colony_sizes, outputs, teststart, testend, testpoints, samplesize, name, prediction = None):
    """ Analyze mean and variance of histograms individually, with colony_sizes describing the range of
    the histogram and outputs describing the probabilities
    Compute 95% confidence interval around mean and variance to use as noise around data points
    For all kernels:
        derive posterior distribution with optimized hyperparameters
        plot mean function, 5 samples, and uncertainty
        find best model with LOOCV
    Args:
        colony_sizes: numpy array of population sizes
        outputs: nested list of frequencies of each population size
        teststart: first value of test set
        testend: last value of test set
        testpoints: number of points in test set
        samplesize: number of observations for each population size
        name: name of experiment
        prediction: value of test set for which prediction and uncertainty should be computed
    """
    print(f"\n--- GPR ANALYSIS OF {name} ---")

    x = colony_sizes.reshape(-1,1)
    x_s = create_test_data(N_s = testpoints, start = teststart, end = testend).reshape(-1,1)

    # calculate mean number of histograms with margin of 95% confidence interval
    y_mean = np.array([comp_mean(out, np.arange(len(out))) for out in outputs]).reshape(-1,1)
    noise_mean = np.array([comp_margin(t=1.962, sd=comp_sd(obs, np.arange(len(obs)), y_mean[i]), N=samplesize) for (obs,i) \
                   in zip(outputs, np.arange(len(outputs)))])       # t for 95% with df=n-1=99

    # calculate variance of histograms as second output
    y_var = np.array([comp_sd(obs, np.arange(len(obs)), y_mean[i])**2 for (obs,i) in zip(outputs, np.arange(len(outputs)))])
    # TODO: noise is actually asymmetric, but here I just take a border to get larger margin
    # for 95% confidence and df=n-1=99: chi_(1-alpha/2)^2 = 73.361
    noise_var = np.array([(((99 * var) / 73.361) - var) for var in y_var])
    # calculate mean percentage of survivors for each colony size
    y_surv = np.array([comp_surv(out, np.arange(len(out))) for out in outputs]).reshape(-1,1)
    noise_surv = np.array([comp_margin(t=1.962, sd=comp_sd(obs, np.arange(len(obs)), y_surv[i]), N=samplesize) / 1000 for (obs,i) \
                   in zip(outputs, np.arange(len(outputs)))])       # t for 95% with df=n-1=99
    #noise_surv = 1e-3

    print("\n--- Predict mean & variance of histograms independently ---")
    min_mse_mean, best_pred_mean, best_kernel_mean = 1000, 0, 0
    min_mse_var, best_pred_var, best_kernel_var = 1000, 0, 0
    min_mse_surv, best_pred_surv, best_kernel_surv = 1000, 0, 0
    for kernel in all_kernels:
        print(kernel.__name__)
        print('mean:')
        posterior(x, x_s, y_mean, kernel, params, noise_mean, True, [1, 1, 1, 1, 1], True, f'{name}_mean', prediction) 
        pred_mean, mse_mean = loocv(x, x_s, y_mean, kernel, noise_mean)
        if mse_mean < min_mse_mean:
            min_mse_mean, best_pred_mean, best_kernel_mean = mse_mean, pred_mean, kernel
        
        #print('surv:')
        #posterior(x, x_s, y_surv, kernel, params, noise_surv, True, [1, 10, 1, 1, 30], True, f'{name}_surv', prediction) 
        #pred_surv, mse_surv = loocv(x, x_s, y_surv, kernel, noise_surv)
        #if mse_surv < min_mse_surv:
        #    min_mse_surv, best_pred_surv, best_kernel_surv = mse_surv, pred_surv, kernel
        
        print('var:')
        posterior(x, x_s, y_var, kernel, params, noise_var, True, [1, 1, 1, 1, 1], True, f'{name}_var', prediction) 
        pred_var, mse_var = loocv(x, x_s, y_var, kernel, noise_var)
        if mse_var < min_mse_var:
            min_mse_var, best_pred_var, best_kernel_var = mse_var, pred_var, kernel
            
    print(f'\nMEAN - Best results for {best_kernel_mean.__name__.split("_",1)[1]}:', best_pred_mean, '\nMSE: ', min_mse_mean)
    #print(f'\nSURV - Best results for {best_kernel_surv.__name__.split("_",1)[1]}:', best_pred_surv, '\nMSE: ', min_mse_surv)
    print(f'\nVAR - Best results for {best_kernel_var.__name__.split("_",1)[1]}:', best_pred_var, '\nMSE: ', min_mse_var)
    

def main():

    
    # MORGANE BEE DATA
    # Dataset 1 PO - 60 samples
    colony_sizes_po, outputs_po = read_hist_exp("bees_morgane/hist1_PO.txt")
    analyse_hist(colony_sizes = colony_sizes_po, outputs = outputs_po, 
                 teststart = 0, testend = 13, testpoints = 100, samplesize = 60, name = 'Experiment_A', prediction = 7)

    colony_sizes_po, outputs_po = read_hist_exp("bees_morgane/dataset3_data.txt")
    analyse_hist(colony_sizes = colony_sizes_po, outputs = outputs_po, 
                 teststart = 0, testend = 13, testpoints = 100, samplesize = 40, name = 'Experiment_B', prediction = 7)

    # Dataset 1 IAA - 60 samples
    colony_sizes_iaa, outputs_iaa = read_hist_exp("bees_morgane/hist1_IAA.txt")
    analyse_hist(colony_sizes_iaa, outputs_iaa, 0, 13, 100, 60, 'beesMorgane1IAA', prediction = 7)

    
    # Dataset 2 - samples: 68,68,60,56,52,48
    colony_sizes_2, outputs_2 = read_hist_exp("bees_morgane/hist2.txt")
    analyse_hist(colony_sizes_2, outputs_2, 0, 17, 100, 58, 'Experiment_C', prediction = 7)

    

    #colony_sizes, outputs = read_stochnet_hist(100)
    #analyse_hist(colony_sizes, outputs, 10, 150, 500, 100, 'stochnet')


if __name__ == "__main__":
    sys.exit(main())