"""
Gaussian Process Classification

Estimate satisfaction function for property

"""

import os
import numpy as np
import numpy.matlib
import math
from scipy.stats import norm
from scipy.special import erf 
# scipy gives upper triangular matrix, numpy lower
from scipy.linalg import cholesky
import matplotlib.pyplot as plt

import sys
import warnings
from numpy.core.fromnumeric import reshape
warnings.filterwarnings("ignore")
from create_data import *
from kernels import *
from scipy.optimize import minimize

import pickle


""" Global Parameters """

# define all kernels
all_kernels = [kernel_linear, kernel_rbf, kernel_periodic, kernel_add_p_r, kernel_add_r_l,
              kernel_mult_p_l, kernel_mult_p_r, kernel_mult_r_l]

# correction for kernel computation to ensure numerical stability
correction = 1e-4



""" Helper Functions """

def plot_training(x, y, scale, name):
    """ Plot observations as scatter plot showing the satisfaction probabilities
    """
    plt.figure(figsize=(12,6))
    plt.scatter(x, y, marker='o', c='blue')
    plt.title(f'Training dataset with {scale} trajectories per input point')
    plt.xlabel('$\theta$')
    plt.ylabel('Satisfaction probability')
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.savefig(f'../figures/results/gpc/{name}_training.png')

def get_bees_data(thresh, scale):
    """ Read txt files containing number of stinging bees after the experiments
    and collect how often property is satisfied for trajectory
    """
    collect_data = {}

    for dirpath, dirs, files in os.walk("../data/stochnet"):
        for file in files:
            nbees = int((file.split("_")[1]).split(".")[0])
            with open(os.path.join(dirpath, file), 'r') as f:
                data = f.read()
                x_deadbees = (data.split("]")[0])[2:].replace(".","")
                y_frequencies = (data.split("]")[1])[2:]
                # compute number of living bees 
                bees = np.array([nbees-int(i) for i in x_deadbees.split()])
                freq = np.array([int(i) for i in y_frequencies.split()])
                
                threshold = np.ceil(thresh * nbees)            
                satisfactions = np.sum(freq[bees >= threshold])
                
                collect_data[nbees] = satisfactions
                
    # population size n together with number of trajectories satisfying property, sort by n
    paramValueSet = []
    paramValueOutputs = []
    for key in sorted(collect_data):
        paramValueSet.append(key)
        paramValueOutputs.append(collect_data[key])

    return np.array(paramValueSet).reshape(-1,1), (np.array(paramValueOutputs).reshape(-1,1))/scale

def standardNormalCDF(x):
    """ Compute probit values (probabilities from real values)
    """
    return 1/2 + 1/2 * erf(x * (1/np.sqrt(2)))



""" Helper Functions for EP """

def marginal_moments(Term, gauss_LC, gauss_LC_t):
    """
    Computes marginal moments
    
    Args:
        Term: v_tilde, tau_tilde (datapoints, 2)
        gauss_LC:
        gauss_LC_t: 
        
    Returns:
        logZappx: 
        gauss_m: mu (datapoints,1)
        gauss_diagV: diagonal of sigma^2 (datapoints,1)
    """
    
    datapoints = len(Term)

    # A = LC' * (tau_tilde * gauss_LC) + I 
    tmp = np.multiply(Term[:,1], gauss_LC)
    A = np.matmul(gauss_LC_t, tmp) + 1 * np.eye(datapoints)
    gauss_L = cholesky(A).T  

    # W = L\LC' -> Solve L*W = LC'
    gauss_W = np.linalg.solve(gauss_L, gauss_LC_t)
    gauss_diagV = np.diagonal(np.matmul(gauss_W.T, gauss_W)).reshape(-1,1)
    
    # m = W'*(W * v_tilde)
    tmp = np.matmul(gauss_W, Term[:,0])
    gauss_m = np.matmul(gauss_W.T, tmp).reshape(-1,1)

    # logdet = -2*sum(log(diag(L))) + 2*sum(log(diag(LC)))
    logdet = -2*np.sum(np.log(np.diagonal(gauss_L))) # + 2*np.sum(np.log(np.diag(gauss_LC))) (das ist schon logdet_LC)
    logdet_LC = 2*np.sum(np.log(np.diagonal(gauss_LC).reshape(-1,1)))
    logdet += logdet_LC

    # logZappx = 1/2(m' * v_tilde + logdet)
    logZappx = 0.5 * (np.dot(gauss_m.T, Term[:,0]) + logdet)

    return logZappx, gauss_m, gauss_diagV


def gausshermite(nodes):
    """
    Gauss-Hermite 
    https://indico.frib.msu.edu/event/15/attachments/40/157/Gaussian_Quadrature_Numerical_Recipes.pdf
    
    Approximate integral of a formula by the sum of its functional values at some points
    
    Args:
        nodes: number of Gauss-Hermite nodes (96,1)
        
    Returns:
        x0: abscissas (96,1)
        w0: weights (96,1)
    """

    x0 = np.zeros((nodes, 1))
    w0 = np.zeros((nodes, 1))
    m = int((nodes+1)/2)
    z,pp,p1,p2,p3 = 0,0,0,0,0
    
    for i in range(m):
        if i==0:
            z = np.sqrt(2*nodes+1) - 1.85575 * ((2*nodes+1)**(-0.166667))
        elif i==1:
            z = z - 1.14 * (nodes**0.426) / z
        elif i==2:
            z = 1.86 * z - 0.86 * x0[0]
        elif i==3:
            z = 1.91 * z - 0.91 * x0[1]
        else:
            z = 2.0 * z - x0[i - 2]

        for its in range(10):
            p1 = 1/np.sqrt(np.sqrt(np.pi))
            p2 = 0
            for j in range(1,nodes+1):
                p3=p2
                p2=p1
                a = z*np.sqrt(2/j)*p2
                b = np.sqrt((j-1)/j)*p3
                p1=a-b
            pp=np.sqrt(2*nodes)*p2
            z1=z
            z=z1-p1/pp
            if np.abs(z-z1)<2.2204e-16:
                break

        x0[i] = z
        x0[nodes-1-i] = -z
        w0[i] = 2/(pp*pp)
        w0[nodes-1-i] = w0[i]

    w0 = np.divide(w0, np.sqrt(np.pi))
    x0 = np.multiply(x0, np.sqrt(2))
    x0 = np.sort(x0, axis=None).reshape(-1,1)
    
    return x0, w0


def GaussHermiteNQ(FuncPar_p, FuncPar_q, cav_m, cav_v, xGH, logwGH):
    """
    Gauss-Hermite numerical quadrature for moment computation
    
    Args:
        FuncPar_p: number of runs satisfying property for each parameter value (input) (datapoints,1)
        FuncPar_q: number of runs not satisfying property (datapoints,1)
        cav_m: cavity mean mu_-i
        cav_v: cavity variance sigma^2_-i
        xGH: abscissas (Gauss-Hermite)
        logwGH: weights (Gauss-Hermite)
        
    Returns:
        logZ: largest element of expectation of normally distributed variable?
        Cumul: mu_hat, sigma^2_hat (datapoints,2)
    """
    
    Nnodes = len(xGH)
    datapoints = len(FuncPar_p)
    
    # sigma_-i
    stdv = np.sqrt(cav_v).reshape(-1,1)
    Y = np.matmul(stdv, xGH.reshape(1,-1)) + numpy.matlib.repmat(cav_m, 1, Nnodes)
    G = np.array(logprobitpow(Y, FuncPar_p, FuncPar_q) + numpy.matlib.repmat(logwGH.T, datapoints, 1))
        
    # maximum of each row (input value) over all 96 nodes
    maxG = G.max(axis=1).reshape(-1,1)
    # subtract maximum value
    G = G - np.matlib.repmat(maxG, 1, 96)
    # exponential value
    expG = np.exp(G)
    # denominator (row sum)
    denominator = expG.sum(axis=1).reshape(-1,1)
    logdenominator = np.log(denominator)
    logZ = maxG + logdenominator
    
    Cumul = np.zeros((len(FuncPar_p), 2))

    # deltam = stdv * (expG * xGH) / denominator
    deltam = np.divide(np.multiply(stdv, np.matmul(expG, xGH)), denominator)

    # mu_hat = mu_-i + deltam    
    Cumul[:,0] = (cav_m + deltam).reshape(-1)
    
    xGH2 = xGH**2
    deltam2 = deltam**2

    # sigma^2_hat
    Cumul[:,1] = (np.divide(np.multiply(cav_v, np.matmul(expG, xGH2)), denominator) - deltam2).reshape(-1)
        
    return logZ, Cumul


def cavities(gauss_diagV, gauss_m, Term):
    """
    Compute cavity distribution by removing the effect of a single factor from q
    
    Args:
        gauss_diagV: sigma^2
        gauss_m: mu
        Term: v_tilde, tau_tilde (datapoints,2)
        
    Returns:
        cav_m: cavity mean mu_-i
        cav_diagV: cavity variance sigma^2_-i
    """
    
    # s = 1 / (1 + -tau_tilde * diagV)
    s = np.divide(1, (1 + np.multiply(-Term[:,1].reshape(-1,1), gauss_diagV)))

    # cav_diagV = s * diagV
    cav_diagV = np.multiply(s, gauss_diagV)
    
    # cav_m = s * (m + (-v_tilde * diagV))
    cav_m = np.multiply(s, (gauss_m + np.multiply(-Term[:,0].reshape(-1,1), gauss_diagV)))
    
    return cav_m, cav_diagV
        

def ep_update(cav_diagV, cav_m, Term, eps_damp, gauss_LikPar_p,
              gauss_LikPar_q, gauss_xGauss, gauss_logwGauss):
    """
    Update site parameters
    
    Args:
        cav_diagV: cavity variance sigma^2_-i
        cav_m: cavity mean mu_-i
        Term: v_tilde, tau_tilde (datapoints,2)
        eps_damp: 0.5
        gauss_LikPar_p: number of runs satisfying property for each parameter (datapoints,1)
        gauss_LikPar_q: number of runs not satisfying property (datapoints,1)
        gauss_xGauss: abscissas of Gauss-Hermite
        gauss_logwGauss: weights of Gauss-Hermite
        
    Returns:
        TermNew: updated v_tilde, tau_tilde (datapoints,2)
        logZterms:
        logZ:
    """
    
    datapoints = len(Term)

    # Cumul = [mu_hat, sigma^2_hat]    
    # Evaluate new approximation q_hat(x) by setting the sufficient statistics equal to that of probit*cavity
    logZ, Cumul = GaussHermiteNQ(gauss_LikPar_p, gauss_LikPar_q, cav_m, cav_diagV, gauss_xGauss, gauss_logwGauss)
    
    m2 = cav_m**2
    logV = np.log(cav_diagV)
    
    # cumul1 = mu_hat^2
    cumul1 = (Cumul[:,0]**2).reshape(-1,1)
    # cumul2 = log(sigma^2_hat)
    cumul2 = (np.log(Cumul[:,1])).reshape(-1,1)    
    
    # logZ + 1/2 * ((mu_-i^2 / sigma_-i^2) + log(sigma_-i^2) - (mu_hat^2 / sigma_hat^2) + log(sigma_hat^2))
    logZterms = logZ + np.multiply(np.divide(m2, cav_diagV) + logV - 
                                   (np.divide(cumul1, Cumul[:,1].reshape(-1,1)) + cumul2), 1/2)
    
    ### hier sind die ganzen if-bedingungen, um negative werte und anderes zu pruefen
    # kommt aber nie in die bedingungen rein
    c1 = np.zeros((datapoints,1))
    c2 = np.zeros((datapoints,1))
    for i in np.arange(datapoints):
        if (1/cav_diagV[i] == 1/Cumul[i,1]):
            c2[i] = 1e-4
        else:
            c2[i] = (1 / Cumul[i,1]) - (1 / cav_diagV[i])
    
    for k in np.arange(datapoints):
        if((1/c2[k] == np.infty) and (1/cav_diagV[k] == 0)):
            c1[k] = cav_m[k] * cav_diagV[k]
        else:
            c1[k] = Cumul[k,0] / Cumul[k,1] - cav_m[k] / cav_diagV[k]
    

    for j in np.arange(datapoints):
        if (1/c2[j] + cav_diagV[j]) < 0:
            c1[j] = Term[j,0]
            c2[j] = Term[j,1]
        else:
            continue
            
    TermNew = np.concatenate((c1, c2), axis=1)
    TermNew = np.multiply(Term, (1 - eps_damp)) + np.multiply(TermNew, eps_damp)
    
    return TermNew, logZterms, logZ


def logprobitpow(X, p, q):
    """
    Compute ncdflogbc for matrices -> log of standard normal cdf by 10th order Taylor expansion in the negative domain
    log likelihood evaluations for various probit power likelihoods
    
    Args:
        X: matrix (datapoints,96)
        p: number of runs satisfying property for each parameter, repeated (datapoints,96)
        q: number of runs not satisfying property, repeated (datapoints,96)
        
    Returns:
        Za+Zb:
    """
    
    threshold = -np.sqrt(2)*5
    Za = []
    y = []
    j = 0
    for x in X:
        y.append([])
        for i in x:
            if i >= 0:
                y[j].append(np.log(1 + erf(i/np.sqrt(2))) - np.log(2))
            elif ((threshold < i) and (i < 0)):
                y[j].append(np.log(1 - erf((-i)/np.sqrt(2))) - np.log(2))
            elif i <= threshold:
                y[j].append(-1/2 * np.log(np.pi) - np.log(2) - 1/2 * (-i) * (-i) - \
                np.log((-i)) + np.log(1 - 1/(-i) + 3/((-i)**4) - 15/((-i)**6) + 105/((-i)**8) - 945/((-i)**10)))
        j+=1
    Za = np.multiply(y, numpy.matlib.repmat(p.reshape(-1,1), 1, 96))

    Zb = []
    y = []
    j = 0
    for x in (-X):
        y.append([])
        for i in x:
            if i >= 0:
                y[j].append(np.log(1 + erf(i/np.sqrt(2))) - np.log(2))
            elif ((threshold < i) and (i < 0)):
                y[j].append(np.log(1 - erf((-i)/np.sqrt(2))) - np.log(2))
            elif i <= threshold:
                y[j].append(-1/2 * np.log(np.pi) - np.log(2) - 1/2 * (-i) * (-i) - \
                np.log((-i)) + np.log(1 - 1/(-i) + 3/((-i)**4) - 15/((-i)**6) + 105/((-i)**8) - 945/((-i)**10)))
        j+=1

    Zb = np.multiply(y, numpy.matlib.repmat(q.reshape(-1,1), 1, 96))
    return Za + Zb
    




""" Analyses """

def expectation_propagation(paramValueSet, paramValueOutputs, scale, params):
    """
    Expectation Propagation Algorithm
    """

    datapoints = len(paramValueSet)

    # Prior
    gauss_C = kernel_rbf(paramValueSet, paramValueSet, params) + correction * np.eye(datapoints) # covariance training set

    gauss_LC_t = cholesky(gauss_C)  # cholesky decomposition, returns U from A=U'*U (U=L')
    gauss_LC = gauss_LC_t.T  # transpose LC' 
    gauss_LC_diag = np.diagonal(gauss_LC).reshape(-1,1)

    logZprior = 0.5*(2*np.sum(np.log(gauss_LC_diag)))

    logZterms = np.zeros(datapoints).reshape(-1,1)
    logZloo = np.zeros(datapoints).reshape(-1,1)
    Term = np.zeros((datapoints, 2))  # Term = v_tilde, tau_tilde

    # compute marginal moments mu and sigma^2
    _, gauss_m, gauss_diagV = marginal_moments(Term, gauss_LC, gauss_LC_t)

    # related to likelihood approximation
    # true observation values (number of trajectories satisfying property)
    gauss_LikPar_p = paramValueOutputs * scale
    gauss_LikPar_q = scale - gauss_LikPar_p 

    # gauss hermite: quadrature to approximate values of integral, returns abscissas (x) and weights (w) of
    # n-point Gauss-Hermite quadrature formula
    nodes = 96
    gauss_xGauss, gauss_wGauss = gausshermite(nodes)
    gauss_logwGauss = np.log(gauss_wGauss)

    # configurations for loop initialization
    MaxIter=1000
    tol=1e-6
    logZold=0
    logZ = 2*tol
    steps=0
    logZappx=0
    eps_damp=0.5

    while (np.abs(logZ-logZold)>tol) and (steps<MaxIter):
        steps += 1
        logZold = logZ
        
        # find cavity distribution parameters mu_-i and sigma^2_-i
        cav_m, cav_diagV = cavities(gauss_diagV, gauss_m, Term)
        
        # update marginal moments mu_hat and sigma^2_hat, and site parameters v_tilde and tau_tilde
        Term, logZterms, logZloo = ep_update(cav_diagV, cav_m, Term, eps_damp, gauss_LikPar_p,
                        gauss_LikPar_q, gauss_xGauss, gauss_logwGauss)
        
        # recompute mu and sigma^2 from the updated parameters
        logZappx, gauss_m, gauss_diagV = marginal_moments(Term, gauss_LC, gauss_LC_t)
        
        logZ = logZterms.sum() + logZappx
        
        print("Iteration ", steps)
        
    print("\nFinish")
                
    logZ = logZ - logZprior

    v_tilde = Term[:,0].reshape(-1,1)
    tau_tilde = Term[:,1].reshape(-1,1)
    diagSigma_tilde = 1/tau_tilde
    mu_tilde = np.multiply(v_tilde, diagSigma_tilde)
    Sigma_tilde = np.zeros((datapoints, datapoints))
    np.fill_diagonal(Sigma_tilde, diagSigma_tilde)

    # inverse of K + Sigma_tilde
    invC = np.linalg.solve((gauss_C + Sigma_tilde), np.eye(len(mu_tilde)))

    return mu_tilde, invC


def perform_ep(x, f, scale, params):
    """
    Take care that matrix is positive definite, then perform EP
    If not, change kernel's hyperparameters and try again
    """
    var = params['var']
    ell = params['ell']
    mu_tilde = None
    invC = None
    try:
        mu_tilde, invC = expectation_propagation(x, f, scale, params)
    except (np.linalg.linalg.LinAlgError, ValueError) as err:
        if ell > 20:
            print("NOT CONVERGING")
        else:
            if any(s in str(err) for s in ['not positive definite', 'infs or NaNs']):
                params = {'var': 1/3*var,
                        'ell': 1+ell,        
                        'var_b': 1,
                        'off': 1}
                print('nochmal')
                perform_ep(x, f, scale, params)
            else:
                raise
    return mu_tilde, invC



def get_posterior(x, x_s, f, mu_tilde, invC, kernel, params, name):
    """ 
    Compute posterior distribution, derive predictive mean and variance
    Plot mean function together with 95% confidence interval

    Args:
        x: training set (inputs)
        x_s: test set (inputs)
        f: training set (outputs)
        mu_tilde: 
        invC: inverse of K + Sigma_tilde
        kernel: kernel function
        params: hyperparameters of kernel

    """

    # calculate variances of testset and covariances of test & training set (apply kernel)
    kss = kernel(x_s, x_s, params) #+ correction * np.eye(testpoints) 
    ks = kernel(x_s, x, params) #+ correction * np.eye(datapoints)

    # predictive mean 
    fs = np.matmul(np.matmul(ks, invC), mu_tilde)

    # predictive variance -> here only diagonal
    vfs = (np.diagonal(kss) - (np.diagonal(np.matmul(np.matmul(ks, invC), ks.T)))).reshape(-1,1)

    cached_denominator = (1 / np.sqrt(1 + vfs)).reshape(-1,1)

    # get probabilities with probit function
    probabilities = standardNormalCDF(fs * cached_denominator)

    # compute confidence bounds
    lowerbound = standardNormalCDF((fs - 1.96 * np.sqrt(vfs).reshape(-1,1)) *
                                cached_denominator)
    upperbound = standardNormalCDF((fs + 1.96 * np.sqrt(vfs).reshape(-1,1)) *
                                cached_denominator)

    # plot data
    plt.figure(figsize=(12,6))
    plt.plot(x_s, probabilities, lw=1.5, ls='-')
    plt.fill_between(x_s.ravel(), lowerbound.ravel(), upperbound.ravel(), alpha=0.2)
    plt.scatter(x, f, marker='o', c='blue')
    plt.yticks(np.arange(0, 0.6, step=0.1))
    plt.title('Predictive probability together with 95% confidence interval')
    plt.xlabel('Population size $N$')
    plt.ylabel('Satisfaction probability')
    plt.savefig(f'../figures/results/gpc/{name}_posterior.png')

    return probabilities


def coeff_variation(probs, name):
    c = []
    for t in probs:
        c.append(np.std(probs[t])/np.mean(probs[t]))
    plt.figure(figsize=(12,6))
    plt.plot(probs.keys(), c, lw=1.5, ls='-')
    plt.scatter(probs.keys(), c, marker='o', c='blue')
    plt.title('Coefficient of variation for different thresholds as: sd/mean')
    plt.xlabel('Threshold $t$')
    plt.ylabel('Coefficient of variation')
    plt.savefig(f'../figures/results/gpc/{name}_variation.png')




""" Run complete program """

def analyse_ex_paper():
    """ 
    Run GPC for viral example of Smoothed Model Checking Paper (Bortolussi)

    """

    # number of trajectories per input point
    scale = 5 
    # uncertain input parameter \theta that is varied
    paramValueSet = np.linspace(0.5, 5, 20).reshape(-1,1) 
    # statistical outputs of satisfaction 
    paramValueOutput = np.array([1,1,0.8,1,1,1,1,0.6,0.6,0.6,0.8,0,0.8,0.2,0.6,0,0.4,0.4,0,0.2]).reshape(-1,1) 

    # plot training data
    plot_training(paramValueSet, paramValueOutput, scale, 'paper_ex')
    print("Actual data (number of runs satisfying property): ", (paramValueOutput * scale).reshape(1,-1))

    # define default hyperparameters for kernels
    params = {'var': 1/5,
            'ell': 1,        
            'var_b': 1,
            'off': 1}

    # perform EP
    mu_tilde, invC = perform_ep(paramValueSet, paramValueOutput, scale, params)
    
    # derive predictive probabilities and confidence intervals, save plot
    testset = np.linspace(0, 5, 50).reshape(-1,1)
    get_posterior(paramValueSet, testset, paramValueOutput, mu_tilde, invC, kernel_rbf, params, 'paper_ex')




def analyse_bees(t, v, l):
    """ 
    Analyze satisfaction probability for different population sizes to find out if function is robust
    Input: histogram data for different population sizes n
    Then compute GPC of satisfaction probabilty

    """

    scale = 1000
    thresh = t
    paramValueSet, paramValueOutput = get_bees_data(thresh, scale)

    #scale = 1000
    plot_training(paramValueSet, paramValueOutput, scale, f'bees_stochnet_{thresh}')

    # define default hyperparameters for kernels
    # variance = max-min / 2 for output values (if this is 0, set to 1)
    # lengthscale = max - min / 10 for input values
    params = {'var': v,
            'ell': l,        
            'var_b': 1,
            'off': 1}
   
    mu_tilde, invC = perform_ep(paramValueSet, paramValueOutput, scale, params)

    # derive predictive probabilities and confidence intervals, save plot
    testset = np.linspace(15, 150, 500).reshape(-1,1)
    p = get_posterior(paramValueSet, testset, paramValueOutput, mu_tilde, invC, kernel_rbf, params, f'bees_stochnet_{thresh}')

    return p


    

def main():
    #analyse_ex_paper()

    probs = {}
    threshs = np.arange(0.09, 0.25, 0.02)

    #threshs = np.arange(0.20, 0.25, 0.02)

    for t in threshs:
        print("t = ", t)
        probs[t] = analyse_bees(t, 0.01, 10)

    #analyse_bees(0.2, 0.05, 10)
    #analyse_bees(0.15, 0.05, 10)
    #analyse_bees(0.17, 0.05, 10)
    #analyse_bees(0.23, 0.01, 10)
    #analyse_bees(0.13, 0.01, 10)
    #analyse_bees(0.11, 0.01, 10)
    #analyse_bees(0.14, 0.01, 10)
    #analyse_bees(0.16, 0.01, 10)
    #analyse_bees(0.18, 0.01, 10)
    #analyse_bees(0.19, 0.01, 10)
    #infile = open('../../data_10.p','rb')
    #new_dict = pickle.load(infile)
    #infile.close()

    coeff_variation(probs, f'bees_stochnet')


    #print(new_dict)


if __name__ == "__main__":
    sys.exit(main())