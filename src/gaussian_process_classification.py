"""
Gaussian Process Classification

Estimate satisfaction function for property
1-dimensional or 1-dimensional input

"""

import sys
import warnings
import numpy as np
import numpy.matlib
from scipy.special import erf 
from scipy.linalg import cholesky  # scipy gives upper triangular matrix, numpy lower
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.offsetbox import AnchoredText

from read_data import *
from create_data import *
from kernels import *

warnings.filterwarnings("ignore")




""" Global Parameters """

# correction for kernel computation to ensure numerical stability
correction = 1e-4



""" Helper Functions """

def plot_training(x, y, name): 
    """ Plot observations as scatter plot showing the satisfaction probabilities

    Args:
        x: Training inputs of 1 or 2 dimensions (datapoints, dim)
        y: Training outputs, satisfactions (datapoints, 1)
        name: string that specifies the dataset to save the plot
    """
    # 1 dimension: 2D scatter plot
    if len(x[0,:]) == 1:
        plt.figure(figsize=(12,6))
        plt.scatter(x, y, marker='o', c='blue')
        #plt.title(f'Training dataset with {scale} trajectories per input point')
        plt.xlabel('Population size $N$')
        plt.ylabel('Satisfaction probability')
        plt.yticks(np.arange(0, 1.1, step=0.1))
        plt.savefig(f'../figures/results/gpc/{name}_training.png')

    # 2 dimensions: 3D scatter plot
    elif len(x[0,:]) == 2:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10,10))
        sca = ax.scatter(x[:,0], x[:,1], y, c=y, s=40, cmap=cm.coolwarm)
        ax.set_zticks(np.arange(0, 1.1, step=0.1))
        ax.set_xlabel('Parameter 1')
        ax.set_ylabel('Parameter 2')
        ax.set_zlabel('Satisfaction Probability')
        fig.colorbar(sca, shrink=0.5)
        #ax.view_init(-150,60)
        plt.savefig(f'../figures/results/gpc/{name}_training3d.png')
    
    else:
        raise ValueError('Classification supports only datasets of 1 or 2 dimensions.')

def probitCDF(x):
    """ Compute inverse probit (= CDF) to map real values to probabilities 
    
    Args: 
        x: real value
    Returns:
        CDF of x, corresponds to probability
    """
    return 1/2 + 1/2 * erf(x * (1/np.sqrt(2)))



""" Helper Functions for EP """

def marginal_moments(Term, gauss_LC, gauss_LC_t):
    """
    Computes marginal moments (Rasmussen 3.53)
    
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
        nodes: number of Gauss-Hermite nodes 
        
    Returns:
        x0: abscissas (nodes,1)
        w0: weights (nodes,1)
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
    
    # check negative variances, etc.
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

def expectation_propagation(paramValueSet, paramValueOutputs, scale, kernel, params):
    """
    Expectation Propagation Algorithm

    Args:
        paramValueSet: training input values (datapoints, dim)
        paramValueOutputs: training output values (datapoints, 1)
        scale: sample size (number of observations for each data point)
        kernel: RBF or RBF-ARD
        params: hyperparameters of kernel

    Returns:
        mu_tilde: mean
        invC: inverse of K + Sigma_tilde
    """

    datapoints = len(paramValueSet)

    # Prior
    gauss_C = kernel(paramValueSet, paramValueSet, params) + correction * np.eye(datapoints) # covariance training set

    gauss_LC_t = cholesky(gauss_C)  # cholesky decomposition, returns U from A=U'*U (U=L')
    gauss_LC = gauss_LC_t.T  # transpose LC' 
    gauss_LC_diag = np.diagonal(gauss_LC).reshape(-1,1)

    logZprior = 0.5*(2*np.sum(np.log(gauss_LC_diag)))

    logZterms = np.zeros(datapoints).reshape(-1,1)
    logZloo = np.zeros(datapoints).reshape(-1,1)
    Term = np.zeros((datapoints, 2))  # Term = v_tilde, tau_tilde

    # compute marginal moments mu and sigma^2
    _, gauss_m, gauss_diagV = marginal_moments(Term, gauss_LC, gauss_LC_t)

    # true observation values (number of trajectories satisfying property) for likelihood
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
        
    print("\nFinish\n")
                
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


def perform_ep(x, f, scale, kernel, params):
    """
    Take care that matrix is positive definite, then perform EP
    If not, change kernel's hyperparameters and try again

    Args:
        x: training inputs
        f: training outputs
        scale: sample size
        kernel: RBF or RBF-ARD
        params: hyperparameters of kernel
    """
    var = params['var']
    ell = params['ell']
    ell_dim = params['ell_dim']
    mu_tilde, invC = None, None
    try:
        mu_tilde, invC = expectation_propagation(x, f, scale, kernel, params)
    # catch if matrix computations give numerical errors, then try again
    except (np.linalg.linalg.LinAlgError, ValueError) as err:
        if ell > 40:
            print("NOT CONVERGING")
        else:
            if any(s in str(err) for s in ['not positive definite', 'infs or NaNs']):
                params = {'var': 1/3*var,
                          'ell': 1+ell,        
                          'ell_dim': [ell_dim[0]+1, ell_dim[1]+0.005],
                          'var_b': 1,
                          'off': 1}
                print('RUN AGAIN')
                perform_ep(x, f, scale, kernel, params)
            else:
                raise
    return mu_tilde, invC



def get_posterior(x, x_s, f, mu_tilde, invC, kernel, params, name, t = None, pmc_x = None, pmc_f = None):
    """ 
    Compute posterior distribution, derive predictive mean and variance
    Plot mean function together with 95% confidence interval

    Args:
        x: training set (inputs)
        x_s: test set (inputs)
        f: training set (outputs)
        mu_tilde: mean
        invC: inverse of K + Sigma_tilde
        kernel: kernel function
        params: hyperparameters of kernel
        name: name of experiment for saving plots
        t: threshold to check fitness function, if available
        pmc_x: X values of PMC, if available
        pmc_f: outputs of PMC, if available

    Returns:
        p: predictive probabilities
    """

    # calculate variances of testset and covariances of test & training set (apply kernel)
    kss = kernel(x_s, x_s, params) + correction * np.eye(len(x_s)) 
    ks = kernel(x_s, x, params) #+ correction * np.eye(datapoints)

    # predictive mean 
    fs = np.matmul(np.matmul(ks, invC), mu_tilde)

    # predictive variance -> here only diagonal
    vfs = (np.diagonal(kss) - (np.diagonal(np.matmul(np.matmul(ks, invC), ks.T)))).reshape(-1,1)

    cached_denominator = (1 / np.sqrt(1 + vfs)).reshape(-1,1)

    # get probabilities with probit function for 1 dimension
    if len(x_s[0,:]) == 1:
        probabilities = probitCDF(fs * cached_denominator)

        # compute confidence bounds
        lowerbound = probitCDF((fs - 1.96 * np.sqrt(vfs).reshape(-1,1)) *
                                    cached_denominator)
        upperbound = probitCDF((fs + 1.96 * np.sqrt(vfs).reshape(-1,1)) *
                                    cached_denominator)

        # plot data
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(x_s, probabilities, '--', color='darkblue', lw=3, label='Mean')
        ax.fill_between(x_s.ravel(), lowerbound.ravel(), upperbound.ravel(), alpha=0.2)
        ax.plot(x, f, 'o', ms=8, color='darkblue')
        plt.yticks(np.arange(0, 1.1, step=0.1))
        #plt.title('Predictive probability together with 95% confidence interval')
        if pmc_x is not None:
            ax.plot(pmc_x, pmc_f, '+', ms=4, c='orange', label='PMC')    
        if t is not None:
            at = AnchoredText(f't = {round(t, 2)}', prop=dict(size=22), frameon=True, loc='lower left')
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax.add_artist(at)
        ax.set_xlabel('$X_*$', fontsize=18)
        ax.set_ylabel('$f_*$', fontsize=18, rotation = 0)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()        
        fig.savefig(f'../figures/results/gpc/{name}_posterior.png', dpi = 300)
    
    # get probabilities with probit function for 2 dimensions
    elif len(x_s[0,:]) == 2:
        points = int(np.ceil(np.sqrt(len(x_s))))
        probabilities = probitCDF(fs * cached_denominator).reshape(points,-1).T
        # compute confidence bounds
        lowerbound = probitCDF((fs - 1.96 * np.sqrt(vfs).reshape(-1,1)) *
                                    cached_denominator).reshape(points,-1).T
        upperbound = probitCDF((fs + 1.96 * np.sqrt(vfs).reshape(-1,1)) *
                                    cached_denominator).reshape(points,-1).T

        # Contour Plot 
        p1 = np.unique(x_s[:,0])
        p2 = np.unique(x_s[:,1])
        plt.figure(figsize=(8, 8))
        plt.contourf(p1, p2, probabilities, levels=np.linspace(0,1,11))
        plt.xlabel('$x1_*$')
        plt.ylabel('$x2_*$', rotation=0)
        cbar = plt.colorbar()
        cbar.set_label('Satisfaction Probability')
        plt.autoscale()
        plt.savefig(f'../figures/results/gpc/{name}_posteriorCont.png')

        # Surface Plot with mean posterior and training points
        l1, l2 = np.meshgrid(p1, p2)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10,10))
        surf = ax.plot_surface(l1, l2, probabilities, cmap=cm.coolwarm, vmin=0, vmax=1)
        ax.scatter(x[:,0], x[:,1], f, c='black')
        plt.xlabel('$x1_*$')
        plt.ylabel('$x2_*$', rotation=0)
        cbar = fig.colorbar(surf, shrink=0.5)
        cbar.set_label('Satisfaction Probability')
        plt.savefig(f'../figures/results/gpc/{name}_posteriorSurf.png', dpi=300)

        # Surface Plot with confidence bounds and training points
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10,10))
        surf = ax.plot_surface(l1, l2, lowerbound, cmap=cm.coolwarm, vmin=0, vmax=1)
        surf = ax.plot_surface(l1, l2, upperbound, cmap=cm.coolwarm, vmin=0, vmax=1)
        ax.scatter(x[:,0], x[:,1], f, c='black')
        plt.xlabel('$x1_*$')
        plt.ylabel('$x2_*$', rotation=0)
        cbar = fig.colorbar(surf, shrink=0.5)
        cbar.set_label('Satisfaction Probability')
        plt.savefig(f'../figures/results/gpc/{name}_posteriorSurfConf.png', dpi=300)

    else:
        print("Classification can only handle 1 or 2 dimensions.")

    return probabilities


def coeff_variation(probs, name):
    """
    Compute coefficient of variation for different posterior probabilities,
    plot together with mean and standard deviation of function

    Args:
        probs: list of probabilities of different posterior distributions
        name: name of experiment for saving plot
    """
    m = []  # mean
    s = []  # standard deviation
    cv = [] # coefficient of variation
    for t in probs:
        m.append(np.mean(probs[t]))
        s.append(np.std(probs[t]))
        cv.append(np.std(probs[t])/np.mean(probs[t]))
    plt.figure(figsize=(12,6))
    plt.plot(probs.keys(), m, lw=3, ls='--')
    plt.scatter(probs.keys(), m, marker='o', label='Mean')
    plt.plot(probs.keys(), s, lw=3, ls='--')
    plt.scatter(probs.keys(), s, marker='o', label='SD')
    plt.plot(probs.keys(), cv, lw=3, ls='-')
    plt.scatter(probs.keys(), cv, marker='o', label='$c_v$')
    plt.axhline(y = 0.1, color = 'black', ls = ':', label = 'low $c_v$')
    #plt.title('Coefficient of variation for different thresholds as: sd/mean')
    plt.xlabel('$t$', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim((0,1))
    plt.legend(fontsize=18)
    plt.tight_layout()        
    plt.savefig(f'../figures/results/gpc/{name}_variation.png', dpi=300)

    print('Coefficients of variation: ', cv)


def calc_mse(x, x_s, f, probs):
    """ Compute MSE for posterior function of the training data points
    
    Parameters:
        x: Training data input
        x_s: Test data input 
        f: Training data output
        probs: computed probabilities (posterior)
        
    Returns:  
        mse: Average distance of predictions to true training data output
    """
    predictions = []
    if len(x_s[0,:]) == 1:
        for i in x:
            idx = np.absolute(x_s - i).argmin()
            predictions.append(probs[idx])

        mse = np.square(np.subtract(f.reshape(1,-1), predictions)).mean()
    elif len(x_s[0,:]) == 2:
        probs = probs.reshape(len(x_s),-1)
        for i in x:
            idx = np.absolute(x_s - i).argmin()
            if idx > len(x_s):
                idx = len(x_s) - 1
            predictions.append(probs[idx])
        mse = np.square(np.subtract(f.reshape(1,-1), predictions)).mean()

    return mse




""" Run complete program 

def analyse_ex_paper():
    Run GPC for viral example of Smoothed Model Checking Paper (Bortolussi)


    # number of trajectories per input point
    scale = 5 
    # uncertain input parameter \theta that is varied
    paramValueSet = np.linspace(0.5, 5, 20).reshape(-1,1) 
    # statistical outputs of satisfaction 
    paramValueOutput = np.array([1,1,0.8,1,1,1,1,0.6,0.6,0.6,0.8,0,0.8,0.2,0.6,0,0.4,0.4,0,0.2]).reshape(-1,1) 

    # plot training data
    plot_training(paramValueSet, paramValueOutput, 'paper_ex')
    print("Actual data (number of runs satisfying property): ", (paramValueOutput * scale).reshape(1,-1))

    # define default hyperparameters for kernels
    params = {'var': 1/5,
            'ell': 1,        
            'ell_dim': [2, 5],
            'var_b': 1,
            'off': 1}

    # perform EP
    mu_tilde, invC = perform_ep(paramValueSet, paramValueOutput, scale, kernel_rbf, params)
    
    # derive predictive probabilities and confidence intervals, save plot
    testset = np.linspace(0, 5, 50).reshape(-1,1)
    get_posterior(paramValueSet, testset, paramValueOutput, mu_tilde, invC, kernel_rbf, params, 'paper_ex')
"""



def smmc_1dim(paramValueSet, paramValueOutput, name, scale, variance, ell, teststart, testend, testpoints, thresh = None):
    """
    Perform SMMC for 1-dimensional input

    Args:
        paramValueSet: training inputs
        paramValueOutput: training outputs (satisfaction probabilities)
        name: name of experiment to save plots
        scale: sample size (number of observations per data point)
        variance: hyperparameter of kernel
        ell: hyperparameter of kernel, 1- or 2-dim.
        teststart: first value of testset
        testend: last value of testset
        testpoints: number of points of testset, for which posterior should be evaluated
        thresh: threshold for fitness function, if available

    Returns: 
        p: predictive probabilities
        mse: MSE of predictions to training outputs
    """
    if thresh is not None:
        name = name + "_" + str(round(thresh, 4))

    plot_training(paramValueSet, paramValueOutput, name)

    # define default hyperparameters for kernels
    # variance = max-min / 2 for output values (if this is 0, set to 1) -> das noch scaled?
    # lengthscale = max - min / 10 for input values
    params = {'var': variance,
             'ell': ell,        
             'ell_dim': [1, 1],
             'var_b': 1,
             'off': 1}
   
    mu_tilde, invC = perform_ep(paramValueSet, paramValueOutput, scale, kernel_rbf, params)

    # derive predictive probabilities and confidence intervals, save plot
    testset = np.linspace(teststart, testend, testpoints).reshape(-1,1)
    p = get_posterior(paramValueSet, testset, paramValueOutput, mu_tilde, invC, kernel_rbf, params, name, thresh)

    # compute MSE
    mse = calc_mse(paramValueSet, testset, paramValueOutput, p)
    print("MSE: ", mse)

    return p



"""
def analyse_stoch(thresh, v, l, scale, teststart, testend, testpoints):
    
    Analyze satisfaction probability for different population sizes to find out if function is robust
    Input: histogram data for different population sizes n (simulated data - Stochnet)
    Then compute GPC of satisfaction probabilty
    Args:
        t: threshold for "min. t bees are alive after experiment"
        v: variance of kernel
        l: lengthscale of kernel
    Returns:
        p: predictive probabilities

    

    #scale = 1000
    thresh
    paramValueSet, paramValueOutput = read_stochnet(thresh, scale)
    plot_training(paramValueSet, paramValueOutput, f'bees_stochnet_{round(thresh, 2)}')

    # define default hyperparameters for kernels
    # variance = max-min / 2 for output values (if this is 0, set to 1) -> das noch scaled?
    # lengthscale = max - min / 10 for input values
    params = {'var': v,
            'ell': l,        
            'ell_dim': [2, 5],
            'var_b': 1,
            'off': 1}
   
    mu_tilde, invC = perform_ep(paramValueSet, paramValueOutput, scale, kernel_rbf, params)

    # derive predictive probabilities and confidence intervals, save plot
    testset = np.linspace(teststart, testend, testpoints).reshape(-1,1)
    p = get_posterior(paramValueSet, testset, paramValueOutput, mu_tilde, invC, kernel_rbf, params, f'bees_stochnet_{round(thresh, 2)}', thresh)

    return p
"""


# TODO: noch anpassen!
def analyse_stoch2(thresh, v, l, scale):
    """ 
    For 2 dimensions -> vary N and one of the rates k
    Analyze satisfaction probability for different population sizes to find out if function is robust
    Input: histogram data for different population sizes n (simulated data - Stochnet)
    Then compute GPC of satisfaction probabilty
    Args:
        t: threshold for "min. t bees are alive after experiment"
        v: variance of kernel
        l: lengthscale of kernel
    Returns:
        p: predictive probabilities

    """

    #scale = 1000
    #thresh = t
    paramValueSet, paramValueOutput = read_stochnet2(thresh, scale)
    plot_training(paramValueSet, paramValueOutput,  f'bees_stochnet2_{round(thresh, 4)}')

    
    # define default hyperparameters for kernels
    # variance = max-min / 2 for output values (if this is 0, set to 1)
    # lengthscale = max - min / 10 for input values

    
    params = {'var': v,
            'ell': 1,        
            'ell_dim': l,
            'var_b': 1,
            'off': 1}
   
    mu_tilde, invC = perform_ep(paramValueSet, paramValueOutput, scale, kernel_rbf_ard, params)

    # derive predictive probabilities and confidence intervals, save plot
    # define test set for which posterior is derived

    Ns = 225  # number of testpoints
    ptest1 = [5, 100]  # range of uncertain input parameter p1
    ptest2 = [0, 0.1]  # range of uncertain input parameter p2
    npoints = int(np.ceil(Ns**(1/2)))  # number of points per dimension
    testset = np.zeros((Ns,2))  # create grid of equally spaced input points
    lin1 = np.linspace(ptest1[0], ptest1[1], npoints)
    lin2 = np.linspace(ptest2[0], ptest2[1], npoints)
    testset[:,0] = np.repeat(lin1, npoints)
    testset[:,1] = np.tile(lin2, npoints)

    #testset = np.linspace(15, 150, 500).reshape(-1,1)
    p = get_posterior(paramValueSet, testset, paramValueOutput, mu_tilde, invC, kernel_rbf_ard, params, f'bees_stochnet2_{round(thresh, 4)}', thresh)

    # compute MSE
    mse = calc_mse(paramValueSet, testset, paramValueOutput, p)
    print("MSE: ", mse)

    return p
    


def analyse_prism_bee2(v, l, scale):
    """ 
    For 2 dimensions -> vary N and one of the rates k
    Analyze satisfaction probability for different population sizes to find out if function is robust
    Input: histogram data for different population sizes n (simulated data - Stochnet)
    Then compute GPC of satisfaction probabilty
    Args:
        t: threshold for "min. t bees are alive after experiment"
        v: variance of kernel
        l: lengthscale of kernel
    Returns:
        p: predictive probabilities

    """
    paramValueSet, paramValueOutput = read_bee_prism2()
    plot_training(paramValueSet, paramValueOutput,  'prism_bees2')
    
    
    params = {'var': v,
            'ell': 1,        
            'ell_dim': l,
            'var_b': 1,
            'off': 1}
   
    mu_tilde, invC = perform_ep(paramValueSet, paramValueOutput, scale, kernel_rbf_ard, params)

    # derive predictive probabilities and confidence intervals, save plot
    # define test set for which posterior is derived

    Ns = 225  # number of testpoints
    ptest1 = [0, 1]  # range of uncertain input parameter p1
    ptest2 = [0, 1]  # range of uncertain input parameter p2
    npoints = int(np.ceil(Ns**(1/2)))  # number of points per dimension
    testset = np.zeros((Ns,2))  # create grid of equally spaced input points
    lin1 = np.linspace(ptest1[0], ptest1[1], npoints)
    lin2 = np.linspace(ptest2[0], ptest2[1], npoints)
    testset[:,0] = np.repeat(lin1, npoints)
    testset[:,1] = np.tile(lin2, npoints)

    #testset = np.linspace(15, 150, 500).reshape(-1,1)
    p = get_posterior(paramValueSet, testset, paramValueOutput, mu_tilde, invC, kernel_rbf_ard, params, 'prism_bees2')

    # compute MSE
    mse = calc_mse(paramValueSet, testset, paramValueOutput, p)
    print("MSE: ", mse)

    return p
    
    


def main():
    # analyse example of Bortolussi paper to check if results are similar
    """
    print('-----BORTOLUSSI EXAMPLE-----')
    x = np.linspace(0.5, 5, 20).reshape(-1,1) 
    f = np.array([1,1,0.8,1,1,1,1,0.6,0.6,0.6,0.8,0,0.8,0.2,0.6,0,0.4,0.4,0,0.2]).reshape(-1,1) 
    smmc_1dim(x, f, 'paper_ex', 5, 1/5, 1, 0, 5, 50)
    """

    # analyse Stochnet CRN simulations to infer fitness function
    # property = What is the probability that >= t bees are alive after each run?
    """
    print('-----STOCHNET CRN 1 DIM FITNESS FUNCTION-----')
    probs = {}
    threshs = np.arange(0.09, 0.25, 0.02)
    scale = 100
    for t in threshs:
        print("t = ", t)
        x, f = read_stochnet(t, scale)
        probs[t] = smmc_1dim(x, f, 'bees_stochnet', scale, 0.15, 10, 10, 155, 500, t)
    coeff_variation(probs, f'bees_stochnet')
    """

    # find parameter k1 of CRN for fitness function with given threshold
    #analyse_stoch2(thresh = 0.11, v = 0.0005, l = [10, 0.01], scale = 1000)

    
    # analyse experiment data from Morgane
    """
    print('-----EXPERIMENTAL DATA 1 DIM FITNESS FUNCTION-----')
    col, outputs = read_hist_exp("bees_morgane/hist1_IAA.txt")
    probs = {}
    threshs = np.arange(0, 1.1, 0.1)
    for t in threshs:
        print("t = ", t)
        x, f = read_exp_smmc(col, outputs, t)
        probs[t] = smmc_1dim(x, f, 'bees_morganeB', 60, 0.2, 1.75, 0, 12, 100, t)
    coeff_variation(probs, f'bees_morganeB')
    """

    #PRISM DTMC example
    #print('-----PRISM DTMC 1 DIM-----\n')
    #x, f = read_bee_prism()
    #smmc_1dim(x, f, 'prism_bees', 50, 0.05, 0.1, 0, 1, 50)
    #print('-----PRISM DTMC 2 DIM-----\n')
    analyse_prism_bee2(0.05, [0.1,0.1], 50)





if __name__ == "__main__":
    sys.exit(main())