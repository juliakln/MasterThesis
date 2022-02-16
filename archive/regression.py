"""
Use standard regression methods for an example with single valued output
to compare with results of GPR

"""

import sys
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np
from create_data import *
from sklearn import linear_model



def linear_reg(x, y, x_true):
    """ Ordinary Least Squares Linear Regression
    Parameters:
    x : numpy array
        Training x data
    y : numpy array
        Training y data we perform the regression on
    x_true : numpy array
        All x values the true baseline function was evaluated on

    Returns:
    reg.intercept_ : float
        Intercept of linear function
    reg.coef_ : array
        Coefficients of linear function
    reg.score : float
        Coefficient of determination R^2 of prediction
    predictions : array
        Returns predicted values
    """
    x_true = x_true.reshape(-1,1)
    reg = linear_model.LinearRegression()
    reg.fit(x, y)
    predictions = reg.predict(x_true)
    return reg, predictions

def plot_linear_reg(x_true, y_true, x, y, reg_pred):
    plt.figure(figsize=(12,6))
    plt.plot(x_true, y_true, label='True Function')
    plt.scatter(x, y, color='red', label='True Data Points')
    plt.plot(x_true, reg_pred, 'g-', label='Linear Regression')
    plt.xlabel('colony size')
    plt.ylabel('y')
    plt.legend(loc='upper left')
    plt.savefig('figures/results/single_output_linearreg.png')


def loocv(x, x_true, y):
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

    Returns:
    predictions : list
        List of predictions of left-out training data points    
    l2dist : float
        Distance of predictions to true training data output
    """
    predictions = []
    N = len(x) - 1
    for leave_out in range(N+1):
        x_new = np.delete(x, leave_out).reshape(-1,1)
        y_new = np.delete(y, leave_out).reshape(-1,1)

        reg, pred = linear_reg(x_new, y_new, x_true)        

        #print(pred)

        # prediction for left out data point
        idx_pred = np.absolute(x_true-x[leave_out]).argmin()
        #print(idx_pred)
        predictions.append(pred.item(idx_pred))

    l2dist = np.linalg.norm(y.reshape(1,-1) - predictions)

    return predictions, l2dist



def main():
    # import data
    [x_true,y_true], [x,y] = create_single_output()
    x = x.reshape(-1,1)
    reg, reg_pred = linear_reg(x, y, x_true)
    print('R^2 = ', reg.score(x,y))
    plot_linear_reg(x_true,y_true,x,y,reg_pred)
    l2dist = np.linalg.norm(y.reshape(1,-1) - reg.predict(x))
    print('L2 distance: ', l2dist)
    print('Predictions: ', reg.predict(x))

    p, l = loocv(x, x_true, y)
    print(p)
    print(l)


if __name__ == "__main__":
    sys.exit(main())