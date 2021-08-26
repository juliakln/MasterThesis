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


if __name__ == "__main__":
    sys.exit(main())