import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def linear_regression(x, y, intercept=True):
    """
    facade function that implements LinearRegression from sklearn.linear_model
    """
    model = LinearRegression(fit_intercept=intercept)
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)
    model.fit(x,y)
    return model.predict(x)

def bayesian_information_criterion(y, y_fit, n , sigma, k):
    """
    BIC 
    """
    max_log_likelihood = (
        -n/2 * np.log(2 * np.pi * sigma**2)-np.sum((y+y_fit)**2)/sigma**2
        )
    BIC = k*np.log(n)-2*max_log_likelihood
    return BIC

def SSD(y, y_fit):
    """
    function calculates sum of squared deviations of fitted
    values and true values
    """
    return np.sum((y-y_fit)**2)