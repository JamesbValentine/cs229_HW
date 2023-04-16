import numpy as np


def compute_cost(X, y, theta):
    # Initialize some useful values
    m = y.size
    cost = 0

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta.
    #                You should set the variable "cost" to the correct value.
    cost = np.sum((np.dot(X, theta) - y) ** 2) / (2 * m)
    
    ### Three ways to calculate the cost function: the most important thing is dimension
    a = np.inner(np.dot(X, theta) - y, np.dot(X, theta) - y) / (2 * m)

    b = np.sum((np.dot(X, theta) - y) ** 2) / (2 * m)

    c = np.dot((np.dot(X, theta) - y), (np.dot(X, theta) - y).T) / (2 * m)
    # ==========================================================

    return cost
