import numpy as np
from computeCost import *


def gradient_descent(X, y, theta, alpha, num_iters):
    # Initialize some useful values
    m = y.size
    J_history = np.zeros(num_iters)

    for i in range(0, num_iters):
        # ===================== Your Code Here =====================
        # Instructions : Perform a single gradient step on the parameter vector theta
        #
        # Hint: X.shape = (97, 2), y.shape = (97, ), theta.shape = (2, )
        
        ###### it is batch gradient descent, cuz we go through the entire sample for each iteration
        error = np.dot(X, theta) - y
        theta = theta - alpha/m * np.dot(error.T, X)
        # ===========================================================
        # Save the cost every iteration
        J_history[i] = compute_cost(X, y, theta)

    return theta, J_history


def gradient_descent_multi(X, y, theta, alpha, num_iters):
    # Initialize some useful values
    m = y.size
    J_history = np.zeros(num_iters)

    for i in range(0, num_iters):
        # ===================== Your Code Here =====================
        # Instructions : Perform a single gradient step on the parameter vector theta
        #
        error = np.dot(X, theta) - y
        theta = theta - alpha/m * np.dot(error.T, X)
        # ===========================================================
        # Save the cost every iteration
        J_history[i] = compute_cost(X, y, theta)

    return theta, J_history
    