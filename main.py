import numpy as np
from stepwise_regression.stepwise_regression_func import stepwise_model_structure

def main():

    t = np.arange(0, 100.1, 0.1, dtype=float)
    x1 = np.sin(t)
    x2 = np.cos(3 * t)
    
    # define measurements
    y = 0.4 + 1.0 * x1 + 0.5 * x2 + 0.2 * x1 * x2
    z = y + np.random.rand(y.size) * 0.1
    y = np.vstack(y)
    z = np.vstack(z)
    
    # define candidate regressors
    X = np.array([x1, x2, x1**2, np.sin(x2**2), x1*x2, x1*x2**2]).T
    
    # initialize the regressor
    A = np.vstack(np.ones(t.size).T)
    
    # call stepwise regression
    k_final, A_final, log = stepwise_model_structure(A, X, z, stop_criteria='r2', plot_report=True)
    
    print("k final = {}".format(k_final))
    
if __name__ == "__main__":
    exit(main())