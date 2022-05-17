import numpy as np

def backward_elimination(A, z, Fout):
    num_of_sample   = A.shape[0]
    num_of_param  = A.shape[1]
    
    F   = np.inf * np.ones([num_of_param, 1])
    for jj in range(1, num_of_param):
        X   = A
        k0  = np.linalg.lstsq(X, z, rcond=None)[0]
        y0  = X @ k0
        SS0 = k0.T @ X.T @ z - num_of_sample * np.mean(z)**2
        
        X   = np.delete(X, jj, axis=1)
        k1  = np.linalg.lstsq(X, z, rcond=None)[0]
        y1  = X @ k1
        SS1 = k1.T @ X.T @ z - num_of_sample * np.mean(z)**2
        
        s2 = np.sum((z-y0)**2) / (num_of_sample-num_of_param)
        
        F[jj] = (SS0 - SS1) / s2
    F0  = np.min(np.abs(F))
    j   = np.where(F == F0)[0]
    if F0 < Fout:
        return j, F0, True
    else:
        return j, F0, False


def forward_selection(X, A, z, Fin, j):
    """Forward selection using F test

    Args:
        X (np.array): Candidate regressors
        A (np.array): Regressors including bias term as the first column, (N * np) matrix
        z (np.array): Measurement, N * 1 vector
        Fin (float): Gate, if F0 > Fin, elminate the corresponding regressor
        j (int): Index of the regressor with maximum F0
    
    Returns:
        F0 (float): F0 Value of the input regressors
        in (bool): Bool indicating if the regressor is selected
     """
    sampling_num    = A.shape[0]
    regressor_num   = A.shape[1]
    candidate_regressor_num  = X.shape[1]
    z = np.vstack(z)
    
    F = np.zeros((candidate_regressor_num, 1))
    XX = np.hstack((A, np.vstack(X[:,j])))
    k0 = np.linalg.lstsq(XX, z, rcond=None)[0]
    y0 = XX @ np.vstack(k0)
    ss0 = k0.T @ XX.T @ z - sampling_num * np.mean(z)**2
    
    XX = A
    k1 = np.linalg.lstsq(XX, z, rcond=None)[0]
    y1 = XX @ np.vstack(k1)
    ss1 = k1.T @ XX.T @ z - sampling_num * np.mean(z)**2
    
    s2 = np.sum((z - np.vstack(y0))**2) / (sampling_num - regressor_num - 1)
    
    F[j] = (ss0 - ss1) / s2         
    
    F0 = np.max(np.abs(F))
    if F0 > Fin:
        take_it = True
    else: 
        take_it = False
    
    return F0, take_it



def find_F(X, z):
    """Compute F_p for F-test

    Args:
        X (numpy.array): matrix with n regressors as columns
        z (numpy.array): measurement, N*1 demension vector
    
    Reference:
        [1] T. Lombaerts "Fault Tolerant Flight Control,a Physical Model Approach"  
        chapter 5.2.2 p167
    
    Author:
        Sihao Sun
        sihao.sun@outlook.com
        16-May-2022
    """
    
    N = X.shape[0]
    p = X.shape[1]
    
    theta   = np.linalg.inv(X.T @ X) @ X.T @ z
    rss     = np.sum((z - X @ theta)**2)
    cov     = np.linalg.inv(X.T @ X) * rss / (N - p)
    s2      = np.diag(cov)
    
    return theta.T**2 / s2

def find_PSE(y,z,p):
    """Compute predicted square error(PSE)

    Args:
        y (numpy.array): model output, 1d vector
        z (numpy.array): measurement, 1d vector
        p (numpy.array): number of regressors
    """
    z = z.squeeze()
    y = y.squeeze()
    
    N = len(z)
    e = z - y
    s_max = np.sum((z - np.mean(z))**2) / N
    
    pse = e @ e.T / N + s_max * p / N
    
    return pse

def find_R2(y, z):
    """Compute R2 to evaluate model accuracy

    Args:
        y (numpy.array): model output, 1d vector
        z (numpy.array): measurement, 1d vector
    """
    z = z.squeeze()
    y = y.squeeze()
    
    R2 = 1.0 - np.sum((z - y)**2) / np.sum((z - np.mean(z))**2)
    return R2

def find_RMS(x, z):
    """Compute normalized root mean square deviation

    Args:
        x (numpy.array): model output, 1d vector
        z (numpy.array): measurement, 1d vector
    """
    
    NRMSD = np.sqrt(np.mean((x.squeeze()-z.squeeze())**2)) / (z.max() - z.min())
    return NRMSD
