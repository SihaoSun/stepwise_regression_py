from cmath import inf
import numpy as np
import matplotlib.pyplot as plt
from stepwise_regression.helper import backward_elimination, find_PSE, find_R2, find_RMS, forward_selection

def stepwise_model_structure(A, X, z, stop_criteria, plot_report):
    x_out = np.array([])
    log = []
    
    step = 0
    
    num_samples = A.shape[0]
    num_regressors = A.shape[1]
    N = num_samples
    p = num_regressors
    
    k = np.linalg.lstsq(A, z, rcond=None)[0]
    y = A @ np.vstack(k)
    
    pse = find_PSE(y, z, p)
    r2 = find_R2(y, z)

    if p > 1:
        f0 = (num_samples - p) / (p - 1) * r2 / (1 - r2)
    else:
        f0 = np.inf
        
    pse_last = pse
    r2_last = r2
    f0_last = f0
    
    pse_tol = pse / 1000
    A_last = A
    
    print("Initial PSE is {}".format(r2))
    print("Initial R2 is {}".format(r2))
    
    if plot_report:
        fig = plt.figure(1)
        plt.plot(z)
    
    while 1:
        log.append(np.zeros(7))
        log[step][0] = step
        r = np.vstack(z) - np.vstack(y)
        
        # forward selection
        V = np.zeros(X.shape)
        for jj in range(X.shape[1]):
            x = X[:, jj]
            ka = np.linalg.lstsq(A, x, rcond = None)[0]
            V[:, jj] = np.squeeze(np.vstack(x) - A @ np.vstack(ka))
        
        cor = np.abs(np.corrcoef(V.T, r.T))[-1, 0:-1]
        j = np.where(cor == cor.max())[0].tolist()[0]
        _, select = forward_selection(X, A, z, 5, j)
        
        if select == True:
            Xin = X[:, j]
            A = np.hstack((A, np.vstack(Xin)))
            X = np.delete(X, j, 1)
        else:
            log.pop(j)
            print("No qualified candidates\n")
            break
        
        
        if np.array_equal(Xin.squeeze(), x_out.squeeze()):
            A = np.delete(A, -1, 1)
            k = np.linalg.lstsq(A, z)[0]
            y = A @ np.vstack(k)
            print("x_in equals x_out\n")
            break
        
        # backward elimination
        i, _, out = backward_elimination(A, z, Fout=4)
        if out == True:
            x_out = A[:, i]
            A = np.delete(A, i, 1)
        else:
            x_out = np.array([])
        
        # calculate parameters
        k = np.linalg.lstsq(A, z, rcond=None)[0]
        y = A @ np.vstack(k)
        
        if plot_report:
            plt.figure(1)
            plt.plot(y, 'r-')
        
        print("======= step = {} ==========\n".format(step))
        print("selected = {}\n".format(j))
        log[step][1] = j
        if out == True:
            print("kick out -> {}\n".format(i))
            log[step, 2] = i
        
        # define stopping criteria
        p = A.shape[1]
        pse = find_PSE(y, z, p)
        r2 = find_R2(y, z)
        rmse = find_RMS(y, z)
        if p > 1 and r2 < 1:
            f0 = (N - p) / (p - 1) * r2 / (1 - r2)
        else:
            f0 = np.inf
        
        log[step][3] = pse
        log[step][4] = r2
        log[step][5] = f0
        log[step][6] = rmse
        
        print("PSE = {}\n".format(pse))
        print("R2 = {}\n".format(r2))
        
        if stop_criteria == 'pse':
            if pse >= pse_last * 0.99:
                A = A_last
                k = np.vstack(np.linalg.lstsq(A, z)[0])
                print("STOP CRITERIA: pse increases\n")
                break
            if pse < pse_tol:
                print("STOP CRITIERA: pse is sufficiently small\n")
                break
        elif stop_criteria == 'r2':
            if r2 <= r2_last * 1.005:
                print('STOP CRITERIA: r2 has no significant growth\n')
                break
        elif stop_criteria == 'f0':
            if f0 <= f0_last:
                print('STOP CRITERIA: f0 reaches the maximum\n')
                break
        else:
            if step >= 10:
                print('STOP CRITERIA: no stopping criteria assigned, too many steps!\n')
    
        A_last = A
        pse_last = pse
        r2_last = r2
        f0_last = f0
        if not np.array(X).any():
            print("STOP CRITERIA: no candidate regressors left over steps\n")
            break
        
        step += 1
        
    return k, A , log