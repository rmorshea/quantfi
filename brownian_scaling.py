import random
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

def gaussian(x, mu, sig):
    return np.exp(-(x-mu)**2/2/sig**2)/np.sqrt(2*np.pi)/sig

def brownian_scaling(T, W, c, mu=0, sigma=1):
    '''Increase the resolution of a wiener series by a factor of c.
    
    T = the given Time series.
    W = the associated Wiener series.
    c = Scaling factor (integer greater than 1).
    mu = Mean of W's underlying normal distribution.
    sigma = Standard deviation of W's underlying normal distribution.
    '''
    dT = T[1]-T[0]
    dt = float(T[1]-T[0])/c
    t_series = []
    w_series = []
    for i in range(len(T)-1):
        t = T[i]
        w_t = W[i]
        t_next = T[i+1]
        w_next = W[i+1]
        t_series.append(t)
        w_series.append(w_t)
        for j in range(c-1):
            t+=dt
            dW = (w_next-w_t)
            if gaussian(dW,0,np.sqrt(t_next-t)*sigma)<random.random()<(1-(t_next-t)/dT):
                w_t+=np.abs(random.gauss(0,np.sqrt(dt)*sigma))*float(dW)/abs(dW)
            else:
                w_t+=random.gauss(0,np.sqrt(dt)*sigma)
            t_series.append(t)
            w_series.append(w_t)
    t_series.append(T[-1])
    w_series.append(W[-1])
    return t_series,w_series