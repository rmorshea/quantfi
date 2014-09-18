import numpy as np
from math import erfc
from random import random
from random import gauss

def w_series(n, dt, t_init=0, w_init=0.0):
    """Returns one realization of a Wiener process with n steps of length dt.
    The time and Wiener series can be initialized using t_init and w_init respectively.
    """
    n+=1
    t_series = np.arange(t_init,(n-0.1)*dt,dt)
    h = t_series[1]-t_series[0]
    z = np.random.normal(0.0,1.0,n)
    dw = np.sqrt(h)*z
    dw[0] = w_init
    w_series = dw.cumsum()
    return t_series, w_series

def raise_res(T, W, c, mu=0, sigma=1):
    '''Increase the resolution of a wiener series by a factor of c.
        
        Returns a more reolved Wiener series and its associate time series
        
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
            drawfrm_cum = np.sqrt(2)*np.sqrt(t_next-t)*sigma*erfc(random())
            if np.sqrt(2)*np.sqrt(t_next-t)*sigma*erfc(-2*random())<abs(dW):
                w_t+=abs(gauss(0,np.sqrt(dt)*sigma))*float(dW)/abs(dW)
            else:
                w_t+=gauss(0,np.sqrt(dt)*sigma)
            t_series.append(t)
            w_series.append(w_t)
    t_series.append(T[-1])
    w_series.append(W[-1])
    return t_series,w_series