import random
import numpy as np

def wiener(n, dt, t_init=0, w_init=0.0):
    """Returns one realization of a Wiener process with n steps of length dt.
    The time and Wiener series can be initialized using t_init and w_init respectively."""
    n+=1
    t_series = np.arange(t_init,n*dt,dt)
    h = t_series[1]-t_series[0]
    z = np.random.normal(0.0,1.0,n)
    dw = np.sqrt(h)*z
    dw[0] = w_init
    w_series = dw.cumsum()
    return t_series, w_series

def get_dxs(x,lim=None):
    '''returns a finite difference series based on the input data
    
    x = input data series
    lim = returned differences are between +/-lim.'''
    return [x[i+1]-x[i] for i in range(len(x)-1) if lim==None or abs(x[i+1]-x[i])<lim]

def incr_res(T, W, c, mu=0, sigma=1):
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
            sig = np.sqrt(t_next-t)*sigma
            gaussian = np.exp(-dW**2/2/sig**2)/np.sqrt(2*np.pi)/sig
            if gaussian<random.random()<(1-(t_next-t)/dT):
                w_t+=np.abs(random.gauss(0,np.sqrt(dt)*sigma))*float(dW)/abs(dW)
            else:
                w_t+=random.gauss(0,np.sqrt(dt)*sigma)
            t_series.append(t)
            w_series.append(w_t)
    t_series.append(T[-1])
    w_series.append(W[-1])
    return t_series,w_series

def exp_estep(n, dt, a, b, m=1, mu=0, sigma=1, w_init=0, p_init=1, chain_length=0): 
    '''Moves a price series forward through n time steps using Euler's method.
    
    Details
    -------
    The Stochastic DE conatins a drift term and growth is reduced by noise.
    Returns a price series along with its corisponding wiener and time series.
    
    Argumets
    --------
    n = The number of steps taken
    dt = The size of the time steps
    a = Constant controling the significance of price drift
    b = Constant controling the significance of the noise
    m = The default case refers to standard geometric brownian motion
    mu = Mean of the normal distribution
    sigma = Standard deviation of the normal distribution
    w_init = Initial value of the wiener series
    p_init = Initial value of the price series. If chaining, give p_init of the
        previously existing series which the returned one will be linked to.
    chain_length = The number of time steps in the previously existing series which
        the returned ones will be linked to. (default assumes no chaining)
    
    (!) Note: it is suggested that all arguments be given as floats.
    '''
    w_t = w_init
    p_t = p_init
    if chain_length>0:
        t_series = np.arange(chain_length*dt,(n+chain_length-0.9)*dt,dt)
        w_series = []
        p_series = []
    else:
        t_series = np.arange(0,(n+0.1)*dt,dt)
        w_series = [w_init]
        p_series = [p_init]
    for i in range(n):
        dw = random.gauss(mu,np.sqrt(dt)*sigma)
        # Incrament p(t) with dp as given by the DE.
        # Incrament w(t) with dw as given by the normal distribution
        p_t += (a*dt+b*dw)*p_t**(m)
        w_t += dw
        # Append new values to each series;
        w_series.append(w_t)
        p_series.append(p_t)
    return t_series,w_series,p_series

def exp_astep(n, dt, a, b, m=1, mu=0, sigma=1, w_init=0, p_init=1, chain_length=0):
    '''Moves a price series forward through n time steps using an analytical method.
    
    Details
    -------
    The Stochastic DE conatins a drift term and growth is reduced by noise.
    Returns a price series along with its corisponding wiener and time series.
    
    Argumets
    ---------
    n = The number of steps taken
    dt = The size of the time steps
    a = Constant controling the significance of price drift
    b = Constant controling the significance of the noise
    m = The default case refers to standard geometric brownian motion
    mu = Mean of the normal distribution
    sigma = Standard deviation of the normal distribution
    w_init = Initial value of the wiener series
    p_init = Initial value of the price series. If chaining, give p_init of the
        previously existing series which the returned one will be linked to.
    chain_length = The length of the previously existing price series
        (default assumes no chaining)
    
    (!) Note: it is suggested that all arguments be given as floats.
    '''
    w_t = w_init
    p_t = p_init
    if chain_length>0:
        t_series = np.arange(chain_length*dt,(n+chain_length-0.9)*dt,dt)
        w_series = []
        p_series = []
        trim = 0
    else:
        t_series = np.arange(0,(n+0.1)*dt,dt)
        w_series = [float(w_init)]
        p_series = [float(p_init)]
        trim = 1
    for t in t_series[trim:]:
        dw = random.gauss(mu,np.sqrt(dt)*sigma)
        w_t += dw
        p_series.append(p_init*np.exp((a-0.5*b**2)*t + b*w_t))
        w_series.append(w_t)
    return t_series,w_series,p_series

def exp_etrace(series, x, y_init, dt, a, b, m=1, mu=0, sigma=1, chain=False):
    """Returns a price/wiener series based on a retrace of a wiener/price series.
    
    series = 'wiener'
    -----------------
    Retraces the values of the wiener series to create a price series.
    x = The retraced wiener series.
    y_init = Initial value of the price series.
    
    series = 'price':
    -----------------
    Retraces the values of the price series to create a wiener series.
    x = The retraced price series.
    y_init = Initial value of the weiner series.
    
    Parameters
    ----------
    dt = The size of the time steps
    a = Constant controling the significance of price drift
    b = Constant controling the significance of the noise
    m = The default case refers to standard geometric brownian motion
    mu = Mean of the normal distribution
    sigma = Standard deviation of the normal distribution
    chain = Return the series excluding its initial value?
    
    (!) Note: it is suggested that all arguments be given as floats.
    """
    
    if series == 'wiener':
        p_t = y_init
        w_series = x
        if chain==True:
            p_series = []
        else:
            p_series = [y_init]
        for i in range(1,len(w_series)):
            dw = w_series[i]-w_series[i-1]
            p_t += (a*dt+b*dw)*p_t**(m)
            p_series.append(p_t)
        return p_series
            
    elif series == 'price':
        w_t = y_init
        p_series = x
        if chain==True:
            w_series = []
        else:
            w_series = [y_init]
        for i in range(1,len(p_series)):
            dP = (p_series[i] - p_series[i-1])/(p_series[i-1]**(m))
            w_t+=(dP-a*dt)/b
            w_series.append(w_t)
        return w_series
    
def exp_atrace(series, x, dt, a, b, m=1, mu=0, sigma=1, p_init=1, chain=False):
    """Returns a price/wiener series by tracing over a wiener/price series.
    
    series = 'wiener'
    -----------------
    Retraces the values of the wiener series to create a price series.
    x = a wiener series.
    
    series = 'price':
    -----------------
    Retraces the values of the price series to create a wiener series.
    x = a price series.
    
    Parameters
    ----------
    dt = The size of the time steps.
    a = Constant controling the significance of price drift.
    b = Constant controling the significance of the noise.
    m = The default case refers to standard geometric brownian motion.
    mu = Mean of the normal distribution.
    sigma = Standard deviation of the normal distribution.
    p_init = Initial value of the price series.
    chain = Return the series excluding its initial value?
    
    (!) Note: it is suggested that all arguments be given as floats.
    """
    
    
    if series == 'wiener':
        p_series = []
        w_series = x
        if chain==True:
            i = 1
        else:
            i = 0
        for t in [dt*j for j in range(i,len(w_series))]:
            alph = (a-0.5*b**2)*t
            beta = b*w_series[i]
            p_series.append(p_init*np.exp(alph + beta))
            i+=1
        return p_series
    
    
    if series == 'price':
        p_series = x
        w_series = []
        if chain==True:
            i = 1
        else:
            i = 0
        for t in [dt*j for j in range(i,len(p_series))]:
            alph = (a-0.5*b**2)*t
            w_series.append((np.log(float(p_series[i])/p_init)-alph)/b)
            i+=1
        return w_series