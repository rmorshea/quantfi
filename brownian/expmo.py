import numpy as np
from random import gauss

def estep(n, dt, a, b, c, m=1, mu=0, sigma=1, w_init=0., p_init=1., chain=(0,1)):
    '''Moves a price series forward through n time steps using Euler's method.
    
    Details
    -------
    The Stochastic DE has drift and momentum terms where growth is reduced by noise.
    Returns an expmo type price series with its corisponding Wiener and time series.
    
    Argumets
    --------
    n = The number of steps taken.
    dt = The size of the time steps.
    a = Constant controling the significance of price drift.
    b = Constant controling the significance of the noise.
    c = Constant controling the influence of momentum (c=0 implies no influence).
    m = The default case refers to standard geometric brownian motion.
    mu = Mean of the normal distribution.
    sigma = Standard deviation of the normal distribution.
    w_init = Initial value of the wiener series.
    p_init = Initial value of the price series. If chaining, give p_init of the
        previously existing series which the returned one will be linked to.
    chain = Tuple with length of previously existing price series and ending value.
    
    (!) Note: it is suggested that all arguments be given as floats.
    '''
    w_t = w_init
    p_t = p_init
    p_last = p_init
    if chain[0]>0:
        t_series = np.arange(chain_length*dt,(n+chain_length-0.9)*dt,dt)
        w_series = []
        p_series = []
    else:
        t_series = np.arange(0,(n+0.1)*dt,dt)
        w_series = [w_init]
        p_series = [p_init]
    for i in range(n):
        dw = gauss(mu,np.sqrt(dt)*sigma)
        dp = (p_t-p_last)/p_last
        p_last = p_t
        #incrament p(t) and w(t), then append values
        w_t += dw
        p_t += (a*dt + c*dp*dt + b*dw)*p_t**(m)
        w_series.append(w_t)
        p_series.append(p_t)
    return t_series,w_series,p_series

def astep(n, dt, a, b, c, mu=0, sigma=1, w_init=0., p_init=1., chain_length=0):
    '''Moves a price series forward through n time steps using an analytical method.
    
    Details
    -------
    The Stochastic DE has drift and momentum terms where growth is reduced by noise.
    Returns an expmo type price series with its corisponding Wiener and time series.
    Only analagous to expm_estep where m=1.
    
    Argumets
    --------
    n = The number of steps taken.
    dt = The size of the time steps.
    a = Constant controling the significance of price drift.
    b = Constant controling the significance of the noise.
    c = Constant controling the influence of momentum (c!=1, c=0 => no influence).
    mu = Mean of the normal distribution.
    sigma = Standard deviation of the normal distribution.
    w_init = Initial value of the wiener series.
    p_init = Initial value of the price series. If chaining, give p_init of the
        previously existing series which the returned one will be linked to.
    chain = Tuple with length of previously existing price series and ending value.
    
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
        nu = gauss(mu,np.sqrt(dt)*sigma)
        w_t += nu
        p_series.append(p_init*np.exp(((a-b**2/2./(1-c*dt))*t+b*w_t)/(1-c*dt)))
        w_series.append(w_t)
    return t_series,w_series,p_series

def etrace(series, x, y_init, dt, a, b, c, m=1, mu=0, sigma=1, y_last=None, chain=False):
    """Returns a price/wiener series by tracing over a wiener/price series.
    
    series = 'wiener'
    -----------------
    Traces a wiener series with Euler's method to make an expmo type price series.
    x = The retraced wiener series.
    y_init = Initial value of the price series.
    
    series = 'price':
    -----------------
    Traces an expmo type price series with Euler's method to make a wiener series.
    x = The retraced price series.
    y_init = Initial value of the weiner series.
    
    Parameters
    ----------
    dt = The size of the time steps
    a = Constant controling the significance of price drift
    b = Constant controling the significance of the noise
    c = Constant controling the influence of momentum (c!=1, c=0 => no influence).
    m = The default case refers to standard geometric brownian motion
    mu = Mean of the normal distribution
    sigma = Standard deviation of the normal distribution
    chain = Return the series excluding its initial value?
    
    (!) Note: it is suggested that all arguments be given as floats."""

    if series == 'wiener':
        w_series = x
        p_t = y_init
        p_last = y_init
        if chain==True:
            p_series = []
        else:
            p_series = [float(y_init)]
        for i in range(1,len(w_series)):
            dw = w_series[i]-w_series[i-1]
            dp = (p_t-p_last)/p_last
            p_last = p_t
            #incrament p(t) and w(t), then append values
            p_t += (a*dt + c*dp*dt + b*dw)*p_t**(m)
            p_series.append(p_t)
        
        return p_series

    elif series == 'price':
        if y_last == None:
            if chain == True:
                raise TypeError('must input a value for y_last')
            else:
                y_last = 1
        w_t = y_init
        p_series = [y_last] + x
        if chain==True:
            w_series = []
        else:
            w_series = [float(y_init)]
        for i in range(2,len(p_series)):
            dp0 = (p_series[i-1] - p_series[i-2])/p_series[i-2]
            dp1 = (p_series[i] - p_series[i-1])
            w_t+=(dp1/p_series[i-1]**(m) - a*dt - c*dp0*dt)/b
            w_series.append(w_t)
        return w_series

def atrace(series, x, dt, a, b, c, mu=0, sigma=1, p_init=1, chain=False):
    """Returns a price/wiener series by tracing over a wiener/price series.
    
    series = 'wiener'
    -----------------
    Analyticaly traces a wiener series to make an expmo type price series.
    x = a wiener series.
    
    series = 'price':
    -----------------
    Analyticaly traces an expmo type price series to make a wiener series.
    x = a price series.
    
    Parameters
    ----------
    dt = The size of the time steps.
    a = Constant controling the significance of price drift.
    b = Constant controling the significance of the noise.
    c = Constant controling the influence of momentum (c!=1, c=0 => no influence).
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
            p_series.append(p_init*np.exp(((a-b**2/2./(1-c*dt))*t+b*w_series[i])/(1-c*dt)))
            i+=1
        return p_series

    if series == 'price':
        w_series = [0.]
        p_series = x
        i = 1
        mu0 = 0
        for t in [dt*j for j in range(1,len(p_series))]:
            w_series.append((np.log(p_series[i]/p_init)*(1-c*dt)-(a-b**2/2./(1-c*dt))*t)/b)
            i+=1
        return w_series