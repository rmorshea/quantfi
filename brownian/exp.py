from numpy import arange
from numpy import sqrt
from numpy import exp
from numpy import log
from random import gauss

def exp_estep(n, dt, a, b, m=1, mu=0, sigma=1, w_init=0, p_init=1, chain_length=0):
    '''Moves a price series forward through n time steps using Euler's method.
    
    Details
    -------
    The Stochastic DE conatins a drift term and growth is reduced by noise.
    Returns a price series along with its corisponding Wiener and time series.
    
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
        t_series = arange(chain_length*dt,(n+chain_length-0.9)*dt,dt)
        w_series = []
        p_series = []
    else:
        t_series = arange(0,(n+0.1)*dt,dt)
        w_series = [w_init]
        p_series = [p_init]
    for i in range(n):
        dw = gauss(mu,sqrt(dt)*sigma)
        # Incrament p(t) with dp as given by the DE.
        # Incrament w(t) with dw as given by the normal distribution
        p_t += (a*dt+b*dw)*p_t**(m)
        w_t += dw
        # Append new values to each series;
        w_series.append(w_t)
        p_series.append(p_t)
    return t_series,w_series,p_series

def exp_astep(n, dt, a, b, mu=0, sigma=1, w_init=0, p_init=1, chain_length=0):
    '''Moves a price series forward through n time steps using an analytical method.
    
    Details
    -------
    The Stochastic DE conatins a drift term and growth is reduced by noise.
    Returns an exp type price series with its corisponding Wiener and time series.
    Only analagous to exp_estep where m=1.
    
    Argumets
    ---------
    n = The number of steps taken
    dt = The size of the time steps
    a = Constant controling the significance of price drift
    b = Constant controling the significance of the noise
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
        t_series = arange(chain_length*dt,(n+chain_length-0.9)*dt,dt)
        w_series = []
        p_series = []
        trim = 0
    else:
        t_series = arange(0,(n+0.1)*dt,dt)
        w_series = [float(w_init)]
        p_series = [float(p_init)]
        trim = 1
    for t in t_series[trim:]:
        dw = gauss(mu,sqrt(dt)*sigma)
        w_t += dw
        p_series.append(p_init*exp((a-0.5*b**2)*t + b*w_t))
        w_series.append(w_t)
    return t_series,w_series,p_series

def exp_etrace(series, x, y_init, dt, a, b, m=1, mu=0, sigma=1, chain=False):
    """Returns a price/wiener series by tracing over a wiener/price series.
    
    series = 'wiener'
    -----------------
    Traces a wiener series with Euler's method to make a price series.
    x = The retraced wiener series.
    y_init = Initial value of the price series.
    
    series = 'price':
    -----------------
    Traces an exp type price series with Euler's method to make a wiener series.
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
    
def exp_atrace(series, x, dt, a, b, mu=0, sigma=1, p_init=1, chain=False):
    """Returns a price/wiener series by tracing over a wiener/price series.
    
    series = 'wiener'
    -----------------
    Analyticaly traces a wiener series to make a price series.
    x = a wiener series.
    
    series = 'price':
    -----------------
    Analyticaly traces an exp type price series to make a wiener series.
    x = a price series.
    
    Parameters
    ----------
    dt = The size of the time steps.
    a = Constant controling the significance of price drift.
    b = Constant controling the significance of the noise.
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
            p_series.append(p_init*exp(alph + beta))
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
            w_series.append((log(float(p_series[i])/p_init)-alph)/b)
            i+=1
        return w_series