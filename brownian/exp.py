import numpy as np
from random import gauss
from utility_funcs import get_fds
import matplotlib.pyplot as plt
from matplotlib.mlab import normpdf
from IPython.html.widgets import interact
from scipy.stats import ks_2samp

def estep(n, dt, a, b, m=1, mu=0, sigma=1, w_init=0, p_init=1, chain_length=0):
    '''Moves a price series forward through n time steps using Euler's method.
    
    Details
    -------
    The Stochastic DE conatins a drift term and growth is reduced by noise.
    Returns an exp type price series with its corisponding Wiener and time series.
    
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
        dw = gauss(mu,np.sqrt(dt)*sigma)
        # Incrament p(t) with dp as given by the DE.
        # Incrament w(t) with dw as given by the normal distribution
        p_t += (a*dt+b*dw)*p_t**(m)
        w_t += dw
        # Append new values to each series;
        w_series.append(w_t)
        p_series.append(p_t)
    return t_series,w_series,p_series

def astep(n, dt, a, b, mu=0, sigma=1, w_init=0, p_init=1, chain_length=0):
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
        dw = gauss(mu,np.sqrt(dt)*sigma)
        w_t += dw
        p_series.append(p_init*np.exp((a-0.5*b**2)*t + b*w_t))
        w_series.append(w_t)
    return t_series,w_series,p_series

def etrace(series, ts, x, a, b, y_init, m=1, mu=0, sigma=1, chain=False):
    """Returns a price/wiener series by tracing over a wiener/price series.
    
    ts = The corrisponding time series to x where ts[0]=0

    series = 'wiener'
    -----------------
    Traces a wiener series with Euler's method to make an exp type price series.
    x = The wiener series being retraced.
    y_init = Initial value of the price series.
    
    series = 'price':
    -----------------
    Traces an exp type price series with Euler's method to make a wiener series.
    x = The price series being retraced.
    y_init = Initial value of the weiner series.
    
    Parameters
    ----------
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
            dt = ts[i]-ts[i-1]
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
            dt = ts[i]-ts[i-1]
            dP = (p_series[i] - p_series[i-1])/(p_series[i-1]**(m))
            w_t+=(dP-a*dt)/b
            w_series.append(w_t)
        return w_series
    
def atrace(series, ts, x, a, b, mu=0, sigma=1, p_init=1, chain=False):
    """Returns a price/wiener series by tracing over a wiener/price series.
    
    ts = The corrisponding time series to x where ts[0]=0

    series = 'wiener'
    -----------------
    Analyticaly traces a wiener series to make an exp type price series.
    x = a wiener series.
    
    series = 'price':
    -----------------
    Analyticaly traces an exp type price series to make a wiener series.
    x = a price series.
    
    Parameters
    ----------
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
        for t in ts:
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
        for t in ts:
            alph = (a-0.5*b**2)*t
            w_series.append((np.log(float(p_series[i])/p_init)-alph)/b)
            i+=1
        return w_series

def ksplot(func,alims,blims,t_series,dat_series,*func_auxargs):
    """Compare traced data to a normal distribution with a KS test.

    Returns output of a KS test:
    + The KS statistic
    + The corrisponding p-value
    
    Returns two plots:
    + A trace of the give data and time series
    + A histogram of the trace's scaled fds and a fit gaussian.

    Sliders are used to vary the parameters 'a' and 'b' in func.
    
    Arguments
    ---------
    func = use etrace or atrace to act on dat_series and t_series
    alims = slider limits for the paramter 'a' in func
    blims = slider limits for the paramter 'b' in func
    t_series = the time series corrisponding to dat_series
    dat_series = the data set which will be traced
    func_auxargs = auxilary arguments for func.
    """
    def makeplots(a,b):
        dat = func('price',t_series,dat_series,a,b,*func_auxargs)
        fig, ax = plt.subplots(figsize=(10,2))
        ax.set_title('Data After Trace')
        ax.set_ylabel('wiener',fontsize=13)
        ax.set_xlabel('time',fontsize=13)
        ax.set_xlim(min(t_series),max(t_series))
        ax.plot(t_series,dat)
        
        bnum = 50
        dat_fds = np.array(get_fds(dat))
        t_fds = np.array(get_fds(t_series))
        dat_fds = dat_fds/t_fds
        dat_cnt,dat_mkr = np.histogram(dat_fds,bnum)
        norm_mkr = np.linspace(min(dat_mkr),max(dat_mkr),100)
        norm_cnt = normpdf(norm_mkr,0,np.std(dat_fds))
        norm_cnt = norm_cnt*max(dat_cnt)/max(norm_cnt)
        ksstat,pval = ks_2samp(norm_cnt, dat_cnt)
        print 'KS test using scaled fds of traced dat and a fit gaussian:'
        print 'Statistic value =',ksstat
        print 'Two sided p-value =',pval
        
        fig,ax1 = plt.subplots(figsize=(10,4))
        ax1.set_title('Finite Difference Series Comparison')
        ax1.set_ylabel('count',fontsize=13)
        ax1.set_xlabel('difference',fontsize=13)
        
        wdth = float(max(dat_mkr)-min(dat_mkr))/bnum
        plt.bar(dat_mkr[:-1],dat_cnt,label='traced fds (scaled by dt)',width = wdth,align='center')
        plt.plot(norm_mkr,norm_cnt,label='fit gaussian',color='m')
        ax1.set_xlim(min(dat_mkr)-wdth,max(dat_mkr)+wdth)
        plt.legend(loc='best')
        plt.show()

    interact(makeplots,a=alims,b=blims)