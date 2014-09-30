import numpy as np
from random import gauss
import utility_funcs as ufuncs
import matplotlib.pyplot as plt
from matplotlib.mlab import normpdf
from IPython.html.widgets import interact
from scipy.stats import ks_2samp
from scipy.optimize import minimize

def estep(n, dt, mu, sigma, m=1, mnv=0, var=1, w_init=0, p_init=1, chain_length=0):
    '''Moves a price series forward through n time steps using Euler's method.
    
    Details
    -------
    The Stochastic DE conatins a drift term and growth is reduced by noise.
    Returns an exp type price series with its corisponding Wiener and time series.
    
    Argumets
    --------
    n = The number of steps taken
    dt = The size of the time steps
    mu = Constant controling the significance of price drift
    sigma = Constant controling the significance of the noise
    m = The default case refers to standard geometric brownian motion
    mnv = The mean value of the normal distribution
    var = Standard deviation of the normal distribution
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
        dw = gauss(mnv,np.sqrt(dt)*var)
        # Incrament p(t) with dp as given by the DE.
        # Incrament w(t) with dw as given by the normal distribution
        p_t += (mu*dt+sigma*dw)*p_t**(m)
        w_t += dw
        # Append new values to each series;
        w_series.append(w_t)
        p_series.append(p_t)
    return t_series,w_series,p_series

def astep(n, dt, mu, sigma, mnv=0, var=1, w_init=0, p_init=1, chain_length=0):
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
    mu = Constant controling the significance of price drift
    sigma = Constant controling the significance of the noise
    mnv = The mean value of the normal distribution
    var = Standard deviation of the normal distribution
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
        dw = gauss(mnv,np.sqrt(dt)*var)
        w_t += dw
        p_series.append(p_init*np.exp((mu-0.5*sigma**2)*t + sigma*w_t))
        w_series.append(w_t)
    return t_series,w_series,p_series

def etrace(stype, ts, x, mu, sigma, y_init, m=1, mnv=0, vari=1, chain=False):
    """Returns a price/wiener series by tracing over a wiener/price series.
    
    stype = The type of series provided for the argument x
    ts = The corrisponding time series to x where ts[0]=0

    stype = 'wiener'
    -----------------
    Traces a wiener series with Euler's method to make an exp type price series.
    x = The wiener series being retraced.
    y_init = Initial value of the price series.
    
    stype = 'price':
    -----------------
    Traces an exp type price series with Euler's method to make a wiener series.
    x = The price series being retraced.
    y_init = Initial value of the weiner series.
    
    Parameters
    ----------
    mu = Constant controling the significance of price drift
    sigma = Constant controling the significance of the noise
    m = The default case refers to standard geometric brownian motion
    mnv = The mean value of the normal distribution
    vari = Standard deviation of the normal distribution
    chain = Return the series excluding its initial value?
    
    (!) Note: it is suggested that all arguments be given as floats.
    """
    
    if stype == 'wiener':
        p_t = y_init
        w_series = x
        if chain==True:
            p_series = []
        else:
            p_series = [y_init]
        for i in range(1,len(w_series)):
            dt = ts[i]-ts[i-1]
            dw = w_series[i]-w_series[i-1]
            p_t += (mu*dt+sigma*dw)*p_t**(m)
            p_series.append(p_t)
        return p_series
            
    elif stype == 'price':
        w_t = y_init
        p_series = x
        if chain==True:
            w_series = []
        else:
            w_series = [y_init]
        for i in range(1,len(p_series)):
            dt = ts[i]-ts[i-1]
            dP = (p_series[i] - p_series[i-1])/(p_series[i-1]**(m))
            w_t+=(dP-mu*dt)/sigma
            w_series.append(w_t)
        return w_series
    
def atrace(stype, ts, x, mu, sigma, p_init=1, chain=False):
    """Returns a price/wiener series by tracing over a wiener/price series.
    
    stype = The type of series provided for the argument x
    ts = The corrisponding time series to x where ts[0]=0

    stype = 'wiener'
    -----------------
    Analyticaly traces a wiener series to make an exp type price series.
    x = A wiener series.
    p_init = Initial value of the price series.
    
    stype = 'price':
    -----------------
    Analyticaly traces an exp type price series to make a wiener series.
    x = A price series.
    
    Parameters
    ----------
    mu = Constant controling the significance of price drift.
    sigma = Constant controling the significance of the noise.
    chain = Return the series excluding its initial value?
    """
    
    
    if stype == 'wiener':
        p_series = []
        w_series = x
        if chain==True:
            i = 1
        else:
            i = 0
        for t in ts:
            alph = (mu-0.5*sigma**2)*t
            beta = sigma*w_series[i]
            p_series.append(p_init*np.exp(alph + beta))
            i+=1
        return p_series
    
    
    if stype == 'price':
        p_init = x[0]
        p_series = x
        w_series = []
        if chain==True:
            i = 1
        else:
            i = 0
        for t in ts:
            alph = (mu-0.5*sigma**2)*t
            w_series.append((np.log(float(p_series[i])/p_init)-alph)/sigma)
            i+=1
        return w_series

def ksplot(func,mu_lims,sig_lims,t_series,p_series,bar_num,*func_auxargs):
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
    func = use etrace or atrace to act on p_series and t_series
    mu_lims = slider limits for the paramter 'a' in func
    sig_lims = slider limits for the paramter 'b' in func
    t_series = the time series corrisponding to p_series
    p_series = the data set which will be traced
    bar_num = number of bars in the histogramed plot
    func_auxargs = auxilary arguments for func
    """
    def makeplots(a,b):
        dat = func('price',t_series,p_series,a,b,*func_auxargs)
        fig, ax = plt.subplots(figsize=(10,2))
        ax.set_title('Data After Trace')
        ax.set_ylabel('wiener',fontsize=13)
        ax.set_xlabel('time',fontsize=13)
        ax.set_xlim(min(t_series),max(t_series))
        ax.plot(t_series,dat)
        
        dat_fds = np.array(ufuncs.get_fds(dat))
        dat_cnt,dat_mkr = np.histogram(dat_fds,bar_num)
        norm_hist = np.histogram(dat_fds,bar_num)[0]
        cnt_norm = normpdf(dat_mkr,0,np.std(dat_fds))
        cnt_norm = cnt_norm*max(dat_cnt)/max(cnt_norm)
        ksstat,pval = ks_2samp(cnt_norm, dat_cnt)
        print 'KS test using scaled fds of traced dat and a fit gaussian:'
        print 'Statistic value =',ksstat
        print 'Two sided p-value =',pval
        
        fig,ax1 = plt.subplots(figsize=(10,4))
        ax1.set_title('Finite Difference Series Comparison')
        ax1.set_ylabel('count',fontsize=13)
        ax1.set_xlabel('difference',fontsize=13)
        
        wdth = float(max(dat_mkr)-min(dat_mkr))/bar_num
        plt.bar(dat_mkr[:-1],dat_cnt,label='traced fds (scaled by dt)',width = wdth,align='center')
        plt.plot(dat_mkr,cnt_norm,label='fit gaussian',color='m')
        ax1.set_xlim(min(dat_mkr)-wdth,max(dat_mkr)+wdth)
        plt.legend(loc='best')
        plt.show()

    interact(makeplots,a=mu_lims,b=sig_lims)

def ecomp_params(t_series,p_series):
    '''Compute the expected parameters (mu, sigma) for the euler DE

    Notes:
    1) Calculations based on the assumption of small time steps
    2) give time and data series as Pandas Series objects

    t_series = the time series corrisponding to p_series
    p_series = the data set which will be traced
    '''
    dt_inv = 1/np.mean((t_series-t_series.shift(1))[1:].values)
    prev_vals = p_series.shift(1)
    mu = dt_inv*np.mean(((p_series-prev_vals)/prev_vals)[1:])
    sigma = dt_inv*np.var(((p_series-prev_vals)/prev_vals)[1:])
    
    return mu,sigma

def acomp_params(t_series,p_series):
    '''Compute the expected parameters (mu, sigma) of the analytical solution

    Notes:
    1) Calculations do not require small time steps
    2) Give time and data series as Pandas Series objects

    t_series = The time series corrisponding to p_series
    p_series = The data set which will be traced
    '''
    dt_inv = 1/np.mean((t_series-t_series.shift(1))[1:].values)
    prev_vals = p_series.shift(1)
    lnR = np.log(p_series/prev_vals)
    sigma = dt_inv*np.var(lnR)
    mu = dt_inv*np.mean(lnR)+sigma**2/2

    return mu,sigma

def kscomp(mu_sigma,trc_func,t_series,p_series,num_bins,*faux_args):
    '''Compare a trace's histogramed fds to a normal distribution

    mu_sigma = a tuple of the parameters mu and sigma
    trc_func = Use etrace or atrace to act on p_series and t_series
    t_series = The time series corrisponding to p_series
    p_series = The data set which will be traced
    num_bins = The number of bins used to histogram the fds
    faux_args = Auxilary arguments for trc_func

    Returns
    -------
    ks_2samp: KS static and a p-value in a tuple
    mkr_trc: An array of bin positions for cnt_trc and cnt_norm
    cnt_trc: An array of bin values for trace's histogramed fds
    cnt_norm: An array of bin values for a scaled normal distribution
    '''

    mu,sigma = mu_sigma
    trc = trc_func('price',t_series,p_series,mu,sigma,*faux_args)
    fds_trc = np.array(ufuncs.get_fds(trc))
    cnt_trc,mkr_trc = np.histogram(fds_trc,num_bins)
    mkr_trc=mkr_trc[1:]
    std = np.sqrt(np.mean((t_series-t_series.shift(1))[1:].values))
    cnt_norm = normpdf(mkr_trc,0,std)
    cnt_norm = cnt_norm*max(cnt_trc)/max(cnt_norm)
    
    return ks_2samp(cnt_norm, cnt_trc),mkr_trc,cnt_trc,cnt_norm