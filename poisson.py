import numpy as np
from random import random
from random import gauss

def cmpd_instpoiss(n,dt,lambd):
    '''Return a compound Poisson process with its corrisponding time series.
    
    The likelihood of an event is based on time since last event.
    In other words events are determined instantaneously (at the current time).
    
    n = the number data points
    dt = the size the time steps taken
    lambd = an intenisty parameter controling an exponential distribution.'''
    t_e = 0
    p_t = 0
    t_series = []
    p_series = []
    for i in range(n):
        t_e+=dt
        if np.e**(-lambd*t_e)<random():
            t_series.append(i*dt)
            p_series.append(p_t)
            p_t+=gauss(0,1*np.sqrt(dt))
            t_e=0
        t_series.append(i*dt)
        p_series.append(p_t)
    t_series.append(n*dt)
    p_series.append(p_t)
    return t_series,p_series

def cmpd_predpoiss(t,lambd):
    '''Return a compound Poisson process with its corrisponding time series.
    
    Time between events is drawn from an exponential distribution.
    In other words future events are predetermined at the time of the last event.
    
    t = upper limit on t
    dt = the size of the time steps taken
    lambd = an intenisty parameter controling an exponential distribution.'''
    p_t = 0
    t_e = 0
    t_series = []
    p_series = []
    while t_e<t:
        t_series.append(t_e)
        p_series.append(p_t)
        dte = np.log(random())/-lambd
        t_e += dte
        t_series.append(t_e)
        p_series.append(p_t)
        p_t += gauss(0,1*np.sqrt(dte))
    t_series[-1] = t
    return t_series,p_series