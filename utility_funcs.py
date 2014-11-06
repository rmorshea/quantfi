def get_fds(w,lim=None):
    '''returns a finite difference series based on the input data
    
    w = input data series
    lim = returned differences are between +/-lim.'''
    return [w[i+1]-w[i] for i in range(len(w)-1) if lim==None or abs(w[i+1]-w[i])<lim]

def get_tseries(dfts):
    '''Get a list of times based on a series of Pandas TimeStamp objects

    The resulting times are given in years'''
    return [(ts-dfts[0]).total_seconds()/3.15569e7 for ts in dfts]