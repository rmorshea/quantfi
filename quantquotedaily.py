import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

base_dir = os.environ.get('QUANT_QUOTE_DAILY')

def get_symbols():
    """Get a list of all available symbols."""
    files = glob.glob(os.path.join(base_dir,'*'))
    syms = [os.path.splitext(os.path.split(f)[1])[0].split('_')[1] for f in files]
    return syms

def get_file(symbol,date):
    """Get the absolute filename for a symbols data."""
    symbol = symbol.lower()
    file = 'allstocks_' + date + '/table_' + symbol + '.csv'
    file = os.path.join(base_dir, file)
    if not os.path.isfile(file):
        raise IOError("File doesn't exist: %r" % file)
    return file

def get_daily_data(symbol,date,ranged=False):
    """Get a Pandas DataFrame with the daily data for the symbol.

    --- arguments ---
    symbol = ticker symbol
    date = give the date stamp(s) for the data 'YYYYmmdd' (+ ',YYYYmmdd')
    ranged = if date gives a range of dates: True, else: False
    
    --- Note ---
    If ranged is True a list of dates without data is also given."""
    dateparse = lambda x: datetime.strptime(x, '%Y%m%d %H%M')
    nodat=[]
    if ranged==True:
        all_dat = []
        date = date.split(',')
        enddate = date[1]
        crntdate = datetime.strptime(date[0], '%Y%m%d')
        check_end_date = datetime.strptime(date[1], '%Y%m%d')
        stringdate = crntdate.strftime('%Y%m%d')
        while stringdate!=enddate:
            try:
                f = get_file(symbol,stringdate)
                df = pd.read_csv(f, header=None, 
                             parse_dates={'datetime':[0,1]},
                             date_parser=dateparse,
                             index_col=0)
                all_dat.append(df)
            except IOError:
                nodat.append(stringdate)
            crntdate += timedelta(days=1)
            stringdate = crntdate.strftime('%Y%m%d')
        final_df = pd.concat(all_dat)
        final_df.columns = ['OPEN','HIGH','LOW','CLOSE','VOLUME','SPLITS','EARNINGS','DIVIDENDS']
    else:
        f = get_file(symbol,date)
        final_df = pd.read_csv(f, header=None,
                               parse_dates={'datetime':[0,1]},
                               date_parser=dateparse,
                               index_col=0)
        final_df.columns = ['OPEN','HIGH','LOW','CLOSE','VOLUME','SPLITS','EARNINGS','DIVIDENDS']
    return final_df,nodat

def calc_returns(df):
    close = df.close
    prev_close = df.close.shift(1)
    df['r'] = (close - prev_close)/prev_close
    df['lnr'] = np.log(close/prev_close)

_how = {'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'}

def resample(df, period):
    return df.resample(period, closed='right', how=_how, label='right')
