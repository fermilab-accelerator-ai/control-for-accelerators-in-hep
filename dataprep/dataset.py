import pandas as pd
import h5py
import numpy as np
import datetime
from functools import reduce


def reformat_dataset(filename):
    status = {'Status': 'OK'}
    
    # Find keys ##
    f= h5py.File(filename, 'r')   
    h5_keys = list(f.keys())
    print(h5_keys)
    print(len(h5_keys))
    f.close()

    ## Read data and reformat ##
    valid_dfs = []
    valid_keys = []
    for i,key in enumerate(h5_keys):
        df = pd.read_hdf(filename,key)
        mean = df.value.mean()
        std = df.value.std()
        if abs(mean)!=0:
            if std/abs(mean)>1e-5:
                df_new          = pd.DataFrame()
                df_new['time']  = pd.to_datetime(df.utc_seconds,unit='s')
                df_new[key]     = pd.to_numeric(df.value)
                df_new          = df_new.reset_index().set_index('time').resample('66ms').mean()
                del df_new['index']
                valid_keys.append(key)
                valid_dfs.append(df_new)
    print('valid_keys',valid_keys)

    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['time'],
                        how='outer'), valid_dfs)
    df_merged.to_csv(filename+'_processed.cvs')
    return status
