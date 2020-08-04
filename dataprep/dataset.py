import pandas as pd
import h5py
import numpy as np
import datetime
from functools import reduce

def reformat_hdf5(filename):
    '''
    Description:
        Method used to reformat the original h5 files a common resampled time index.
        This allows for easier data processing within the TF2 Dataset tools
    :param filename: the name of the RAW h5 file for the ACNET ParamData
    :return: dictionary with method status (not used yet)
    '''
    status = {'Status': 'OK'}
    # Find keys ##
    f = h5py.File(filename, 'r')
    h5_keys = list(f.keys())
    print('keys:'.format(h5_keys))
    print(len(h5_keys))

    ## Read data and reformat ##
    valid_dfs = []
    valid_keys = []
    for i,key in enumerate(h5_keys):
        df = pd.read_hdf(filename,key)
        df_new          = pd.DataFrame()
        df_new['time']  = pd.to_datetime(df.utc_seconds,unit='s')
        df_new[key]     = pd.to_numeric(df.value)
        ## TODO: double check if using mean will cause any problems
        df_new          = df_new.reset_index().set_index('time').resample('66ms').mean()
        del df_new['index']
        valid_keys.append(key)
        valid_dfs.append(df_new)

    ## Print available variables ##
    print('valid_keys',valid_keys)

    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['time'],how='outer'), valid_dfs)
    df_merged.to_hdf(filename+'_processed.h5', 'ACNET')

    return status

def reformat_dataset(filename):
    '''
    Original data format converter for ML workflow ( ## DEPRECATED ##)
    '''
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
    ## Regular CSV
    df_merged.to_csv(filename+'_processed.csv')
    ## Compressed CSV
    df_merged.to_csv(filename+'_processed.csv.gz', compression='gzip')
    return status

def load_data(filename):
    df = pd.read_csv(filename)
    df=df.replace([np.inf, -np.inf], np.nan)
    df=df.dropna(axis=0)
    return df

def load_reformated_hdf5(filename):
    df = pd.read_hdf(filename,'ACNET')
    print(df.columns)
    return df
