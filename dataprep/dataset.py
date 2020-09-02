import pandas as pd
import h5py
import numpy as np
import datetime
from functools import reduce
from sklearn.preprocessing import MinMaxScaler

## TODO: Another ugly hack
look_back=150
look_forward=1

def reformat_data(filename, data_type='h5'):
    '''
    Description:
        Method used to reformat the original h5 files a common resampled time index.
        This allows for easier data processing within the TF2 Dataset tools
    :param filename: the name of the RAW h5 file for the ACNET ParamData
    :param data_type: output data type (h5 or csv)
    :return: dictionary with method status
    '''
    status = {'Status': 'OK'}
    if data_type!='h5' and data_type!='csv':
        return {'Status': 'Failed', 'Message': '"{}" is not a valid output data file type.'.format(data_type)}

    # Find keys ##
    f = h5py.File(filename, 'r')
    h5_keys = list(f.keys())
    print('keys:{}'.format(h5_keys))
    print(len(h5_keys))

    ## Read data and reformat ##
    valid_dfs = []
    valid_keys = []
    for i,key in enumerate(h5_keys):
        df = pd.read_hdf(filename,key)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(axis=0)
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
    df_merged = df_merged.reset_index().set_index('time').resample('66ms').mean()
    df_merged = df_merged.replace([np.inf, -np.inf], np.nan)
    df_merged = df_merged.dropna(axis=0)

    if data_type=='h5':
        df_merged.to_hdf(filename+'_processed.h5', 'ACNET')

    if data_type=='csv':
        ## Regular CSV
        df_merged.to_csv(filename + '_processed.csv')
        ## Compressed CSV
        df_merged.to_csv(filename + '_processed.csv.gz', compression='gzip')

    return status

def load_reformated_cvs(filename,nrows=100000):
    df = pd.read_csv(filename,nrows=nrows)
    df=df.replace([np.inf, -np.inf], np.nan)
    df=df.dropna(axis=0)
    return df

def load_reformated_hdf5(filename):
    df = pd.read_hdf(filename,'ACNET')
    print(df.columns)
    return df

def create_dataset(dataset, look_back=1, look_forward=1):
    '''
     Description:
         Method use to create a single trace for LSTM model
         This allows for easier data processing within the TF2 Dataset tools
     :param dataset: pandas dataframe with variable
     :param look_back: number of time step before prediction
     :param look_forward: number of time step to prediction
     :return: two numpy array (input,output)
     '''
    X, Y = [], []
    offset = look_back + look_forward
    for i in range(len(dataset) - (offset + 1)):
        xx = dataset[i:(i + look_back), 0]
        yy = dataset[(i + look_back):(i + offset), 0]
        X.append(xx)
        Y.append(yy)
    return np.array(X), np.array(Y)


def get_dataset(dataframe, variable='B:VIMIN', split_fraction=0.8,concate_axis=1):
    '''
     Description:
         Method that scales the data and split into train/test datasets
     :param variable: desired variable
     :param dataframe: pandas dataframe
     :param split_fraction: desired split fraction between train and test
     :return: scaler, (x-train,y-train), (x-test,y-test)
    '''
    dataset = dataframe[variable].values
    dataset = dataset.astype('float32')
    dataset = np.reshape(dataset, (-1, 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    ## TODO: Fix
    #print(len(dataset))
    train_size = int(len(dataset) * split_fraction)
    #print(train_size)
    test_size = len(dataset) - train_size
    #print(test_size)

    ## Split dataset
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    ## Create train dataset
    X_train, Y_train = create_dataset(train, look_back, look_forward)
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    Y_train = np.reshape(Y_train, (Y_train.shape[0], Y_train.shape[1]))

    ## Create test dataset
    X_test, Y_test = create_dataset(test, look_back, look_forward)
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    Y_test = np.reshape(Y_test, (Y_test.shape[0], Y_test.shape[1]))
    #print(X_test.shape)
    #print(Y_test.shape)
    return scaler, X_train, Y_train, X_test, Y_test


def get_datasets(dataframe,variables = ['B:VIMIN', 'B:IMINER', 'B:LINFRQ', 'I:IB', 'I:MDAT40'],split_fraction=0.8,concate_axis=1):
    data_list = []
    scalers = []
    for v in range(len(variables)):
        data_list.append(get_dataset(dataframe,variable=variables[v],split_fraction=split_fraction))
    ## TODO: Horrible hack that should quickly be fixed
    scalers = [data_list[0][0],data_list[1][0],data_list[2][0],data_list[3][0],data_list[4][0]]
    X_train = np.concatenate((data_list[0][1], data_list[1][1], data_list[2][1], data_list[3][1], data_list[4][1]), axis=concate_axis)
    Y_train = np.concatenate((data_list[0][2], data_list[1][2], data_list[2][2], data_list[3][2], data_list[4][2]), axis=1)
    X_test = np.concatenate((data_list[0][3], data_list[1][3], data_list[2][3], data_list[3][3], data_list[4][3]), axis=concate_axis)
    Y_test = np.concatenate((data_list[0][4], data_list[1][4], data_list[2][4], data_list[3][4], data_list[4][4]), axis=1)
    return scalers,X_train,Y_train,X_test,Y_test
