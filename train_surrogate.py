# Seed value
# Apparently you may use different seed values at each stage
seed_value = 0
# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os

os.environ ['PYTHONHASHSEED'] = str ( seed_value )
# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf

tf.random.set_seed ( seed_value )
import random

# 2. Set the `python` built-in pseudo-random generator at a fixed value
random.seed ( seed_value )
# 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed ( seed_value )

import json
import time
from globals import *
import pandas as pd
import src.models as models
from datetime import datetime
from keras.optimizer_v2 import adam
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from dataprep.dataset import  load_reformated_csv, create_dataset
from keras.callbacks import ReduceLROnPlateau , EarlyStopping , ModelCheckpoint

from typing import List
from src.loss_analysis import loss_assessment_plots

def get_data_for_variable ( df ,
                            look_back: int = 1 ,
                            look_forward: int = 1 ,
                            variable: str = 'B:VIMIN' ,
                            train_split_size: float = 0.7 ) :
    dataset = df [variable].values  # numpy.ndarray
    dataset = dataset.astype ( 'float32' )
    dataset = np.reshape ( dataset , (-1 , 1) )
    scaler = MinMaxScaler ( feature_range = (0 , 1) )
    # scaler = RobustScaler()
    dataset = scaler.fit_transform ( dataset )

    ## TODO: Fix
    # print(len(dataset))
    train_size = int ( len ( dataset ) * train_split_size )
    # print(train_size)
    test_size = len ( dataset ) - train_size
    # print(test_size)

    train , test = dataset [0 :train_size , :] , dataset [train_size :len ( dataset ) , :]

    X_train , Y_train = create_dataset ( train ,
                                         look_back ,
                                         look_forward )
    # X_train = np.reshape ( X_train , (X_train.shape [0] , 1 , X_train.shape [1]) )
    X_train = np.reshape( X_train , (X_train.shape[0] , X_train.shape[1] , 1) )
    Y_train = np.reshape ( Y_train , (Y_train.shape [0] , Y_train.shape [1]) )
    # print(X_train.shape)
    # print(Y_train.shape)

    X_test , Y_test = create_dataset ( test ,
                                       look_back = look_back ,
                                       look_forward = look_forward )
    # X_test = np.reshape ( X_test , (X_test.shape [0] , 1 , X_test.shape [1]) )
    X_test = np.reshape( X_test , (X_test.shape[0], X_test.shape[1], 1))
    Y_test = np.reshape ( Y_test , (Y_test.shape [0] , Y_test.shape [1]) )
    # print(X_test.shape)
    # print(Y_test.shape)
    return scaler, X_train , Y_train , X_test , Y_test


def get_train_test_split ( variables ,
                           num_outputs ,
                           nsteps: int = 50000 ,
                           look_forward: int = 1 ,
                           look_back: int = 1 ,
                           concate_axis: int = 1
                           ) :
    # get the data in format acceptable to the surrogate [OLD FORMAT]
    # dataset = get_data ( filepath = DATA_FILE_PATH + "/" + DATA_FILE_NAME ,
    #                      nrows = nsteps )
    with open ( DATA_CONFIG ) as json_file :
        data_config = json.load ( json_file )
    df = load_reformated_csv ( filename = data_config [ 'data_dir' ] + data_config [ 'data_filename' ] ,
                                 nrows = nsteps )

    df = df.set_index ( pd.to_datetime ( df.time ) )
    df = df [VARIABLES]
    df = df.dropna ( )
    df = df.drop_duplicates ( )
    print ( len ( df ) )
    print ( df.keys ( ) )

    data_list = []
    x_train_list = []
    x_test_list = []
    for v in range ( len ( variables ) ) :
        data_list.append ( get_data_for_variable ( df = df ,
                                                   variable = variables [v] ,
                                                   look_forward = look_forward ,
                                                   look_back = look_back ,
                                                   train_split_size = 0.7 ) )
        x_train_list.append ( data_list [v] [1] )
        x_test_list.append ( data_list [v] [3] )

    ## Booster model data
    BoX_train = np.concatenate ( x_train_list , axis = concate_axis )
    BoY_train_tuple , BoY_test_tuple = [] , []
    for i in range ( num_outputs ) :
        BoY_train_tuple.append ( data_list [i] [2] )
        BoY_test_tuple.append ( data_list [i] [4] )

    BoY_train = np.concatenate ( BoY_train_tuple , axis = 1 )
    BoX_test = np.concatenate ( x_test_list , axis = concate_axis )
    BoY_test = np.concatenate ( BoY_test_tuple , axis = 1 )

    return data_list , BoX_train , BoY_train , BoX_test , BoY_test


def train ( nsteps: int = 50000 ,
            look_forward: int = 1 ,
            look_back: int = 1 ,
            loss: str = "mse" ,
            optimizer: str = "Adam" ,
            learning_rate: float = 1e-2 ,
            epochs: int = 100 ,
            batch_size: int = 99 ,
            num_outputs: int = 2 ,
            clipnorm: float = 1.0 ,
            clipvalue: float = 0.5 ,
            concate_axis: int = 1 ,
            variables: List [str] = ['B:VIMIN' , 'B:IMINER' , 'B:LINFRQ' , 'B:VIPHAS' , 'I:IB' , 'I:MDAT40' , 'I:MXIB']
            ) :
    # get the data in format acceptable to the surrogate
    data_list , BoX_train , BoY_train , BoX_test , BoY_test = get_train_test_split ( variables = variables ,
                                                                                     num_outputs = num_outputs ,
                                                                                     look_back = look_back ,
                                                                                     look_forward = look_forward ,
                                                                                     nsteps = nsteps ,
                                                                                     concate_axis = concate_axis )
    print ( BoX_train.shape )
    print ( BoY_train.shape )
    print ( BoX_test.shape )
    print ( BoY_test.shape )

    # start training the surrogate
    e = epochs
    bs = batch_size
    in_shape = (len ( variables ) , look_back)
    out_shape = num_outputs
    if concate_axis == 2 :
        # in_shape = (1 , len ( variables ) * look_back)
        in_shape = (look_back, len(variables))

    start_time = time.time ( )
    now = datetime.now ( )
    timestamp = now.strftime ( "D%m%d%Y-T%H%M%S" )
    print ( "date and time:" , timestamp )
    ##
    save_name = SURROGATE_FILE_NAME.format (
        LOOK_BACK, e , bs , int ( nsteps / 1000 ) , len ( VARIABLES ) , OUTPUTS , concate_axis , timestamp , SURROGATE_VERSION )

    # Callbacks
    tensorboard_callback = tf.keras.callbacks.TensorBoard ( log_dir = "./logs" )
    reduce_lr = ReduceLROnPlateau ( monitor = 'val_loss' , factor = 0.85 , patience = 5 , min_lr = 1e-6 , verbose = 1 )
    early_stopping = EarlyStopping ( monitor = 'val_loss' , min_delta = 0 , patience = 10 , verbose = 1 ,
                                     mode = 'auto' ,
                                     baseline = None , restore_best_weights = False )

    ## Model
    booster_model = models.build_lstm_model ( input_shape = in_shape ,
                                              output_shape = out_shape )
    opt = None
    if optimizer == "Adam" :
        opt = adam.Adam ( lr = learning_rate ,
                          clipnorm = clipnorm ,
                          clipvalue = clipvalue )
    booster_model.compile ( loss = loss ,
                            optimizer = opt ,
                            metrics = ['mse' , 'mape' , 'mae'] )
    booster_model.summary ( )

    ## Run multiple versions
    histories = []
    print ( "Running the training for {} splits".format ( N_SPLITS ) )
    kf = KFold ( n_splits = N_SPLITS ,
                 random_state = None ,
                 shuffle = False )
    k = 0
    mcp_name = save_name + '_kfold{}_'.format ( k )
    for train_index , val_index in kf.split ( BoX_train ) :
        ## Prep data
        x_train , x_val = BoX_train [train_index] , BoX_train [val_index]
        y_train , y_val = BoY_train [train_index] , BoY_train [val_index]
        print ( '########################' )
        print ( '### Running {} split. ###'.format ( k ) )
        print ( '### TrainX shape {}  ###'.format ( x_train.shape ) )
        print ( '### TrainY shape {}  ###'.format ( y_train.shape ) )
        print ( '### ValX shape {}  ###'.format ( x_val.shape ) )
        print ( '### ValY shape {}  ###'.format ( y_val.shape ) )
        print ( '########################' )
        ## Save best model callback

        mcp_save = ModelCheckpoint (
            filepath = SURROGATE_CKPT_DIR + "/" + mcp_name + '_e{epoch:02d}_vl{val_loss:.5f}.h5' ,
            period = CKPT_FREQ
        )
        ## Run model
        history = booster_model.fit ( x_train , y_train ,
                                      epochs = e ,
                                      batch_size = bs ,
                                      validation_data = (x_val , y_val) ,
                                      callbacks = [reduce_lr ,
                                                   early_stopping ,
                                                   mcp_save] ,
                                      verbose = 2 ,
                                      shuffle = True )
        histories.append ( history )
        k += 1
        mcp_name = save_name + '_kfold{}_'.format ( k )
        print ( 'Current training time: {}'.format ( time.time ( ) - start_time ) )

    print ( 'Total training time: {}'.format ( time.time ( ) - start_time ) )
    booster_model.save ( SURROGATE_DIR + "/" + mcp_name + '_final.h5' )

    # Save the file name for the latest, trained surrogate booster model in a global variable to be used in agent's training env
    # SURROGATE_MODELS_LIST.append ( SURROGATE_FILE_PATH + "/" + mcp_name + '_final.h5' )

    metrics = (booster_model.metrics_names)
    scores = booster_model.evaluate ( BoX_test ,
                                      BoY_test ,
                                      verbose = 1 )
    for i in range ( len ( metrics ) ) :
        print ( 'Evaluation on test set for {} metric = {}'.format ( metrics [i] , scores [i] ) )

    loss_assessment_plots ( booster_model ,
                            X = BoX_test ,
                            Y = BoY_test ,
                            training_history = histories ,
                            data = data_list ,
                            model_file_name = save_name ,
                            save_plot_name = mcp_name ,
                            concate_axis = concate_axis )

    return None