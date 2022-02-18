#!/usr/bin/python3
import sys; print("Python", sys.version)
import matplotlib.pyplot as plt
import warnings
from src.analysis import plot_test
import numpy as np; print("NumPy", np.__version__)
import seaborn as sns; print("Seaborn", sns.__version__)
import pandas as pd; print("Pandas", pd.__version__)
from globals import *
# Config For Matplotlib
warnings.filterwarnings ("ignore")
plt.rcParams ['axes.titlesize'] = 18
plt.rcParams ['axes.titleweight'] = 'bold'
plt.rcParams ['axes.labelsize'] = 18
plt.rcParams ['axes.labelweight'] = 'regular'
plt.rcParams ['xtick.labelsize'] = 14
plt.rcParams ['ytick.labelsize'] = 14
plt.rcParams ['font.family'] = [u'serif']
plt.rcParams ['font.size'] = 14

def unscale(var_name,
            tseries,
            scale_dict):
  from_model = np.asarray(tseries)
  update = from_model*scale_dict[str(var_name)]["range"] + scale_dict[str(var_name)]["median"]
  return(update)

def loss_assessment_plots ( booster_model ,
                            X ,
                            Y ,
                            training_history ,
                            data ,
                            model_file_name ,
                            save_plot_name: str = "" ,
                            concate_axis: int = 1,
                            save_plots: bool = True) :
    loss_trace = []
    vloss_trace = []
    for k in range ( N_SPLITS ) :
        fold_histories = np.array ( training_history [k].history ['loss'] )
        print ( fold_histories.shape )
        loss_trace.append ( fold_histories )
        vloss_trace.append ( np.array ( training_history [k].history ['val_loss'] ) )
    full_loss_trace = np.concatenate ( loss_trace )
    full_vloss_trace = np.concatenate ( vloss_trace )
    print ( full_loss_trace.shape )
    plt.figure ( figsize = (12 , 10) )
    plt.plot ( full_loss_trace , label = 'loss' )
    plt.plot ( full_vloss_trace , label = 'val_loss' )
    plt.title ( 'model loss' )
    plt.ylabel ( 'loss' )
    plt.xlabel ( 'epochs' )
    plt.legend ( loc = 'upper right' )
    plt.yscale ( 'log' )
    if save_plots:
        plt.savefig ( '{}.png'.format (
            (PLOTS_DIR_FOR_SURROGATE + "/loss_{}_version_").format ( save_plot_name , SURROGATE_VERSION ) ) )

    plot_test ( booster_model ,
                    X ,
                    Y ,
                    nvar = 2 ,
                    name = (PLOTS_DIR_FOR_SURROGATE + "/test_{}_version_").format ( save_plot_name ,
                                                                                    SURROGATE_VERSION ) ,
                    start = 0 ,
                    end = 5000 )

    # plot predictions
    variables = ['B:VIMIN' , 'B:IMINER']
    nvar , start , end = len(variables) , 0 , 1000
    x = np.linspace ( start , end , int ( end - start ) )
    Y_predict = booster_model.predict ( X [start :end , : , :] )
    fig , axs = plt.subplots ( nvar , figsize = (16 , 16) )
    for v in range ( nvar ) :
        # data [v] [0] is the scaler dict used during preprocessing the data prior to training.
        Y_test_var1 = data [v] [0].inverse_transform ( Y [start :end , v].reshape ( -1 , 1 ) )
        Y_predict_var1 = data [v] [0].inverse_transform ( Y_predict [: , v].reshape ( -1 , 1 ) )
        # axs[v].plot(Y_test_var1,Y_predict_var1,'o')
        mape = 100 * abs ( Y_test_var1 - Y_predict_var1 ) / Y_test_var1
        print ( x.shape )
        print ( mape.shape )
        mape = mape.reshape ( -1 , )
        print ( mape.shape )
        print ( 'mape ave:{}'.format ( mape.mean ( ) ) )
        axs [v].plot ( Y_test_var1 , label = 'Data' )
        axs [v].plot ( Y_predict_var1 , label = 'Digital Twin' )
        axs [v].set_ylabel ( variables [v] )
        axs [v].set_xlabel ( 'Time samples' )
        axs [v].legend ( )
    if save_plots:
        plt.savefig ( (PLOTS_DIR_FOR_SURROGATE + "/" + save_plot_name + '_prediction_final_version_{}.png').format (
            SURROGATE_VERSION ) )

    fig , axs = plt.subplots( 1 , figsize = (12 , 12) )
    x_test = X # as BoX_test is input to the function as X
    y_test = Y # as BoY_test is input as Y
    start = 0
    end = X.shape[0]
    Y_predict = booster_model.predict( x_test[start :end , : , :] )

    Y_test_var0 = data[0][0].inverse_transform(y_test[start :end , 0].reshape( -1 , 1 ))
    Y_test_var1 = data[1][0].inverse_transform( y_test[start :end , 1].reshape( -1 , 1 ) )
    Y_predict_var0 = data[0][0].inverse_transform(Y_predict[start:end, 0 ].reshape(-1,1))
    Y_predict_var1 = data[1][0].inverse_transform( Y_predict[start :end , 1].reshape( -1 , 1 ) )

    x_bvimin = data[0][0].inverse_transform(x_test[start:end, 0 , -1].reshape(-1,1))
    x_biminer = data[1][0].inverse_transform(x_test[start:end, 1 , -1].reshape(-1,1))
    x_b_vimin = data[2][0].inverse_transform(x_test[start :end , 2 , -1].reshape( -1 , 1 ))
    x_alpha = 0.01 * data[5][0].inverse_transform(x_test[start :end , 5 , -1].reshape( -1 , 1 ))
    x_gamma = 0.00001 * data[4][0].inverse_transform(x_test[start :end , 5 , -1].reshape( -1 , 1 ))

    beta = [0]
    for i in range( len( x_b_vimin ) ) :
        if i > 0 :
            beta_t = beta[-1] + x_gamma[i] * x_biminer[i]
            beta.append( beta_t[0] )
    beta = np.asarray( beta ).reshape( -1 , 1 )
    BVIMIN_rach = x_b_vimin - 1 * x_alpha * x_biminer - 1 * beta  # predict the next, shiftting happens in the plotting
    BIMINER_rach = 10 * np.add( x_b_vimin , -1 * BVIMIN_rach )
    predicted = np.concatenate(
        (Y_test_var0 , Y_test_var1 , Y_predict_var0 , Y_predict_var1 , BVIMIN_rach , BIMINER_rach) ,
        axis = 1 )
    #   print(predicted.shape)
    df_cool = pd.DataFrame( predicted , columns = ['data B:VIMIN' , 'data B:IMINER' , 'pred B:VIMIN' , 'pred B:IMINER' ,
                                                   'PID B:VIMIN' , 'PID B:IMINER'] )

    sns.scatterplot( data = df_cool , x = "data B:VIMIN" , y = "data B:IMINER" , label = 'Data' )  # , hue="time")
    sns.scatterplot( data = df_cool , x = "pred B:VIMIN" , y = "pred B:IMINER" ,
                     label = 'Digital Twin' )  # , hue="time")

    if save_plots:
        plt.savefig (
            (PLOTS_DIR_FOR_SURROGATE + "/" + model_file_name + '_corr_final_version_{}.png').format ( SURROGATE_VERSION ) )