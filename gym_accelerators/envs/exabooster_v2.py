import os
import gym
import json
import logging
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from gym import spaces
from gym.utils import seeding
from dataprep.dataset import load_reformated_csv, create_dataset
from globals import *

logging.basicConfig ( format = '%(asctime)s - %(levelname)s - %(message)s' )
logger = logging.getLogger ( 'RL-Logger' )
logger.setLevel ( logging.INFO )

np.seterr ( divide = 'ignore' , invalid = 'ignore' )

def get_dataset ( df ,
                  variable: str = 'B:VIMIN',
                  train_test_split: float = 0.7) :
    dataset = df [ variable ].values
    dataset = dataset.astype ( 'float32' )
    dataset = np.reshape ( dataset , (-1 , 1) )

    train_size = int ( len ( dataset ) * train_test_split )
    train , test = dataset [ 0 :train_size , : ] , dataset [ train_size :len ( dataset ) , : ]

    X_train , Y_train = create_dataset ( train ,
                                         look_back = LOOK_BACK,
                                         look_forward = LOOK_FORWARD)
    X_train = np.reshape ( X_train , (X_train.shape [ 0 ] , 1 , X_train.shape [ 1 ]) )
    Y_train = np.reshape ( Y_train , (Y_train.shape [ 0 ] , Y_train.shape [ 1 ]) )

    return X_train , Y_train

def all_inplace_scale ( df ) :
    scale_dict = {}

    for var in VARIABLES :
        our_data2 = df
        trace = our_data2 [ var ].astype ( 'float32' )
        data = np.array ( trace )

        median = np.median ( data )
        upper_quartile = np.percentile ( data , 75 )
        lower_quartile = np.percentile ( data , 25 )

        iqr = upper_quartile - lower_quartile
        lower_whisker = data [ data >= lower_quartile - 1.5 * iqr ].min ( )
        upper_whisker = data [ data <= upper_quartile + 1.5 * iqr ].max ( )

        ranged = upper_whisker - lower_whisker
        # (value − median) / (upper - lower)
        our_data2 [ var ] = 1 / ranged * (data - median)

        scale_dict [ str ( var ) ] = {"median" : median , "range" : ranged}

    return scale_dict


def unscale ( var_name , tseries , scale_dict ) :
    # equivalent to inverse transform
    from_model = np.asarray ( tseries )
    update = from_model * scale_dict [ str ( var_name ) ] [ "range" ] + scale_dict [ str ( var_name ) ] [ "median" ]

    return (update)


def rescale ( var_name , tseries , scale_dict ) :
    # equivalent to transform
    data = np.asarray ( tseries )
    update = 1 / scale_dict [ str ( var_name ) ] [ "range" ] * (data - scale_dict [ str ( var_name ) ] [ "median" ])

    return (update)

def create_dropout_predict_model ( model , dropout ) :
    conf = model.get_config ( )

    for layer in conf [ 'layers' ] :
        # Dropout layers
        if layer [ "class_name" ] == "Dropout" :
            # print(layer)
            layer [ "config" ] [ "rate" ] = dropout
    model_dropout = tf.keras.Model.from_config ( conf )
    model_dropout.set_weights ( model.get_weights ( ) )
    return model_dropout

class ExaBooster_v2(gym.Env):

    def __init__(self):
        self.episodes = 0
        self.steps = 0
        self.total_reward = 0
        self.data_total_reward = 0
        self.total_iminer = 0
        self.data_total_iminer = 0
        self.diff = 0
        self.max_steps = 100
        self.nactions = 15

        # Define boundary
        self.min_BIMIN = 103.1
        self.max_BIMIN = 103.6

        self.variables = VARIABLES
        self.nvariables = len(self.variables)
        logger.info('Number of variables:{}'.format(self.nvariables))

        # Load surrogate models
        model = keras.models.load_model (
            LATEST_SURROGATE_MODEL ,
            compile = False )

        self.booster_model = create_dropout_predict_model ( model , .2 )

        # Load data to initialize the env
        with open(DATA_CONFIG) as json_file:
            data_config = json.load(json_file)
        data = load_reformated_csv(filename = data_config['data_dir'] + data_config['data_filename'],
                                   nrows = NSTEPS)
        data['B:VIMIN'] = data['B:VIMIN'].shift(-1)
        data = data.set_index(pd.to_datetime(data.time))
        data = data.dropna()
        data = data.drop_duplicates()
        logger.info ( 'Number of samples:{}'.format ( data.shape ) )
        self.scalers = all_inplace_scale ( data )
        data_list = []
        x_train = []
        for v in range(len(self.variables)):
            data_list.append(get_dataset(data, variable=self.variables[v]))
            x_train.append(data_list[v][0])
        # Axis
        concate_axis = 1
        self.X_train = np.concatenate(x_train, axis=concate_axis)
        print( 'Data shape:{}'.format( self.X_train.shape ) )
        self.nbatches = self.X_train.shape[0]
        self.nsamples = self.X_train.shape[2]
        if IN_PLAY_MODE == False:
            self.batch_id = np.random.randint( low = 500 , high = 1100 )
        else:
            self.batch_id = 10
        self.data_state = None

        print('Data shape:{}'.format(self.X_train.shape))
        self.observation_space = spaces.Box(
            low=0,
            high=+1,
            shape=(self.nvariables,),
            dtype=np.float64
        )

        # Dynamically allocate
        data['B:VIMIN_DIFF'] = data['B:VIMIN'] - data['B:VIMIN'].shift(-1)
        self.action_space = spaces.Discrete(self.nactions)
        self.actionMap_VIMIN = []
        for i in range(1, self.nactions + 1):
            self.actionMap_VIMIN.append(data['B:VIMIN_DIFF'].quantile(i / (self.nactions + 1)))

        self.VIMIN = 0
        self.state = np.zeros(shape=(1, self.nvariables, self.nsamples))
        self.predicted_state = np.zeros(shape=(1, self.nvariables, 1))
        logger.debug('Init pred shape:{}'.format(self.predicted_state.shape))
        self.do_render = True  # True

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.steps += 1
        done = False

        # Steps:
        # 1) Update VIMIN based on action
        # 2) Predict booster dynamic variables
        # 3) Shift state with new values
        # 4) Update B:IMINER

        # Step 1: Calculate the new B:VINMIN based on policy action
        logger.info('Step() before action VIMIN:{}'.format(self.VIMIN))
        delta_VIMIN = self.actionMap_VIMIN[action]
        # DENORN_BVIMIN = self.scalers[0].inverse_transform(np.array([self.VIMIN]).reshape(1, -1))
        DENORN_BVIMIN = unscale ( self.variables [ 0 ] , np.array ( [ self.VIMIN ] ).reshape ( 1 , -1 ) ,
                        self.scalers )
        DENORN_BVIMIN += delta_VIMIN
        logger.debug('Step() descaled VIMIN:{}'.format(DENORN_BVIMIN))

        if DENORN_BVIMIN < self.min_BIMIN or DENORN_BVIMIN > self.max_BIMIN:
            logger.info('Step() descaled VIMIN:{} is out of bounds.'.format(DENORN_BVIMIN))
            done = True

        # self.VIMIN = self.scalers[0].transform(DENORN_BVIMIN)
        self.VIMIN = rescale ( self.variables [ 0 ] , DENORN_BVIMIN ,
                               self.scalers )
        logger.debug('Step() updated VIMIN:{}'.format(self.VIMIN))
        self.state[0][0][self.nsamples - 1] = self.VIMIN

        # Step 2: Predict using booster model
        self.predicted_state = self.booster_model.predict(self.state)
        self.predicted_state = self.predicted_state.reshape(1, OUTPUTS, 1)

        # Step 3: Shift state by one step
        self.state [ 0 , : , 0 :-1 ] = self.state [ 0 , : , 1 : ]  # shift forward

        # Step 4: Update IMINER
        self.state [ 0 ] [ 1 ] [ self.nsamples - 1 ] = self.predicted_state [ 0 , 1 :2 ]

        # Update data state for rendering
        self.data_state = np.copy(self.X_train[self.batch_id + self.steps].reshape(1, self.nvariables, self.nsamples))
        # data_iminer = self.scalers[1].inverse_transform(self.data_state[0][1][self.nsamples - 1].reshape(1, -1))
        data_iminer =  unscale ( self.variables [ 1 ] ,
                                self.data_state [ 0 ] [ 1 ] [ self.nsamples - 1 ].reshape ( 1 , -1 ) ,
                                self.scalers )
        data_reward = -abs(data_iminer)
        # data_reward = np.array(1. * math.exp(-5 * abs(np.asscalar(data_iminer))))

        # Use data for everything but the B:IMINER prediction
        self.state[0, 2:self.nvariables, :] = self.data_state[0, 2:self.nvariables, :]

        iminer = self.predicted_state[0, 1]
        logger.debug('norm iminer:{}'.format(iminer))
        # iminer = self.scalers[1].inverse_transform(np.array([iminer]).reshape(1, -1))
        iminer = unscale ( self.variables [ 1 ] , np.array ( [ iminer ] ) , self.scalers ).reshape ( 1 ,
                                                                                                        -1 )
        logger.debug('iminer:{}'.format(iminer))

        # Reward
        reward = -abs(iminer)
        # reward = np.array(-1 + 1. * math.exp(-5 * abs(np.asscalar(iminer))))
        # reward = np.array(1. * math.exp(-5 * abs(np.asscalar(iminer))))

        if abs(iminer) >= 2:
            logger.info('iminer:{} is out of bounds'.format(iminer))
            done = True
            penalty = 5 * (self.max_steps - self.steps)
            logger.info ( 'penalty:{} is out of bounds'.format ( penalty ) )
            reward -= penalty

        if self.steps >= int(self.max_steps):
            done = True

        self.diff += np.asscalar(abs(data_iminer - iminer))
        self.data_total_reward += np.asscalar(data_reward)
        self.total_reward += np.asscalar(reward)
        self.total_iminer  += np.asscalar(abs(iminer))
        self.data_total_iminer += np.asscalar(abs(data_iminer))

        if self.do_render:
            self.render()

        return self.state[0, :, -1:].flatten(), np.asscalar(reward), done, {}
        # return self.state[0, :, -1:].flatten()

    def reset(self):
        self.episodes += 1
        self.steps = 0
        self.data_total_reward = 0
        self.total_reward = 0
        self.total_iminer = 0
        self.data_total_iminer = 0
        self.diff = 0
        self.data_state = None

        # Prepare the random sample ##
        if IN_PLAY_MODE:
            self.batch_id = np.random.randint( low = 500 , high = 1100 )
        else:
            self.batch_id = 10
        logger.info('Resetting env')
        # self.state = np.zeros(shape=(1,5,150))
        logger.debug('self.state:{}'.format(self.state))
        self.state = None
        self.state = np.copy(self.X_train[self.batch_id].reshape(1, self.nvariables, self.nsamples))
        logger.debug('self.state:{}'.format(self.state))
        logger.debug('reset_data.shape:{}'.format(self.state.shape))
        # self.min_BIMIN = self.scalers[0].inverse_transform(self.state[:, 0, :]).min()
        # self.max_BIMIN = self.scalers[0].inverse_transform(self.state[:, 0, :]).max()
        self.min_BIMIN = unscale( self.variables[0] , self.state[: , 0 , :] , self.scalers ).min( )
        self.max_BIMIN = unscale( self.variables[0] , self.state[: , 0 , :] , self.scalers ).max( )
        logger.info('Lower and upper B:VIMIN: [{},{}]'.format(self.min_BIMIN, self.max_BIMIN))
        self.VIMIN = self.state[0, 0, -1:]
        logger.debug('Normed VIMIN:{}'.format(self.VIMIN))
        # logger.debug('B:VIMIN:{}'.format(self.scalers[0].inverse_transform(np.array([self.VIMIN]).reshape(1, -1))))
        logger.debug( 'B:VIMIN:{}'.format(
            unscale( self.variables[0] , np.array( [self.VIMIN] ) , self.scalers ).reshape( 1 ,
                                                                                               -1 ) ) )
        return self.state[0, :, -1:].flatten()

    def render(self):

        plt.rcParams['axes.titlesize'] = 18
        plt.rcParams['axes.titleweight'] = 'bold'
        plt.rcParams['axes.labelsize'] = 18
        plt.rcParams['axes.labelweight'] = 'regular'
        plt.rcParams['xtick.labelsize'] = 14
        plt.rcParams['ytick.labelsize'] = 14
        plt.rcParams['font.family'] = [u'serif']
        plt.rcParams['font.size'] = 14
        plt.rcParams['font.family'] = [u'serif']
        plt.rcParams['font.size'] = 16

        logger.debug('render()')
        sns.set_style("ticks")
        nvars = 2  # len(self.variables)
        fig, axs = plt.subplots(nvars, figsize=(12, 8))
        logger.debug('self.state:{}'.format(self.state))
        for v in range(0, nvars):
            utrace = self.state[0, v, :]
            # trace = self.scalers[v].inverse_transform(utrace.reshape(-1, 1))
            trace = unscale( self.variables[v] , utrace.reshape( -1 , 1 ) ,
                                  self.scalers )
            if v == 0:
                iminer_imp = 0
                if self.total_iminer > 0:
                    iminer_imp = self.data_total_iminer / self.total_iminer
                axs[v].set_title('Raw data reward: {:.2f} - RL agent reward: {:.2f} - Improvement: {:.2f} '.format(self.data_total_reward,
                                                                                                                   self.total_reward, iminer_imp))
            axs[v].plot(trace, label='Digital twin', color='black')

            # if v==1:
            data_utrace = self.data_state[0, v, :]
            data_trace =unscale( self.variables[v] , data_utrace.reshape( -1 , 1 ) ,
                                  self.scalers )
            # data_trace = self.scalers[v].inverse_transform(data_utrace.reshape(-1, 1))
            if v == 1:
                x = np.linspace(0, LOOK_BACK-1 , LOOK_BACK)
                axs[v].fill_between(x, -data_trace.flatten(), +data_trace.flatten(), alpha=0.2, color='red')
            axs[v].plot(data_trace, 'r--', label='Data')
            axs[v].set_xlabel('time')
            axs[v].set_ylabel('{}'.format(self.variables[v]))
            axs[v].legend(loc='upper left')
        plt.savefig ( EPISODES_PLOTS_DIR + '/' + 'arch_{}_dqn_int{}_out{}_env_EXB_{}_train_episode_{}_step_{}.png'.format ( ARCH_TYPE, nvars, OUTPUTS,ENV_VERSION,self.episodes , self.steps ) )
        plt.close('all')