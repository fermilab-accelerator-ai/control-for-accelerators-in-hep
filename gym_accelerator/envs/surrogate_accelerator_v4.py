import gym
#import os
from gym import spaces
from gym.utils import seeding
import pandas as pd

# Framework class
import os
import sys
cwd = os.getcwd()
new = 'C:/Users/dkafkes/Desktop/fermi/accelerator-reinforcement-learning/control-for-accelerators-in-hep/dataprep'

sys.path.append(new)
os.chdir(sys.path[-1])
print(os.getcwd())
import dataset as dp
os.chdir(cwd)

from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RL-Logger')
logger.setLevel(logging.INFO)

np.seterr(divide='ignore', invalid='ignore')


def create_dataset(dataset, look_back=10 * 15, look_forward=1):
    X, Y = [], []
    offset = look_back + look_forward
    for i in range(len(dataset) - (offset + 1)):
        xx = dataset[i:(i + look_back), 0]
        yy = dataset[(i + look_back):(i + offset), 0]
        X.append(xx)
        Y.append(yy)
    return np.array(X), np.array(Y)


def get_dataset(df, variable='B:VIMIN'):
    dataset = df[variable].values
    dataset = dataset.astype('float32')
    dataset = np.reshape(dataset, (-1, 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    train_size = int(len(dataset) * 0.70)
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    X_train, Y_train = create_dataset(train, look_back=15)
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    Y_train = np.reshape(Y_train, (Y_train.shape[0], Y_train.shape[1]))

    # X_test, Y_test = create_dataset(test, look_back=15)
    # X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    # Y_test = np.reshape(Y_test, (Y_test.shape[0],  Y_test.shape[1]))

    return scaler, X_train, Y_train #, X_test, Y_test

# def data_distribution_plot(model, BoX_test, BoY_test, episode, step):
#   fig, axs = plt.subplots(1, figsize=(12,12))
#   x_test=BoX_test
#   y_test=BoY_test
#   start=0
#   end=BoX_test.shape[0]
#   Y_predict = model.predict(x_test[start:end,:,:])
#   Y_test_var0 = data_list[0][0].inverse_transform(y_test[start:end,0].reshape(-1,1)).reshape(-1,1)
#   Y_test_var1 = data_list[1][0].inverse_transform(y_test[start:end,1].reshape(-1,1)).reshape(-1,1)
#   Y_predict_var0 = data_list[0][0].inverse_transform(Y_predict[:,0].reshape(-1,1)).reshape(-1,1)
#   Y_predict_var1 = data_list[1][0].inverse_transform(Y_predict[:,1].reshape(-1,1)).reshape(-1,1)
#   np_predict = np.concatenate((Y_test_var0,Y_test_var1,Y_predict_var0,Y_predict_var1),axis=concate_axis) 
#   df_cool = pd.DataFrame(np_predict,columns=['data_va0','data_va1','pred_va0','pred_va1'])

#   sns.scatterplot(data=df_cool, x="data_va0", y="data_va1", label='Data')#, hue="time")
#   sns.scatterplot(data=df_cool, x="pred_va0", y="pred_va1", label='Digital Twin')#, hue="time")
#   #sns.scatterplot(data=df_cool, x="data_va1", y="pred_va1", label='Data')#, hue="time")
#   plt.savefig(os.path.join(directory), 'episode{}_step{}_corr_final.png'.format(episode, step))

class Surrogate_Accelerator_v4(gym.Env):
    def __init__(self):

        self.save_dir = os.getcwd() #'./'
        self.episodes = 0
        self.steps = 0
        self.max_steps = 100
        self.total_reward = 0
        self.data_total_reward = 0
        self.diff = 0

        # Define boundary
        self.min_BIMIN = 103.1
        self.max_BIMIN = 103.6
        # self.max_IMINER = 1
        # self.variables = ['B:VIMIN', 'B:IMINER', 'B:LINFRQ', 'I:IB', 'I:MDAT40']
        # self.nbatches = 0
        # self.nsamples = 0

        # Load surrogate models
        #https://github.com/keras-team/keras/issues/14040 needs it to be false
        self.booster_model = keras.models.load_model(
            '../surrogate_models/databricks models/2to2_1sec/fullbooster_noshift_e250_bs99_k_invar2_outvar2_axis1_mmscaler_t0_D11202020-T175745_kfold4__final.h5', compile = False)

            #fullbooster_noshift_e250_bs99_nsteps250k_invar5_outvar3_axis1_mmscaler_t0_D10122020'
            #'-T175237_kfold2__e16_vl0.00038.h5')
        # ("good") fullbooster_noshift_e250_bs99_nsteps250k_nvar5_axis1_mmscaler_t0_D10122020-T133515_kfold4__final.h5')
        # fullbooster_noshift_e250_bs99_nsteps250k_nvar7_axis1_mmscaler_t0_D10112020-T174516_kfold4__final.h5')
        # self.injector_model = keras.models.load_model(
        #     '../surrogate_models/model_injector_adam256_e350_bs99_nsteps250k.h5')

        # Load scalers

        # Load data to initialize the env
        filename = '310_11_more_params.csv'
        data = dp.load_reformated_cvs('../data/' + filename, nrows=250000)
        data['B:VIMIN'] = data['B:VIMIN'].shift(-1)
        data = data.set_index(pd.to_datetime(data.time))
        data = data.dropna()
        data = data.drop_duplicates()
        self.variables = ['B:VIMIN', 'B:IMINER'] #'B:LINFRQ', 'I:IB', 'I:MDAT40']
        # self.variables = ['B:VIMIN', 'B:IMINER', 'B:VIPHAS', 'B:LINFRQ', 'I:IB', 'I:MDAT40', 'I:MXIB']
        self.nvariables = len(self.variables)
        logger.info('Number of variables:{}'.format(self.nvariables))

        self.scalers = []
        data_list = []
        x_train = []
        # get_dataset also normalizes the data
        for v in range(len(self.variables)):
            data_list.append(get_dataset(data, variable=self.variables[v]))
            self.scalers.append(data_list[v][0])
            x_train.append(data_list[v][1])

        # Axis
        self.concate_axis = 1
        self.X_train = np.concatenate(x_train, axis=self.concate_axis)

        self.nbatches = self.X_train.shape[0]
        self.nsamples = self.X_train.shape[2]
        self.batch_id = 10  # np.random.randint(0, high=self.nbatches)
        self.data_state = None

        print('Data shape:{}'.format(self.X_train.shape))
        self.observation_space = spaces.Box(
            low=0,
            high=+1,
            shape=(self.nvariables,),
            dtype=np.float64
        )

        self.actionMap_VIMIN = [0, 0.0001, 0.005, 0.001, -0.0001, -0.005, -0.001]
        self.action_space = spaces.Discrete(7)
        self.VIMIN = 0
        ##
        self.state = np.zeros(shape=(1, self.nvariables, self.nsamples))
        self.predicted_state = np.zeros(shape=(1, self.nvariables, 1))
        logger.debug('Init pred shape:{}'.format(self.predicted_state.shape))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.steps += 1
        logger.info('Episode/State: {}/{}'.format(self.episodes, self.steps))
        done = False

        # Steps:
        # 1) Update VIMIN based on action
        # 2) Predict booster variables
        # 3) Predict next step for injector
        # 4) Shift state with new values

        # Step 1: Calculate the new B:VIMIN based on policy action
        logger.info('Step() before action VIMIN:{}'.format(self.VIMIN))
        delta_VIMIN = self.actionMap_VIMIN[int(action)]
        DENORN_BVIMIN = self.scalers[0].inverse_transform(np.array([self.VIMIN]).reshape(1, -1))
        DENORN_BVIMIN += delta_VIMIN
        logger.debug('Step() descaled VIMIN:{}'.format(DENORN_BVIMIN))
        if DENORN_BVIMIN < self.min_BIMIN or DENORN_BVIMIN > self.max_BIMIN:
            logger.info('Step() descaled VIMIN:{} is out of bounds.'.format(DENORN_BVIMIN))
            done = True

        self.VIMIN = self.scalers[0].transform(DENORN_BVIMIN)
        logger.debug('Step() updated VIMIN:{}'.format(self.VIMIN))
        self.state[0][0][self.nsamples - 1] = self.VIMIN

        # Step 2: Predict using booster model
        self.predicted_state = self.booster_model.predict(self.state)
        #print(self.predicted_state)
        self.predicted_state = self.predicted_state.reshape(1, 2, 1) #used to be 3 in the center #TODO: make dynamic

        # Step 3: Update IMINER and LINFQN
        # print(self.state.shape)
        # logger.debug('Step() state with pre-state model\n{}'.format(self.state[0,:,-2:]))
        # self.state[0, 1:3, -1:] = self.predicted_state[0,1:3]
        # logger.debug('Step() state with updated state model\n{}'.format(self.state[0,:,-2:]))
        # print(self.state.shape)

        # Predict the injector variables
        # injector_input = self.state[0,3:5,:].reshape(1,2,150)
        # injector_prediction = self.injector_model.predict(injector_input).reshape(1,2,1)
        #
        # logger.debug('Step() state with pre-injector state model\n{}'.format(self.state[0,:,-2:]))
        # logger.debug('Step() state with injector state model shape\n{}'.format(injector_prediction.shape))
        # #self.state[0, 3:5, -1:] = injector_prediction[0,:]
        # logger.debug('Step() state with updated injector state model\n{}'.format(self.state[0,:,-2:]))
        #
        # Step 4: Shift state by one step
        self.state[0, :, 0:-1] = self.state[0, :, 1:]

        # Update IMINER
        self.state[0][1][self.nsamples - 1] = self.predicted_state[0, 1:2]
        # ## Update injector variables
        # self.state[0, 3:5, -1:] = injector_prediction[0,:]

        # Update data state for rendering
        self.data_state = None
        self.data_state = np.copy(self.X_train[self.batch_id + self.steps].reshape(1, self.nvariables, self.nsamples))
        data_iminer = self.scalers[1].inverse_transform(self.data_state[0][1][self.nsamples - 1].reshape(1, -1))
        #where's data_vimin
        data_reward = -abs(data_iminer)
        #data_reward = np.exp(-2*np.abs(data_iminer))

        # Use data for everything but the B:IMINER prediction
        self.state[0, 2:self.nvariables, :] = self.data_state[0, 2:self.nvariables, :]

        iminer = self.predicted_state[0, 1]
        logger.debug('norm iminer:{}'.format(iminer))
        iminer = self.scalers[1].inverse_transform(np.array([iminer]).reshape(1, -1))
        logger.debug('iminer:{}'.format(iminer))

        # Reward
        reward = -abs(iminer)
        #reward2 = np.exp(-2*np.abs(iminer))

        if abs(iminer) >= 2:
            logger.info('iminer:{} is out of bounds'.format(iminer))
            done = True

        if done:
            penalty = 5 * (self.max_steps - self.steps)
            logger.info('penalty:{} is out of bounds'.format(penalty))
            reward -= penalty

        if self.steps >= int(self.max_steps):
            done = True

        self.diff += np.asscalar(abs(data_iminer - iminer))
        self.data_total_reward += np.asscalar(data_reward)
        self.total_reward += np.asscalar(reward)

        self.render()

        return self.state[0, :, -1:].flatten(), np.asscalar(reward), done, {}

    def reset(self):
        self.episodes += 1
        self.steps = 0
        self.data_total_reward = 0
        self.total_reward = 0
        self.diff = 0
        self.data_state = None

        # Prepare the random sample ##
        self.batch_id = 10
        # self.batch_id = np.random.randint(0, high=self.nbatches)
        logger.info('Resetting env')
        # self.state = np.zeros(shape=(1,5,150))
        logger.debug('self.state:{}'.format(self.state))
        self.state = None
        self.state = np.copy(self.X_train[self.batch_id].reshape(1, self.nvariables, self.nsamples))
        logger.debug('self.state:{}'.format(self.state))
        logger.debug('reset_data.shape:{}'.format(self.state.shape))
        self.VIMIN = self.state[0, 0, -1:]
        logger.debug('Normed VIMIN:{}'.format(self.VIMIN))
        logger.debug('B:VIMIN:{}'.format(self.scalers[0].inverse_transform(np.array([self.VIMIN]).reshape(1, -1))))
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
        #logger.debug('Save path:{}'.format(self.save_dir))

        #coming up against windows character limit
        #render_dir = os.path.join(self.save_dir, 'render')
        # render_dir = 'render'
        # print(render_dir)
        # logger.debug('Render path:{}'.format(render_dir))

        # print("Current working directory: ", os.getcwd())
        # cwd = os.getcwd()

        # if not os.path.exists(render_dir):
        #     os.mkdir(render_dir)
        
        #os.chdir(os.path.join(self.save_dir, render_dir))

        import seaborn as sns
        sns.set_style("ticks")
        nvars = 2  # len(self.variables)
        fig, axs = plt.subplots(nvars, figsize=(12, 8))
        logger.debug('self.state:{}'.format(self.state))
        for v in range(0, nvars):
            utrace = self.state[0, v, :]
            trace = self.scalers[v].inverse_transform(utrace.reshape(-1, 1))
            if v == 0:
                axs[v].set_title('Raw data reward: {:.2f} - RL agent reward: {:.2f} '.format(self.data_total_reward,
                                                                                             self.total_reward))
            axs[v].plot(trace, label='Digital twin', color='black')

            # if v==1:
            data_utrace = self.data_state[0, v, :]
            data_trace = self.scalers[v].inverse_transform(data_utrace.reshape(-1, 1))
            if v == 1:
                x = np.linspace(0, 14, 15) #np.linspace(0, 149, 150) #TODO: change this so that it is dynamic for lookback
                axs[v].fill_between(x, -data_trace.flatten(), +data_trace.flatten(), alpha=0.2, color='red')
            axs[v].plot(data_trace, 'r--', label='Data')
            axs[v].set_xlabel('time')
            axs[v].set_ylabel('{}'.format(self.variables[v]))
            axs[v].legend(loc='upper left')

        plt.savefig('episode{}_step{}_v1.png'.format(self.episodes, self.steps))
        plt.clf()

        fig, axs = plt.subplots(1, figsize=(12,12))

        # print(self.scalers[0])
        # print(self.scalers[1])
        # print(self.state[0][0])
        # print(self.state[1][0])

        Y_agent_bvimin = self.scalers[0].inverse_transform(self.state[0][0].reshape(-1,1)).reshape(-1,1) #[start:end,0]
        Y_agent_biminer = self.scalers[1].inverse_transform(self.state[0][1].reshape(-1,1)).reshape(-1,1) #[start:end,1]

        Y_data_bvimin = self.scalers[0].inverse_transform(self.data_state[0][0].reshape(-1,1)).reshape(-1,1) #[start:end,0]
        Y_data_biminer = self.scalers[1].inverse_transform(self.data_state[0][1].reshape(-1,1)).reshape(-1,1) #[start:end,1]

        np_predict = np.concatenate((Y_data_bvimin, Y_data_biminer, Y_agent_bvimin, Y_agent_biminer),axis=self.concate_axis) 
        df_cool = pd.DataFrame(np_predict, columns=['bvimin_data','biminer_data', 'bvimin_agent', 'biminer_agent'])
        sns.scatterplot(data=df_cool, x="bvimin_data", y="biminer_data", label='Data')#, hue="time")
        sns.scatterplot(data=df_cool, x="bvimin_agent", y="biminer_agent", label='Digital Twin')#, hue="time")
        #sns.scatterplot(data=df_cool, x="data_va1", y="pred_va1", label='Data')#, hue="time")
        plt.savefig('corr_episode{}_step{}.png'.format(self.episodes, self.steps))

        plt.close('all')
        os.chdir(cwd)
