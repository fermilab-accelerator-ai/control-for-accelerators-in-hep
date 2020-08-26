import random, gym, os
from gym import spaces
from gym.utils import seeding
import numpy as np
import pandas as pd
import dataprep.dataset as dp
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RL-Logger')
logger.setLevel(logging.INFO)

np.seterr(divide='ignore', invalid='ignore')
        
class Surrogate_Accelerator(gym.Env):
  def __init__(self):

    self.episodes = 0
    self.steps= 0
    ## Define boundary ##
    #self.min_BIMIN = 103.3
    #self.max_BIMIN = 103.4
    #self.max_IMINER = 1
    self.variables = ['B:VIMIN', 'B:IMINER', 'B:LINFRQ', 'I:IB', 'I:MDAT40']
    self.nbatches = 0
    self.nsamples = 0

    ## Load surrogate model ##
    self.surrogate_model = keras.models.load_model('../surrogate_models/hep_accelerator_5var_08124020_v1.h5')
    self.surrogate_model.summary()

    ## Load data to initilize the env ##
    filename = 'MLParamData_1583906408.4261804_From_MLrn_2020-03-10+00_00_00_to_2020-03-11+00_00_00.h5_processed.csv.gz'
    data = dp.load_reformated_cvs('../data/' + filename)
    self.scalers, self.X_train, self.Y_train, _ , _ = dp.get_datasets(data,self.variables)
    self.nbatches = self.X_train.shape[0]
    self.nsamples = self.X_train.shape[2]
    logger.info('X_train.shape:{}'.format(self.X_train.shape))

    #self.low_state = np.array(
    #  [self.min_BIMIN,-self.max_IMINER], dtype=np.float64
    #)
    
    #self.high_state = np.array(
    #  [self.max_BIMIN, self.max_IMINER], dtype=np.float64
    #)
    
    self.observation_space = spaces.Box(
      low   = 0,
      high  = +1,
      shape = (5,),
      dtype = np.float64
    )

    self.actionMap_VIMIN = [0, 0.0001, 0.001,  0.01, -0.0001,-0.001, -0.01]
    self.action_space = spaces.Discrete(7)
    self.VIMIN = 0
    ##
    self.state = np.zeros(5)## shape=(1,1,750)
    self.predicted_state = np.zeros(shape=(1,1,5))
    print('Init pred shape:{}'.format(self.predicted_state.shape))
    self.reset

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]
  
  def step(self,action):
    self.steps += 1
    ## Calculate the new B:VINMIN based on policy action
    delta_VIMIN = self.actionMap_VIMIN[action]
    self.VIMIN += delta_VIMIN

    ## Update the B:VIMIN based on the action for  the in the predicted state
    print('Step() predicted_state:{}'.format(self.predicted_state))
    print('Step() predicted_state shape{}'.format(self.predicted_state.shape))
    print('Step() predicted_state reshaped:{}'.format(self.predicted_state))
    self.predicted_state[0,0,0] =  self.VIMIN
    #print(self.predicted_state.shape)
    print('Step() modified predicted_state:{}'.format(self.predicted_state))

    ## Shift trace by removing the oldest step and adding the new prediction.
    start_trace =0
    end_trace   =0
    for i in range(len(self.variables)):
        length = int(self.nsamples/len(self.variables))
        end_trace   = start_trace+length
        print('Step() start/stop/length of trace: {}/{}/{}'.format(start_trace,end_trace,length))
        self.state[0, 0, start_trace:end_trace-1] = self.state[0,0,start_trace+1:end_trace]
        print('Step() replace:{}'.format(self.predicted_state[0,0,i]))
        self.state[0, 0, end_trace-1:end_trace] = self.predicted_state[0,0,i]
        start_trace = end_trace

    print('Step state:{}'.format(self.state.shape))
    #print(self.state[0, 0,-2:-1])

    ## Predict new state
    self.predicted_state = self.surrogate_model.predict(self.state)
    print('SM predicted_state shape{}'.format(self.predicted_state.shape))
    self.predicted_state = self.predicted_state.reshape(1,1,5)
    print('Step() model predicted_state:{}'.format(self.predicted_state))
    iminer = self.predicted_state[0][0][1]
    print('norm iminer:{}'.format(iminer))
    iminer = self.scalers[1].inverse_transform(np.array([iminer]).reshape(1,-1))
    print('iminer:{}'.format(iminer))
    reward = -abs(iminer)
    done = bool(abs(iminer) >= 10)
    ##self.render()

    return self.predicted_state.flatten(), reward.flatten(), done, {}
  
  def reset(self):
    self.episodes += 1
    self.steps = 0
    ## Prepare the random sample ##
    this_batch = np.random.randint(0, high=self.nbatches)
    reset_data = self.X_train[this_batch]
    logger.info('reset_data.shape:{}'.format(reset_data.shape))
    self.state = reset_data.flatten()
    self.VIMIN = self.state[int(self.nsamples/len(self.variables))]
    logger.info('Normed VIMIN:{}'.format(self.VIMIN))
    logger.info('B:VIMIN:{}'.format(self.scalers[0].inverse_transform(np.array([self.VIMIN]).reshape(1,-1))))
    self.state = self.state.reshape(1,1,-1)
    logger.info('New state shape: {}'.format(self.state.shape))
    ## Load latest state
    start_trace = 0
    end_trace = 0
    for i in range(len(self.variables)):
        start_trace = end_trace
        end_trace   = start_trace+int(self.nsamples/len(self.variables))-1
        self.predicted_state[0, 0, i] = self.state[0, 0, end_trace - 1:end_trace]
    print('Reset newest states{}'.format(self.predicted_state))
    return self.predicted_state.flatten()

  def render(self):
    '''
    :return:
    '''
    print('render()')
    import seaborn as sns
    sns.set_style("ticks")
    fig, axs = plt.subplots(len(self.variables), figsize=(8, 12))
    start_trace =0
    end_trace   =0
    for v in range(len(self.variables)):
      start_trace = end_trace
      end_trace = start_trace + int(self.nsamples / len(self.variables)) - 1
      print(self.variables[v])
      utrace = self.state[0,0,start_trace:end_trace]
      #print('utrace:\n {}'.format(utrace))
      trace = self.scalers[v].inverse_transform(utrace.reshape(-1,1))
      #print('trace:\n {}'.format(trace))
      axs[v].plot(trace)
      #axs[v].legend(title=self.variables[v])

    #plt.show()
    #print(os.getcwd() )
    plt.savefig('../render/episode{}_step{}_v1.png'.format(self.episodes,self.steps))
    #plt.close()