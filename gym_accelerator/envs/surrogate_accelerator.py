import random, gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pandas as pd
import dataprep.dataset as dp
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RL-Logger')
logger.setLevel(logging.INFO)

np.seterr(divide='ignore', invalid='ignore')
        
class Surrogate_Accelerator(gym.Env):
  def __init__(self):

    ## Define boundary ##
    self.min_BIMIN = 103.3
    self.max_BIMIN = 103.4
    self.max_IMINER = 1
    self.variables = ['B:VIMIN', 'B:IMINER', 'B:LINFRQ', 'I:IB', 'I:MDAT40']
    self.nbatches = 0
    self.nsamples = 0

    ## Load surrogate model ##
    self.surrogate_model = keras.models.load_model('../surrogate_models/hep_accelerator_5var_08124020_v1.h5')
    self.surrogate_model.summary()

    ## Load data to initilize the env ##
    filename = 'MLParamData_1583906408.4261804_From_MLrn_2020-03-10+00_00_00_to_2020-03-11+00_00_00.h5_processed.csv.gz'
    self.data = dp.load_reformated_cvs('../data/' + filename)

    self.low_state = np.array(
      [self.min_BIMIN,-self.max_IMINER], dtype=np.float64
    )
    
    self.high_state = np.array(
      [self.max_BIMIN, self.max_IMINER], dtype=np.float64
    )
    
    self.observation_space = spaces.Box(
      low   = -self.max_IMINER, 
      high  =  self.max_IMINER, 
      shape = (1,),
      dtype = np.float64
    )

    self.actionMap_VIMIN = [0, 0.0001, 0.001,  0.01, -0.0001,-0.001, -0.01]
    self.action_space = spaces.Discrete(7)
    self.VIMIN = 0
    ##
    self.state = np.zeros(5)
    self.predicted_state = np.zeros(5)
    self.reset

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]
  
  def step(self,action):

    ## Calculate the new B:VINMIN based on policy action
    delta_VIMIN = self.actionMap_VIMIN[action]
    self.VIMIN += delta_VIMIN

    ## Update the B:VIMIN in the predicted state
    self.predicted_state[0] =  self.VIMIN
    self.predicted_state = self.predicted_state.reshape(1,1,-1)
    print(self.predicted_state.shape)

    ## TODO: Fix concatenate
    ## Concate the last prediction to the current state
    self.state = np.concatenate([self.state,self.predicted_state],axis=2)
    print(self.state.shape)

    ## Pop off the older sample from state

    print(self.state.shape)

    ## Predict new state
    self.predicted_state = self.surrogate_model.predict(self.state)
    print(self.predicted_state)
    iminer = self.predicted_state[0][1]
    iminer = self.scalers[1].inverse_transform(np.array([iminer]).reshape(1,-1))
    print(iminer)
    reward = -abs(iminer)
    done = bool(abs(iminer) >= self.max_IMINER*10)



    return self.predicted_state, reward, done, {}
  
  def reset(self):
    ## Prepare the random sample ##
    self.scalers, X_train, Y_train, _ , _ = dp.get_datasets(self.data,self.variables)
    self.nbatches = X_train.shape[0]
    self.nsamples = X_train.shape[2]
    logger.info(X_train.shape)
    this_batch = np.random.randint(0, high=self.nbatches)
    reset_data = X_train[this_batch]
    logger.info(reset_data.shape)
    self.state = reset_data.flatten()#reshape(1,-1)
    self.VIMIN = self.state[int(self.nsamples/len(self.variables))]
    logger.info(self.VIMIN)
    logger.info(self.scalers[0].inverse_transform(np.array([self.VIMIN]).reshape(1,-1)))
    self.state = self.state.reshape(1,1,-1)
    logger.info(self.state.shape)
    return self.state