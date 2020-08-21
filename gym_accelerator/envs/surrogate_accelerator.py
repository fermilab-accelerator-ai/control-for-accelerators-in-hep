import random, gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pandas as pd
import dataprep.dataset as dp
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

np.seterr(divide='ignore', invalid='ignore')
        
class Surrogate_Accelerator(gym.Env):
  def __init__(self):

    ## Define boundary ##
    self.min_BIMIN = 103.3
    self.max_BIMIN = 103.4
    self.max_IMINER = 1

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

    ##
    self.state = np.zeros(5)
    self.reset

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]
  
  def step(self,action):

    ##
    delta_VIMIN = self.actionMap_VIMIN[action]
    self.VIMIN += delta_VIMIN

    ## New state
    self.new_state = self.surrogate_model(self.VIMIN)

    ## R
    self.done = bool(
      abs(self.err) >= self.max_IMINER*20 # fail
    )
    
    self.reward = 0
    if self.done:
      self.reward = -10
    self.reward = -abs(self.err)
    
    #print("step-->action/reward: ",action,self.reward)
    self.state = np.array([self.err])
    #print("step-->state/action/reward: ",self.state,action,self.reward)
    #print("end of step-->\n")
    return self.state, self.reward, self.done, {}
  
  def reset(self):
    ## Prepare the random sample ##
    variables = ['B:VIMIN', 'B:IMINER', 'B:LINFRQ', 'I:IB', 'I:MDAT40']
    self.scalers, X_train, Y_train, _ , _ = dp.get_datasets(self.data,variables)
    nbatches = X_train.shape[0]
    nsamples = X_train.shape[2]
    print(X_train.shape)
    this_batch = np.random.randint(0, high=nbatches)
    reset_data = X_train[this_batch]
    print(reset_data.shape)
    self.state = reset_data.flatten()#reshape(1,-1)
    self.VIMIN = self.state[int(nsamples/len(variables))]
    print(self.VIMIN)
    print(scalers[0].inverse_transform(np.array([self.VIMIN]).reshape(1,-1)))
    return self.state