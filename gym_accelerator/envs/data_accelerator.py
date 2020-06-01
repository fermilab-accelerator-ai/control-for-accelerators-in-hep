import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding

#
import subprocess, os, math, random, json
from collections import defaultdict 
import pandas as pd
import shutil, joblib

from keras.models import load_model

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('DATA-AcceleratorModel-Logger')
logger.setLevel(logging.ERROR)

class Data_Accelerator(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,cfg_file='../../cfg/data_setup.json'):
     
        """ 
        Description:
           Environment used to run the Booster's surrogate model for unregulated prediction and a regulation on that prediction to minimize the error

        Model source code:
           https://gitlab.pnnl.gov/schr476/control-for-accelerators-in-hep.git

        Model execusion procedure:
           
        Environment variables:
           - Current setting (B_VIMIN)
           - Current regulated error (B:IMINER) --calculated from B:IMINER = 10*(B:VIMIN_regulated-B_VIMIN)
           - Current regulated measurement (B:VIMIN_regulated) --add error regulation to B:VIMIN
        
        Observation states (for RL agent):
           - Current unregulated measurement (B:VIMIN) --provided by the surrogate model
           - Current regulated measurement (B:VIMIN_reg) -- calculated using the input actions
           - Current regulated error (B:IMINER)
           - Current alpha 
           - Current gamma

        Action space: 
           - Change delta alpha
           - Change delta gamma
           
        Reward:
           - smape (B:IMINER)
           
        """
        ## Load config
        self.basePath = os.path.dirname(__file__)+'/' 
        self.cfg = []
        with open(self.basePath + cfg_file) as json_file:
            self.cfg = json.load(json_file)
        #print (self.cfg)
        
        ## Application setup
        self.data = self.load_data()
        self.data_index = 0

        ## Initial settings
        self.alpha = 8.5e-2
        self.gamma = 7.535e-5
        self.alpha_baseline = self.alpha
        self.gamma_baseline = self.gamma
        self.beta_last = 0.0
        self.beta_last_baseline = 0.0

        ##
        self.best_alpha = self.alpha
        self.best_reward = -100
        self.step_counter=0
        ##
        self.state = {} 
        
        self.state['B_VIMIN']     = self.data['B_VIMIN'][self.data_index:self.data_index+15].tolist()
        self.state['B:VIMIN_data']= self.data['B:VIMIN'][self.data_index:self.data_index+15].tolist()
        
        self.actionMap_alpha = [0, 0.0001, 0.001,  0.01,   
                                  -0.0001,-0.001, -0.01]
        
        ##
        self.next_state = {}

        ## Env state output:  ##
        alpha_min = 7.500e-2#8.500e-2 + 5*5e-02
        alpha_max = 1.200e-1#8.500e-2 - 5*5e-02
        gamma_min = 1e-6#7.535e-5 - 5*1e-05
        gamma_max = 1e-2#7.535e-5 + 5*1e-05
        
        m_min = np.ones(15)*100
        low = np.append(np.array([100]), m_min)
        low = np.append(low,alpha_min)
        low = np.append(low,gamma_min)
        
        m_max = np.ones(15)*110
        high = np.append(np.array([110]), m_max)
        high = np.append(high, alpha_max)
        high = np.append(high, gamma_max)
        
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        logger.info('alpha_high %s' % str(self.observation_space.high[16]))
        logger.info('alpha_low %s' % str(self.observation_space.low[16]))
        
        self.action_space = spaces.Discrete(7)

    def step(self, action):

        self.step_counter+=1
        ## Default
        reward =-1
        done = True
        
        ## Update 
        delta_alpha = self.actionMap_alpha[action] 
        delta_gamma = 0.0
        self.alpha = self.alpha+delta_alpha
        self.gamma = self.gamma+delta_gamma
         
        ## Check action boundary conditions
        if self.alpha>self.observation_space.high[16]:
            self.alpha-=delta_alpha
            logger.info("High alpha:  %s (%s) " %(self.alpha,delta_alpha))
            return self._getState(self.state),reward,done, {}
        if self.alpha<self.observation_space.low[16]:
            self.alpha+=delta_alpha
            logger.info("Low alpha:  %s (%s) " %(self.alpha,delta_alpha))
            return self._getState(self.state),reward,done, {}
         
        ## Copy the setting to the next time
        self.next_state['B_VIMIN']       = self.state['B_VIMIN']
        self.next_state['B:VIMIN_data']  = self.data['B:VIMIN'][self.data_index:self.data_index+15].tolist()
        
        self.data_index += 16
        if self.data_index+15 > len(self.data['B:VIMIN']): self.data_index=0
        ## Calculate the regulated voltage
        reg = self._get_regulation(self.next_state['B:VIMIN_data'])
        reg_baseline = self._get_regulation_baseline(self.next_state['B:VIMIN_data'])
        self.reg = reg
        self.reg_baseline = reg_baseline
        
        err = 10 * (reg - self.next_state['B:VIMIN_data'])
        self.err = err
        err_avg = self.err_avg(err)
        
        err_baseline = 10 * (reg_baseline - self.next_state['B:VIMIN_data'])
        self.err_baseline = err_baseline
        err_avg_baseline = self.err_avg(err_baseline)
        
        ## The regulated voltage should push the measured voltage closer to the set voltage
        biminer_limit=1e-5
        reward = biminer_limit/abs(err_avg) #np.average(reward)

        ## Update current states 
        self.state['B_VIMIN']         = self.next_state['B_VIMIN']
        self.state['B:VIMIN_data']    = self.next_state['B:VIMIN_data']

        if self.best_reward < reward:
            self.best_reward = reward
            self.best_alpha = self.alpha
        #
        done = False #Question: done == end of cycle?
        return self._getState(self.state), reward, done, {}
                   

    def _getState(self, state):
        ls = [ v for v in state.values() ]
        o = np.array(ls[0][0])
        o = np.append(o,np.array(ls[1])).tolist()
        o = np.append(o,self.alpha)
        o = np.append(o,self.gamma)
        #np_next_state = np.array(list_state)
        return o

    def _get_prediction(self,cur_state):
        x = self.x_scaler.transform(np.array([[cur_state]]))
        x = np.expand_dims(x, axis=0)
        #x = np.expand_dims(x, axis=0)
        y_pred = self.model.predict(x)[0]
        y_pred_rescaled = self.y_scaler.inverse_transform(y_pred)
        return y_pred_rescaled
    
    def _get_regulation(self,fitted):
        #print("====>adjusted")
        beta_last = self.beta_last
        gamma = self.gamma
        alpha = self.alpha
        #print("alpha: ",alpha)
        #print("gamma: ",gamma)
        
        length = len(fitted)
        beta=np.zeros(length)
        beta[0]=beta_last
        
        _MIN = self.state["B_VIMIN"] #setting
        ER = 10*(np.array(fitted)-_MIN) # TODO: Ask Jason and Gabe if this is correct 
        #print (self.data_index)
        #print ("ER: ",ER)
        
        beta[1:] = [beta[i-1]+gamma*ER[i] for i in range(1,length)]
        MIN_regulated = _MIN + alpha * ER + beta #predict the next, shiftting happens in the plotting
        
        self.beta_last = beta[-1]
        return MIN_regulated
    
    def _get_regulation_baseline(self,fitted):
        #print("====>baseline")
        beta_last = self.beta_last_baseline
        gamma = self.gamma_baseline
        alpha = self.alpha_baseline
        
        length = len(fitted)
        beta=np.zeros(length)
        beta[0]=beta_last
        
        _MIN = self.state["B_VIMIN"] #setting
        ER = 10*(np.array(fitted)-_MIN) # TODO: Ask Jason and Gabe if this is correct 
        
        beta[1:] = [beta[i-1]+gamma*ER[i] for i in range(1,length)]
        MIN_regulated = _MIN + alpha * ER + beta #predict the next, shiftting happens in the plotting
        
        self.beta_last_baseline = beta[-1]
        return MIN_regulated
         
    def reset(self):
        self.alpha = self.best_alpha
        self.data_index += 16
        if self.data_index+15 > len(self.data['B:VIMIN'])-30: self.data_index=0
        self.state['B_VIMIN']     = self.data['B_VIMIN'][self.data_index:self.data_index+15].tolist()
        self.state['B:VIMIN_data']= self.data['B:VIMIN'][self.data_index:self.data_index+15].tolist()
        return self._getState(self.state) 

    def render(self):
        return 0

    def close(self):
        return 0
        
    def load_data(self):
        dataPath = self.basePath + self.cfg['data_dir'] + self.cfg['data_name']
        df = pd.read_csv(dataPath)
        df=df.replace([np.inf, -np.inf], np.nan)
        df=df.dropna(axis=0)
        df=df.round(decimals=5)
        print(len(df))
        return df
    
    def RMS(self,err):
        err_avg = np.average(abs(err))
        RMS = np.sqrt(np.sum(np.square(err-err_avg)))
        return RMS
    
    def err_avg(self,err):
        err_avg = np.average(abs(err))
        return err_avg
