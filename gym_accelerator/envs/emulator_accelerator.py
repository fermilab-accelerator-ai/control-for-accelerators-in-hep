import random, gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class emulator(gym.Env):
  def __init__(self,df):
    self.df = df
    
    self.min_BIMIN = 103
    self.max_BIMIN = 104
    self.max_IMINER = 1
    
    self.low_state = np.array(
      [self.min_BIMIN,-self.max_IMINER], dtype=np.float32
    )
    
    self.high_state = np.array(
      [self.max_BIMIN, self.max_IMINER], dtype=np.float32
    )
    
    self.observation_space = spaces.Box(
      low   = -self.max_IMINER, 
      high  =  self.max_IMINER, 
      shape = (1,),
      dtype = np.float32
    )
    
    self.action_space = spaces.Box(
      low   = self.min_BIMIN,
      high  = self.max_BIMIN,
      shape = (1,),
      dtype = np.float32
    )
    
    #print(df)
    self.state = [self.df["B:IMINER"][0]]
    #print("end of init-->state:", self.state)
    #self.state = self.reset(self.df)
    #print(self.state)
    
  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]
      
  def predict(self,x):
    #print ("predict-->action x: ",x)
    y=self._BIMINER_linear(x)
    r=self._random_from_cdf(x)
    #print("y+r: ",y,r)
    return y+r
  
  def step(self,action):
    self.err = self.predict(action)
    #print("step-->error: ",self.err)
    #print("error>max_IMINER: ", self.err, self.max_IMINER)
    self.done = bool(
      abs(self.err) >= self.max_IMINER*20 # fail
    )
    
    self.reward = 0
    if self.done:
      self.reward = -10
    self.reward = -abs(self.err)
    
    print("step-->action/reward: ",action,self.reward)
    self.state = np.array([self.err])
    #print("end of step-->\n")
    return self.state, self.reward, self.done, {}
  
  def reset(self,df):
    self.seed()
    self.df = df
    self.df = self._prepData(self.df)
    init = self.np_random.uniform(low=self.min_BIMIN, high=self.max_BIMIN) #random init. control
    #print ('reset-->init: random action:',init)
    self.err = self.predict(init)
    self.state = np.array([self.err])
    #print('end of reset-->state:',self.state)
    
  def _prepData(self,df):
    self.m, self.b = self._LinearRegression(df)
    df["IMINER_linear"]=self._BIMINER_linear(df["B:VIMIN"]) # the linear regression portion
    df["IMINER_std"]=df["B:IMINER"]-df["IMINER_linear"] # calculate the deviation 
    return df
    
  def _LinearRegression(self,df):
    x_min = df["B:VIMIN"].min()
    x_max = df["B:VIMIN"].max()
    y_min = df[df["B:VIMIN"]==x_min]["B:IMINER"].max()
    y_max = df[df["B:VIMIN"]==x_max]["B:IMINER"].min()
    m=(y_min-y_max)/(x_min-x_max)
    b=y_min-m*x_min
    return m,b
  
  def _BIMINER_linear(self,x):
    y=self.m*x+self.b
    return y
  
  def _random_from_cdf(self,x):
    sampling_window = 0.005
    error_window = 0.01
    
    nbins = 100
    df = self.df
    state = self.state
    #print("state: ==>", state)
    error = state[0]
    #print(error)
    
    #filter according to x-axis:
    y_std = df[df["B:VIMIN"].between(x-sampling_window,x+sampling_window)]
    #y_std = y_std[y_std["B:IMINER"].between(error-error_window,error+error_window)]
    
    # get the y-axis noise value according to the x-axis filtering
    hist, bins = np.histogram(y_std["IMINER_std"], bins=nbins)
    bin_midpoints = bins[:-1] + np.diff(bins)/2
  
    cdf = np.cumsum(hist)
    cdf = cdf / cdf[-1]
  
    values = np.random.rand(len(y_std)*10)
    value_bins = np.searchsorted(cdf, values)
    random_from_cdf = bin_midpoints[value_bins]
  
    if len(random_from_cdf)>0:
      i=random.randint(0,len(random_from_cdf)-1)
      return random_from_cdf[i]
    else:
      return 0