import random, gym, os
from gym import spaces
from gym.utils import seeding
import numpy as np
import pandas as pd
import dataprep.dataset as dp
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RL-Logger')
logger.setLevel(logging.INFO)

np.seterr(divide='ignore', invalid='ignore')
        
class Surrogate_Accelerator_v3(gym.Env):
  def __init__(self):

    self.save_dir='./'
    self.episodes = 0
    self.steps= 0
    self.total_reward=0
    self.max_steps = 100
    ## Define boundary ##
    self.min_BIMIN = 103.3
    self.max_BIMIN = 103.6

    ## Load surrogate models ##
    self.booster_model = keras.models.load_model('../surrogate_models/fullbooster_e250_bs99_nsteps250k_nvar7_axis1_mmscaler_t0_D10062020-T142555_ksplit4__final.h5')

    ## Load scalers ##

    ## Load data ##
    filename = '310_11_more_params.csv'
    data = dp.load_reformated_cvs('../data/' + filename,nrows=250000)
    data['B:VIMIN'] = data['B:VIMIN'].shift(-1)
    data = data.set_index(pd.to_datetime(data.time))
    data = data.dropna()
    data = data.drop_duplicates()

    self.variables = ['B:VIMIN', 'B:IMINER', 'B:VIPHAS', 'B:LINFRQ', 'I:IB', 'I:MDAT40', 'I:MXIB']
    self.scalers = []
    data_list = []
    x_train = []
    ## get_dataset also normalizes the data
    for v in range(len(self.variables)):
      data_list.append(self.get_dataset(data,variable=self.variables[v]))
      self.scalers.append(data_list[v][0])
      x_train.append(data_list[v][1])

    # Axis
    concate_axis = 1

    ## data
    self.X_train = np.concatenate(x_train, axis=concate_axis)
    self.Y_train = np.concatenate((data_list[0][2], data_list[1][2]), axis=1)

    self.X_train_raw = np.copy(self.X_train)
    self.nbatches = self.X_train.shape[0]
    self.nsamples = self.X_train.shape[2]
    self.batch_id = 0 #np.random.randint(0, high=self.nbatches)

    self.observation_space = spaces.Box(
      low   = 0,
      high  = +1,
      shape = (len(self.variables),),
      dtype = np.float64
    )

    self.actionMap_VIMIN = [0, 0.0001, 0.005, 0.001 , -0.0001,-0.005, -0.001]
    self.action_space = spaces.Discrete(7)
    self.VIMIN = 0
    ##
    self.state = np.zeros(shape=(1,len(self.variables),150))
    self.predicted_state = np.zeros(shape=(1,len(self.variables),1))
    logger.debug('Init pred shape:{}'.format(self.predicted_state.shape))

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]
  
  def step(self,action):
    self.steps += 1
    logger.info('Episode/State: {}/{}'.format(self.episodes,self.steps))
    done = False

    ''' Steps:
      1) Update VIMIN based on action
      2) Predict booster variables
      3) Predict next step for injector
      4) Shift state with new values
    '''

    ## Step 1: Calculate the new B:VINMIN based on policy action
    logger.info('Step() before action VIMIN:{}'.format(self.VIMIN))
    delta_VIMIN = self.actionMap_VIMIN[action]
    DENORN_BVIMIN = self.scalers[0].inverse_transform(np.array([self.VIMIN ]).reshape(1, -1))
    DENORN_BVIMIN += delta_VIMIN
    logger.debug('Step() descaled VIMIN:{}'.format(DENORN_BVIMIN))
    if DENORN_BVIMIN < self.min_BIMIN or DENORN_BVIMIN > self.max_BIMIN:
      logger.info('Step() descaled VIMIN:{} is out of bounds.'.format(DENORN_BVIMIN))
      done = True

    self.VIMIN = self.scalers[0].transform(DENORN_BVIMIN)
    logger.debug('Step() updated VIMIN:{}'.format(self.VIMIN))

    logger.debug('Step() state B:VIMIN\n{}'.format(self.state[0,0,-2:1]))

    ## Shift
    self.state[ 0, 0, -1:] = self.VIMIN
    logger.debug('Step() state with updated action on B:VIMIN\n{}'.format(self.state[0,0,-2:1]))

    ## Step 2: Predict using booster model
    self.predicted_state = self.booster_model.predict(self.state)
    self.predicted_state = self.predicted_state.reshape(1, 2, 1)

    ## Step 3: Update IMINER and LINFQN
    #print(self.state.shape)
    #logger.debug('Step() state with pre-state model\n{}'.format(self.state[0,:,-2:]))
    #self.state[0, 1:3, -1:] = self.predicted_state[0,1:3]
    #logger.debug('Step() state with updated state model\n{}'.format(self.state[0,:,-2:]))
    #print(self.state.shape)

    ## Predict the injector variables
    #injector_input = self.state[0,3:5,:].reshape(1,2,150)
    #injector_prediction = self.injector_model.predict(injector_input).reshape(1,2,1)

    #logger.debug('Step() state with pre-injector state model\n{}'.format(self.state[0,:,-2:]))
    #logger.debug('Step() state with injector state model shape\n{}'.format(injector_prediction.shape))
    ##self.state[0, 3:5, -1:] = injector_prediction[0,:]
    #logger.debug('Step() state with updated injector state model\n{}'.format(self.state[0,:,-2:]))

    ## Step 4: Shift state by one step and update last time stamp using predictions
    self.state[0, :, 0:-1] = self.state[ 0, :, 1:]

    ## Update IMINER
    self.state[0, 1:2, -1:] = self.predicted_state[0,1:2]
    ## Update injector variables
    #self.state[0, 3:5, -1:] = injector_prediction[0,:]

    ## Update data state for rendering
    self.data_state = np.copy(self.X_train[self.batch_id+self.steps].reshape(1, len(self.variables), 150))

    ## Use data for everything but the B:IMINER prediction ##
    self.state[0, 2:len(self.variables),:] = self.data_state[0, 2:len(self.variables),:]

    iminer = self.predicted_state[0,1]
    logger.debug('norm iminer:{}'.format(iminer))
    iminer = self.scalers[1].inverse_transform(np.array([iminer]).reshape(1, -1))
    logger.debug('iminer:{}'.format(iminer))
    reward = -abs(iminer)
    if abs(iminer) >= 2:
      logger.info('iminer:{} is out of bounds'.format(iminer))
      done = True

    if done:
      penalty = 5*(self.max_steps - self.steps)
      logger.info('penalty:{} is out of bounds'.format(penalty))
      reward -= penalty

    if self.steps>=int(self.max_steps):
      done = True

    self.total_reward += np.asscalar(reward)

    self.render()

    #print(self.state[0,:,-1:])
    return self.state[0,:,-1:].flatten(), np.asscalar(reward), done, {}
  
  def reset(self):
    self.episodes += 1
    self.steps = 0
    self.total_reward=0
    ## Prepare the random sample ##
    #self.batch_id = np.random.randint(0, high=self.nbatches)
    logger.info('Resetting env')
    self.batch_id=0
    #self.state = np.zeros(shape=(1,5,150))
    logger.debug('self.state:{}'.format(self.state))
    self.state = None
    self.state = np.copy(self.X_train[self.batch_id].reshape(1,len(self.variables),150))
    self.min_BIMIN = self.scalers[0].inverse_transform(self.state[:,0,:]).min()
    self.max_BIMIN = self.scalers[0].inverse_transform(self.state[:,0,:]).max()
    logger.info('Lower and upper B:VIMIN: [{},{}]'.format(self.min_BIMIN,self.max_BIMIN))
    self.min_BIMIN = self.min_BIMIN*0.9999
    self.max_BIMIN = self.max_BIMIN*1.0001
    logger.info('Lower and upper controls: [{},{}]'.format(self.min_BIMIN,self.max_BIMIN))


    ## Copy as data to keep track of what the true accelerator did
    self.data_state = None
    self.data_state = np.copy(self.X_train[self.batch_id].reshape(1,len(self.variables),150))
    logger.debug('self.state:{}'.format(self.state))
    logger.debug('reset_data.shape:{}'.format(self.state.shape))
    self.VIMIN = self.state[0,0,-1:]
    logger.debug('Normed VIMIN:{}'.format(self.VIMIN))
    logger.debug('B:VIMIN:{}'.format(self.scalers[0].inverse_transform(np.array([self.VIMIN]).reshape(1, -1))))

    return self.state[0,:,-1:]

  def render(self):
    '''
    :return:
    '''
    import matplotlib.pyplot as plt
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
    render_dir = self.save_dir+'/render'
    if not os.path.exists(render_dir):
        os.mkdir(render_dir)
    import seaborn as sns
    sns.set_style("ticks")
    nvars = 2#len(self.variables)
    fig, axs = plt.subplots(nvars , figsize=(8, 8))
    start_trace =0
    end_trace   =0
    logger.debug('self.state:{}'.format(self.state))
    for v in range(0,nvars):#len(self.variables)):
      utrace = self.state[0, v, :]
      trace  = self.scalers[v].inverse_transform(utrace.reshape(-1, 1))
      if v==0:
        axs[v].set_title('Total Reward: {:.2f}'.format(self.total_reward))
      axs[v].plot(trace, label='RL Action')
      #if v==1:
      data_utrace = self.data_state[0, v, :]
      data_trace = self.scalers[v].inverse_transform(data_utrace.reshape(-1, 1))
      axs[v].plot(data_trace,'r--', label='Data')
      axs[v].set_xlabel('Time steps')
      axs[v].set_ylabel('{}'.format(self.variables[v]));
      axs[v].legend(loc='upper left')

      '''
      start_trace = end_trace
      end_trace = start_trace + int(self.nsamples / len(self.variables)) - 1
      ##print(self.variables[v])
      utrace = self.state[0,0,start_trace:end_trace]
      #print('utrace:\n {}'.format(utrace))
      trace = self.scalers[v].inverse_transform(utrace.reshape(-1,1))
      #print('trace:\n {}'.format(trace))
      axs[v].plot(trace)
      #axs[v].legend(title=self.variables[v])
      '''
    #plt.show()
    #print(os.getcwd() )
    plt.savefig(render_dir + '/episode{}_step{}_v1.png'.format(self.episodes,self.steps))
    plt.close('all')

    #plt.close()

  def create_dataset(self, dataset, look_back=10*15, look_forward=1):
    X, Y = [], []
    offset = look_back + look_forward
    for i in range(len(dataset) - (offset + 1)):
      xx = dataset[i:(i + look_back), 0]
      yy = dataset[(i + look_back):(i + offset), 0]
      X.append(xx)
      Y.append(yy)
    return np.array(X), np.array(Y)

  def get_dataset(self, df, variable='B:VIMIN'):
    dataset = df[variable].values  # numpy.ndarray
    dataset = dataset.astype('float32')
    dataset = np.reshape(dataset, (-1, 1))
    scaler = MinMaxScaler(feature_range=(0.0001, 1))
    dataset = scaler.fit_transform(dataset)

    ## TODO: Fix
    train_size = int(len(dataset) * 0.70)
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    X_train, Y_train = self.create_dataset(train)
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    Y_train = np.reshape(Y_train, (Y_train.shape[0], Y_train.shape[1]))

    #X_test, Y_test = create_dataset(test, look_back, look_forward)
    #X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    #Y_test = np.reshape(Y_test, (Y_test.shape[0], Y_test.shape[1]))

    return scaler, X_train, Y_train#, X_test, Y_test

