import random,sys,os
import numpy as np
from collections import deque
import tensorflow as tf
print("tf version ==> ",tf.__version__)
from tensorflow import keras
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Dropout,Input,GaussianNoise,BatchNormalization,LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import csv,json,math


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed



#from keras.backend.tensorflow_backend import set_session


import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RL-Logger')
logger.setLevel(logging.ERROR)

#The Deep Q-Network (DQN)
class DQN:
    def __init__(self, env,cfg='cfg/dqn_setup.json'):
        self.env = env
        self.memory = deque(maxlen = 2000)
        self.avg_reward = 0
        self.target_train_counter = 0

        self.total_actions_taken = 1
        self.individual_action_taken = np.ones(self.env.action_space.n)

        ##
        tf_version = int((tf.__version__)[0])
        
        if tf_version < 2:
            ## Setup GPU cfg
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            set_session(sess)
        elif tf_version >= 2:
            print("tf >2")
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.compat.v1.Session(config=config)
            tf.compat.v1.keras.backend.set_session(sess)
        
        ## Get hyper-parameters from json cfg file
        data = []
        with open(cfg) as json_file:
            data = json.load(json_file)
            
        self.search_method = "epsilon"
        self.gamma =  float(data['gamma']) if float(data['gamma']) else 0.95  # discount rate
        self.epsilon = float(data['epsilon']) if float(data['epsilon']) else 1.0  # exploration rate
        self.epsilon_min = float(data['epsilon_min']) if float(data['epsilon_min']) else 0.05
        self.epsilon_decay = float(data['epsilon_decay']) if float(data['epsilon_decay']) else 0.995
        self.learning_rate =  float(data['learning_rate']) if float(data['learning_rate']) else  0.001
        self.batch_size = int(data['batch_size']) if int(data['batch_size']) else 32
        self.target_train_interval =  50
        self.tau = float(data['tau']) if float(data['tau']) else 1.0
        self.save_model = './models/'

        self.model = self._build_model()
        self.target_model = self._build_model()    

    def _build_model(self):
        model = Sequential()
        model.add(LSTM(256, return_sequences=True,input_shape=(1, self.env.observation_space.shape[0])))
        #model.add(GaussianNoise(0.1))
        model.add(LSTM(256, return_sequences=True))
        #model.add(GaussianNoise(0.1))
        model.add(LSTM(256))
        #model.add(GaussianNoise(0.1))
        model.add(Dense(self.env.action_space.n,))
        opt = Adam(lr=1e-2)
        model.compile(loss='mean_squared_error', optimizer=opt)
        model.summary()
        return model       

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def action(self, state):
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.env.action_space.n)
            ## Update randomness
            if len(self.memory)>(self.batch_size):
                self.epsilon_adj()
            return action, 0
        else:
            np_state = np.array(state).reshape(1,1,len(state))
            act_values = self.target_model.predict(np_state)
            action = np.argmax(act_values[0])
            return action, 1

    def play(self,state):
        act_values = self.target_model.predict(state)
        return np.argmax(act_values[0])

    def train(self):
        if len(self.memory)<(self.batch_size):
            return

        logger.info('### TRAINING MODEL ###')
        losses = []
        minibatch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state, done in minibatch:
            #print ("minibatch state:",state)
            np_state = np.array(state).reshape(1,1,len(state))
            np_next_state = np.array(next_state).reshape(1,1,len(next_state))
            expectedQ =0
            if not done:
                expectedQ = self.gamma*np.amax(self.target_model.predict(np_next_state)[0])
            target = reward + expectedQ
            target_f = self.model.predict(np_state)
            target_f[0][action] = target
            history = self.model.fit(np_state, target_f, epochs = 1, verbose = 0)
            losses.append(history.history['loss'])
        self.target_train()
        return 0

    def target_train(self):
        self.target_train_counter = 0
        model_weights  = self.model.get_weights()
        target_weights =self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = self.tau*model_weights[i] + (1-self.tau)*target_weights[i]
        self.target_model.set_weights(target_weights)

    def epsilon_adj(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)
        self.target_model.load_weights(name)

    def save(self, name):
        abspath = os.path.abspath(self.save_model + name)
        path = os.path.dirname(abspath)
        if not os.path.exists(path):os.makedirs(path)
        # Save JSON config to disk
        model_json_name = self.save_model + name + '.json'
        json_config = self.model.to_json()
        with open(model_json_name, 'w') as json_file:
            json_file.write(json_config)
        # Save weights to disk
        self.target_model.save_weights(self.save_model + name+'.weights.h5')
        self.target_model.save(self.save_model + name+'.modelall.h5')
        #logger.info('### SAVING MODEL '+abspath+'###')
