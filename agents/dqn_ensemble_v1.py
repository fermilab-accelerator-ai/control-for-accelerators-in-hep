import csv
import json
import logging
import os
import random
from collections import deque

import numpy as np
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from scipy import stats

# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RL-Logger')
logger.setLevel(logging.ERROR)


class DQN:
    def __init__(self, env, cfg='../cfg/dqn_setup.json', nmodels=3):
        logger.info('Building {} models.'.format(nmodels))
        self.env = env
        self.memory = deque(maxlen=5000)
        self.avg_reward = 0
        self.target_train_counter = 0

        self.total_actions_taken = 1
        self.individual_action_taken = np.ones(self.env.action_space.n)
            
        ## Setup GPU cfg
        #config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True
        #sess = tf.Session(config=config)
        #set_session(sess)
        
        ## Get hyper-parameters from json cfg file
        data = []
        with open(cfg) as json_file:
            data = json.load(json_file)
            
        self.search_method = "epsilon"
        self.gamma =  float(data['gamma']) if float(data['gamma']) else 0.95  # discount rate
        self.epsilon = float(data['epsilon']) if float(data['epsilon']) else 1.0  # exploration rate
        self.epsilon_min = float(data['epsilon_min']) if float(data['epsilon_min']) else 0.05
        self.epsilon_decay = float(data['epsilon_decay']) if float(data['epsilon_decay']) else 0.995
        self.learning_rate =  float(data['learning_rate']) if float(data['learning_rate']) else  0.01
        self.batch_size = int(data['batch_size']) if int(data['batch_size']) else 32
        #self.target_train_interval = int(data['target_train_interval']) if int(data['target_train_interval']) else 50
        self.tau = float(data['tau']) if float(data['tau']) else 1.0
        self.save_model = ''#data['save_model'] if str(data['save_model']) else './model'

        self.nmodels = nmodels
        #self.nmodels = float(data['nmodels']) if float(data['nmodels']) else 3.0
        self.do_mode = False
        self.models = []
        self.target_models = []
        self.last_loss = []
        for m in range(self.nmodels):
            self.models.append(self._build_model())
            self.target_models.append(self._build_model())
            self.last_loss.append(100)

        ## Save infomation ##
        train_file_name = "dqn_emsemble_lr%s_v1.log" % str(self.learning_rate)
        self.train_file = open(train_file_name, 'w')
        self.train_writer = csv.writer(self.train_file, delimiter = " ")

    def _build_model(self):
        ## Input: state ##       
        state_input = Input(self.env.observation_space.shape)
        ## Make noisy input data ##
        #state_input = GaussianNoise(0.1)(state_input)
        ## Noisy layer 
        h1 = Dense(56, activation='relu')(state_input)
        #h1 = GaussianNoise(0.1)(h1)
        ## Noisy layer
        h2 = Dense(56, activation='relu')(h1)
        #h2 = GaussianNoise(0.1)(h2)
        ## Output layer
        h3 = Dense(56, activation='relu')(h2)
        ## Output: action ##   
        output = Dense(self.env.action_space.n,activation='linear')(h3)
        model = Model(input=state_input, output=output)
        adam = Adam(lr=self.learning_rate)#, clipnorm=1.0, clipvalue=0.5) ## clipvalue=0.5,clipnorm=1.0,)
        model.compile(loss='mse', optimizer=adam)
        model.summary()
        return model       

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def action(self, state):
        action = 0
        policy_type = 0
        if np.random.rand() <= self.epsilon:
            logger.info('Random action')
            action = random.randrange(self.env.action_space.n)
            ## Update randomness
            if len(self.memory)>(self.batch_size):
                self.epsilon_adj()
        else:
            logger.info('NN action')
            np_state = np.array(state).reshape(1,len(state))
            logger.info('NN action shape{}'.format(np_state.shape))
            act_values_list = []
            for m in range(self.nmodels):
                act_values_list.append(self.target_models[m].predict(np_state)[0])
            print('act_values_list:{}'.format(act_values_list))
            act_values = np.mean(act_values_list, axis=0)
            act_stds = np.std(act_values_list, axis=0)
            act_modes = stats.mode(act_values_list)
            #act_values = []
            #act_stds = []
            #for a in range(self.env.action_space.n):
            #    act_values.append(np.array(act_values_list[a]).mean())
            #    act_stds.append(np.array(act_values_list[a]).std())
            print('action value/mode/std: {}/{}/{}'.format(act_values,act_modes,act_stds))
            action = np.argmax(act_values)
            if self.do_mode:
                action = np.argmax(act_modes)
            policy_type=1

        return action,policy_type

    #def play(self,state):
    #    act_values = self.target_model.predict(state)
    #    return np.argmax(act_values[0])

    def train(self):
        if len(self.memory)<(self.batch_size):
            return

        print('### TRAINING MODEL ###')
        losses = []
        for m in range(self.nmodels):
            minibatch = random.sample(self.memory, self.batch_size)
            batch_states = []
            batch_target = []
            for state, action, reward, next_state, done in minibatch:
                np_state = np.array(state).reshape(1,len(state))
                np_next_state = np.array(next_state).reshape(1,len(next_state))
                expectedQ =0
                if not done:
                    expectedQ = self.gamma*np.amax(self.target_models[m].predict(np_next_state)[0])
                target = reward + expectedQ
                target_f = self.target_models[m].predict(np_state)
                target_f[0][action] = target
            
                if batch_states==[]:
                    batch_states=np_state
                    batch_target=target_f
                else:
                    batch_states=np.append(batch_states,np_state,axis=0)
                    batch_target=np.append(batch_target,target_f,axis=0)
                
            history = self.models[m].fit(batch_states, batch_target, epochs = 1, verbose = 0)
            current_loss = history.history['loss'][0]
            print('Loss for model[{}] {}'.format(m,current_loss))
            losses.append(history.history['loss'][0])
            self.train_writer.writerow([np.mean(losses)])
            self.train_file.flush()
        
            #if current_loss<3*self.last_loss[m]:
            print('### TRAINING TARGET MODEL ###')
            self.target_train()
            self.last_loss[m]=current_loss
            
        return np.mean(losses)

    def target_train(self):
        for m in range(self.nmodels):
            model_weights  = self.models[m].get_weights()
            target_weights =self.target_models[m].get_weights()
            for i in range(len(target_weights)):
                target_weights[i] = self.tau*model_weights[i] + (1-self.tau)*target_weights[i]
            self.target_models[m].set_weights(target_weights)

    def epsilon_adj(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, names):
        for m in range(self.nmodels):
            self.target_models[m].load_weights(names[m])

    def save(self, name):
        abspath = os.path.abspath(self.save_model + name)
        path = os.path.dirname(abspath)
        if not os.path.exists(path):os.makedirs(path)
        for m in range(self.nmodels):
            this_model_name = name + '_id{}'.format(m)
            # Save JSON config to disk
            model_json_name = self.save_model + this_model_name + '.json'
            json_config = self.target_models[m].to_json()
            with open(model_json_name, 'w') as json_file:
                json_file.write(json_config)
            # Save weights to disk
            self.target_models[m].save_weights(self.save_model + this_model_name + '.weights.h5')
            self.target_models[m].save(self.save_model + this_model_name + '.modelall.h5')
            logger.info('### SAVING MODEL ' + abspath + '###')