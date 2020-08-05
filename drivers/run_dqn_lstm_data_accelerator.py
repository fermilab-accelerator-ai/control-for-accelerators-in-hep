# -*- coding: utf-8 -*-
import random, math, gym, time, os, logging, csv, sys
from tqdm import tqdm

##
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RL-Logger')
logger.setLevel(logging.INFO)

from agents.dqn_lstm import DQN

if __name__ == "__main__":
    ###########
    ## Train ##
    ###########
    EPISODES = 100
    NSTEPS   = 10
    best_reward = -100000

    #######################
    ## Setup environment ##
    #######################
    estart = time.time()
    #env = gym.make('gym_accelerator:Data_Accelerator-v0')
    env = gym.make('gym_accelerator:Emulator_Accelerator-v0')
    env._max_episode_steps=NSTEPS
    env.seed(1)
    end = time.time()
    logger.info('Time init environment: %s' % str( (end - estart)/60.0))
    logger.info('Using environment: %s' % env)
    logger.info('Observation_space: %s' % env.observation_space.shape)
    logger.info('Action_size: %s' % env.action_space)

    #################
    ## Setup agent ##
    #################
    agent = DQN(env)
    print("load model:",agent.save_model + './final/data_accelerator_mse_final'+'.weights.h5')
    agent.load(agent.save_model + './final/data_accelerator_mse_final'+'.weights.h5')
    ## Save infomation ##
    train_file_s = open("2_data_accelerator_lstm_episode%s_steps%s_batched_memories_0602420_v1.log" % (str(EPISODES),str(NSTEPS)), 'w')
    train_writer_s = csv.writer(train_file_s, delimiter = " ")   
    train_file_e = open("2_episode_data_accelerator_lstm_episode%s_steps%s_batched_memories_0602420_v1.log" % (str(EPISODES),str(NSTEPS)), 'w')
    train_writer_e = csv.writer(train_file_e, delimiter = " ")  
    
    for e in tqdm(range(EPISODES), desc='RL Episodes', leave=True):
        logger.info('Starting new episode: %s' % str(e))
        current_state = env.reset()
        total_reward=0
        done = False
        step_counter = 0
        episode_loss =[]
        
        while done!=True:
            f_remember = False
            action,policy_type = agent.action(current_state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(current_state, action, reward, next_state, done)
            agent.train()
            logger.info('Current state: %s' % str(current_state))
            #logger.info('Alpha: %s' % str(current_state[16]))
            logger.info('Action: %s' % str(action))
            logger.info('Next state: %s' % str(next_state))
            logger.info('Reward: %s' % str(reward))
            logger.info('Done: %s' % str(done))
            
            ##
            current_state = next_state
            ##
            total_reward+=reward
            step_counter += 1
            
            ##
            if step_counter>=NSTEPS:
                done = True

            ## Save memory
            train_writer_s.writerow([current_state,action,reward,next_state,total_reward,done,policy_type,e])
            train_file_s.flush()
        
        logger.info('total reward: %s' % str(total_reward))
        train_writer_e.writerow([e,total_reward])
        train_file_e.flush()
        print('\ntotal reward: %s' % str(total_reward))
        if total_reward > best_reward:
            agent.save('./best_episodes/dqn_lstm_data_accelerator_mse_'+str(e))
            best_reward = total_reward
           
            
    agent.save('./final/data_accelerator_mse_final')
    train_file_s.close()


