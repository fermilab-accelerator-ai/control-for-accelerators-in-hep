# -*- coding: utf-8 -*-
import random, math, gym, time, os, logging, csv, sys
from tqdm import tqdm
from datetime import datetime

##
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RL-Logger')
logger.setLevel(logging.INFO)

from agents.dqn import DQN
#from agents.dqn_ensemble_v1 import DQN

if __name__ == "__main__":

    now = datetime.now()
    timestamp = now.strftime("D%m%d%Y-T%H%M%S")
    print("date and time:", timestamp)

    doPlay=False

    ###########
    ## Train ##
    ###########
    EPISODES = 50
    NSTEPS   = 100
    best_reward = -100000
    if doPlay==True:
        EPISODES=1
    #######################
    ## Setup environment ##
    #######################
    estart = time.time()
    #env = gym.make('gym_accelerator:Surrogate_Accelerator-v0')
    env = gym.make('gym_accelerator:Surrogate_Accelerator-v3')
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
    #nmodels=10
    #agent = DQN(env,cfg='../cfg/dqn_setup.json',nmodels=nmodels)
    #agent.do_mode = True
    if doPlay==True:
        agent.load('../policy_models/results_dqn_09132020_v2/best_episodes/policy_model_e143_fnal_surrogate_dqn_mlp_episodes250_steps100_09132020.weights.h5')
    # Save infomation #
    save_directory='./results_dqn_surrogate3d_{}_v1/'.format(timestamp)
    if doPlay == True:
        save_directory = './play_results_dqn_surrogate3_{}_v1/'.format(timestamp)
    # Save infomation #
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)
    env.save_dir=save_directory
    safe_file_prefix = 'fnal_surrogate_dqn_mlp_episodes{}_steps{}_{}'.format(EPISODES,NSTEPS,timestamp)
    train_file_s = open(save_directory+safe_file_prefix+'_batched_memories.log','w')
    train_writer_s = csv.writer(train_file_s, delimiter = " ")
    train_file_e = open(save_directory+safe_file_prefix+'_reduced_batched_memories.log','w')
    train_writer_e = csv.writer(train_file_e, delimiter = " ")

    for e in tqdm(range(EPISODES), desc='RL Episodes', leave=True):
        logger.info('Starting new episode: %s' % str(e))
        current_state = env.reset()
        total_reward=0
        done = False
        step_counter = 0
        episode_loss =[]

        while done!=True:
            action,policy_type = agent.action(current_state)
            if doPlay==True:
                action = agent.play(current_state)
            next_state, reward, done, _ = env.step(action)
            if doPlay!=True:
                agent.remember(current_state, action, reward, next_state, done)
                agent.train()
            logger.info('Current state: %s' % str(current_state))
            #logger.info('Alpha: %s' % str(current_state[16]))
            logger.info('Action: %s' % str(action))
            logger.info('Next state: %s' % str(next_state))
            logger.info('Next state shape: %s' % str(next_state.shape))
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
        logger.info('\ntotal reward: %s' % str(total_reward))
        if total_reward > best_reward:
            agent.save(save_directory+'/best_episodes/policy_model_e{}_'.format(e)+safe_file_prefix)
            best_reward = total_reward


    agent.save(save_directory+'/final/policy_model'+safe_file_prefix)
    train_file_s.close()


