# -*- coding: utf-8 -*-
import random, math, gym, time, os, logging, csv, sys
from tqdm import tqdm

##
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RL-Logger')
logger.setLevel(logging.INFO)

from agents.dqn import DQN

if __name__ == "__main__":
    ###########
    ## Train ##
    ###########
    EPISODES = 250
    NSTEPS   = 100
    best_reward = -100000

    #######################
    ## Setup environment ##
    #######################
    estart = time.time()
    #env = gym.make('gym_accelerator:Data_Accelerator-v0')
    env = gym.make('gym_accelerator:Surrogate_Accelerator-v0')
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
    agent = DQN(env,cfg='../cfg/dqn_setup.json')
    # Save infomation #
    timestamp='09112020'
    save_directory='./results_dqn_{}_v1/'.format(timestamp)
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
            next_state, reward, done, _ = env.step(action)
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


