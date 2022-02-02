import gym
import time
import logging
import csv
import os
import random
import numpy as np
import tensorflow as tf
from globals import *
from tqdm import tqdm
from datetime import datetime

# Try plaidml
#os.environ["KERAS_BACKEND"] = "tensorflow.keras.backend"

# Framework class
from agents.dqn import DQN

# Seed value
# Apparently you may use different seed values at each stage
seed_value = 0

# 1. Set the PYTHONHASHSEED environment variable at a fixed value
os.environ['PYTHONHASHSEED'] = str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
tf.random.set_seed(seed_value)

#
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RL-Logger')
logger.setLevel(logging.INFO)

def run(episodes:int = 1,
        nsteps: int = 50,
        doPlay: bool = False):

    now = datetime.now()
    timestamp = now.strftime("D%m%d%Y-T%H%M%S")
    print("date and time:", timestamp)

    # Train
    EPISODES = episodes
    NSTEPS: int = nsteps
    best_reward = -10000
    if doPlay:
        EPISODES = 1
    # Setup environment
    estart = time.time()
    if ENV_TYPE == "continuous":
        env = gym.make ( 'gym_accelerators:ExaBooster-v{}'.format ( ENV_VERSION ) )
    else:
        env = gym.make('gym_accelerators:Surrogate_Accelerator-v{}'.format(ENV_VERSION))
    env._max_episode_steps = NSTEPS
    env.seed(1)
    end = time.time()
    logger.info('Time init environment: %s' % str((end - estart) / 60.0))
    logger.info('Using environment: %s' % env)
    logger.info('Observation_space: %s' % env.observation_space.shape)
    logger.info('Action_size: %s' % env.action_space)

    # Setup agent
    arch_type = ARCH_TYPE
    nmodels = NMODELS
    logger.info('Using DQN {}'.format(arch_type))
    agent = DQN(env,
                cfg= DQN_CONFIG_FILE,
                arch_type=arch_type,
                nmodels=nmodels)
    if doPlay:
        agent.load(LATEST_AGENT_MODEL)
    # Save information
    if ENV_TYPE == "discrete":
        save_directory = DQN_SAVE_DIR +  '/results_dqn_{}_{}_n128_gamma85_250warmup_train5_surrogate_env{}_in{}_out{}_{}/'.format(
            arch_type, nmodels, ENV_VERSION, len(VARIABLES), OUTPUTS, timestamp)
        if doPlay:
            save_directory = DQN_SAVE_DIR + '/play_results_dqn_surrogate_env{}_{}/'.format(ENV_VERSION, timestamp)
    else:
        save_directory = DQN_SAVE_DIR +  '/results_dqn_{}_{}_n128_gamma85_250warmup_train5_exaboost_env{}_in{}_out{}_{}/'.format(
            arch_type, nmodels, ENV_VERSION, len(VARIABLES), OUTPUTS, timestamp)
        if doPlay:
            save_directory = DQN_SAVE_DIR + '/play_results_dqn_exaboost_env{}_{}/'.format(ENV_VERSION, timestamp)
    # Make directory for information
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)
    logger.info('Save directory:{}'.format(save_directory))
    env.save_dir = save_directory
    if ENV_TYPE == "discrete":
        safe_file_prefix = 'fnal_surrogate_env{}_dqn_mlp_episodes{}_steps{}_{}'.format(ENV_VERSION, EPISODES, NSTEPS, timestamp)
    else:
        safe_file_prefix = 'fnal_exaboost_env{}_dqn_mlp_episodes{}_steps{}_{}'.format(ENV_VERSION, EPISODES, NSTEPS, timestamp)

    train_file_s = open(save_directory + safe_file_prefix + '_batched_memories.log', 'w')
    train_writer_s = csv.writer(train_file_s, delimiter=" ")
    train_file_e = open(save_directory + safe_file_prefix + '_reduced_batched_memories.log', 'w')
    train_writer_e = csv.writer(train_file_e, delimiter=" ")

    counter = 0
    for e in tqdm(range(EPISODES), desc='RL Episodes', leave=True):
        logger.info('Starting new episode: %s' % str(e))
        current_state = env.reset()
        total_reward = 0
        done = False
        step_counter = 0
        episode_loss = []

        while not done:
            action, policy_type = agent.action(current_state)
            if doPlay:
                action = agent.play(current_state)
            next_state, reward, done, _ = env.step(action)
            if not doPlay:
                agent.remember(current_state, action, reward, next_state, done)
                if counter >= 250 and counter % 5 == 0:
                    agent.train()

            # Print information
            logger.info('Current state: %s' % str(current_state))
            logger.info('Action: %s' % str(action))
            logger.info('Next state: %s' % str(next_state))
            logger.info('Next state shape: %s' % str(next_state.shape))
            logger.info('Reward: %s' % str(reward))
            logger.info('Done: %s' % str(done))

            # Update  current state
            current_state = next_state

            # Increment total reward
            total_reward += reward
            step_counter += 1
            counter += 1

            # Check if maximum episode steps is reached
            if step_counter >= NSTEPS:
                done = True

            # Save memory
            train_writer_s.writerow([current_state, action, reward, next_state, total_reward, done, policy_type, e])
            train_file_s.flush()

        logger.info('Total reward: %s' % str(total_reward))
        train_writer_e.writerow([e, total_reward])
        train_file_e.flush()
        if total_reward > best_reward:
            agent.save(save_directory + '/best_episodes/policy_model_e{}_'.format(e) + safe_file_prefix)
            best_reward = total_reward

    agent.save(save_directory + '/final/policy_model' + safe_file_prefix)
    train_file_s.close()