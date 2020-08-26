# -*- coding: utf-8 -*-
import random, math, gym, time, os, logging, csv, sys
from tqdm import tqdm

##
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RL-Logger')
logger.setLevel(logging.INFO)

if __name__ == "__main__":


    #######################
    ## Setup environment ##
    #######################
    estart = time.time()
    env = gym.make('gym_accelerator:Surrogate_Accelerator-v0')
    env.reset()
    end = time.time()
    #logger.info('Time init environment: %s' % str((end - estart) / 60.0))
    #logger.info('Using environment: %s' % env)
    #logger.info('Observation_space: %s' % env.observation_space.shape)
    #logger.info('Action_size: %s' % env.action_space)
    for i in range(500):
        print(env.step(0))
