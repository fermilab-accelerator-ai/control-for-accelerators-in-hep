import os
import sys
from datetime import datetime
sys.path.append('.')

DATA_CONFIG = sys.path[-1] + "/config/data_setup.json"

LOOK_BACK: int = 15 #150

LOOK_FORWARD: int = 1

TRAIN_SURROGATE: bool = False  #if True, the code runs to train a new surrogate

# VARIABLES = ['B:VIMIN', 'B:IMINER', 'B:LINFRQ', 'I:IB', 'I:MDAT40']
VARIABLES = ['B:VIMIN', 'B:IMINER', 'B_VIMIN', 'B:LINFRQ', 'I:IB', 'I:MDAT40']

OUTPUTS:int = 2 #2

NSTEPS: int = 250000

N_SPLITS: int = 5

EPOCHS: int = 250

BATCHES: int = 99

SURROGATE_VERSION = 6

DQN_CONFIG_FILE = sys.path[-1 ] + '/config/dqn_setup.json'

ARCH_TYPE = 'MLP'

NMODELS = 1

RESULTS_DIR = sys.path[-1] + '/results'

LATEST_SURROGATE_MODEL = sys.path[-1] + '/models/surrogate_models/surrogate_model_version_6/fullbooster_noshift_look_back15_e250_bs99_nsteps250k_invar6_outvar2_axis2_mmscaler_timestampD02102022-T203032_v6_kfold5__final.h5'

CKPT_FREQ = 20

SURROGATE_CKPT_DIR =  sys.path[-1] + '/models/surrogate_models/surrogate_ckpts_version_{}'.format(SURROGATE_VERSION)

SURROGATE_DIR = sys.path[-1] + '/models/surrogate_models/surrogate_model_version_{}'.format(SURROGATE_VERSION)

SURROGATE_FILE_NAME = 'fullbooster_noshift_look_back{}_e{}_bs{}_nsteps{}k_invar{}_outvar{}_axis{}_mmscaler_timestamp{}_v{}'

ENV_TYPE = "continuous" #"continuous"

ENV_VERSION: int = 1 #2

AGENT_EPISODES: int = 1000

AGENT_NSTEPS: int = 20

IN_PLAY_MODE: bool = False # Turn True to run the agent in the test mode

#  --------------- make new directories to save training plots categorized by timestamp ---------------

now = datetime.now ( )
timestamp = now.strftime ( "D%m%d%Y-T%H%M%S" )

if TRAIN_SURROGATE == True:
    SURROGATE_PLOT_DIR = sys.path[-1] + "/results/plots/surrogate_plots/"
    surrogate_path = os.path.dirname(SURROGATE_PLOT_DIR)
    surrogate_dir = "surrogate_plots_{}".format(timestamp)
    surrogate_dir_path = os.path.join(surrogate_path,surrogate_dir)
    os.mkdir(surrogate_dir_path)
    PLOTS_DIR_FOR_SURROGATE = surrogate_dir_path

POLICY_RESULTS_DIR  = sys.path[-1] +  '/results/plots/policy_plots/'
ep_dir = "episode_plots_{}".format(timestamp)
dir_path = os.path.dirname(POLICY_RESULTS_DIR)
eps_path = os.path.join(dir_path, ep_dir)
os.mkdir(eps_path)
EPISODES_PLOTS_DIR = eps_path
if ENV_TYPE == "discrete":
    corr_dir = "correlation_plots_{}".format( timestamp )
    corr_path = os.path.join(dir_path, corr_dir)
    os.mkdir(corr_path)
    CORR_PLOTS_DIR = corr_path

#  --------------- make new directories to save training plots categorized by timestamp ---------------

DQN_SAVE_DIR = sys.path[-1] + '/models/policy_models'

LATEST_AGENT_MODEL = sys.path[-1] + '/models/policy_models/results_dqn_MLP_1_n128_gamma85_250warmup_train_exaboost_env1_in6_out2_D02172022-T162712/best_episodes/policy_model_e666_fnal_exaboost_env1_dqn_mlp_episodes1000_steps20_D02172022-T162712.weights.h5'
DATA_REWARD_LIST, RL_REWARD_LIST,IMINER_IMP_LIST = [],[],[]
