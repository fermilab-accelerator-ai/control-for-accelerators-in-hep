## How to use the Dockerfile to setup a podman container:

**NOTE:** _The code in this repository can be used to setup a `podman` image (and container based on the new image)._

#### To install the library run the following steps from your terminal:

1. Pull the package from git via `git clone https://github.com/fermilab-accelerator-ai/control-for-accelerators-in-hep.git`. 
2. If `git` is not setup on the server, clone it to on the local machine and `scp` the folder to the server (provided you have the requisite privileges).

#### Build a podman image and container using the following steps:

1. Once the directory is set up run ``podman build -t <IMAGE TAG> -v < ABSOLUTE PATH TO THE DIRECTORY ON THE SYSTEM>:<ABSOLUTE PATH TO THE DIRECTORY IN THE CONTAINER> -f Dockerfile .
``
2. After building the image you can run it using ``podman run -it -v < ABSOLUTE PATH TO THE DIRECTORY ON THE SYSTEM>:<ABSOLUTE PATH TO THE DIRECTORY IN THE CONTAINER>:z <IMAGE TAG>
``
3. If run successfully, you will see a new prompt, for example, ``[root@6ccffd0f6421 /]#``. Type `ls -l` to view the files in the container env. It'll show you all the files on the local directory as we have bind mount the volume of the container to the local.
4. Training for the agent can be run using ``python3 run_training.py`` and the results of the training will be stored in the local volume mount.
5. Container can be exported via ``podman export <CONTAINER ID> > <NAME FOR  THE .TAR.GZ FILE>`` to create a copy for file transfer on the local system.
 
#### To run a command within an existing container use the following steps:

1. Start the container using `podman start <CONTAINER NAME>`
2. Execute the command using `podman exec <CONTAINER NAME> <COMMAND>`. For example, `podman execute gmps-ai python3 run_training.py`
***

## Additional Notes:

* Software requirements remain the same as the previous version of the code. Existing requirements can be viewed in `requirements.txt` and it ought to be used to install any new software packages.
* The environemnt framework is built of [OpenAI Gym](https://gym.openai.com/)
* This is a condensed version of the original code and should not be regarded as the final version.
* [Podman](https://podman.io/getting-started/) is the drop-in replacement for docker.
* Currently, the base image for tensorflow used in the container considers that nvidia GPU is setup on your machine.
* Most global variables that are common throughout the directory are present in `globals.py`. To change the variables, a change must be made in their values in this file ONLY.
  * While `data` folder is empty on the repo, you need the file 'data_release.csv' to run the code. The file can be downloaded from [BOOSTR: A Dataset for Accelerator Control Systems (Partial Release 2020)](https://zenodo.org/record/4088982#.YhAB-ZPMJAc)
***

## Appendix:

_Following variables in the `globals.py` file are used to homogenize the training spanning across different files but using shared variables._

| Variable              | Purpose                                                                                                                                    |
|-----------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| DATA_CONFIG           | Stores the address to the `.json` file that is used to pull the data for training/ testing.                                                |
| LOOK_BACK             | Number of ticks in the look back cycle 1 second (cycle - 15Hz).                                                                            |
| LOOK_FORWARD          | Number of ticks in the look forward cycle                                                                                                  |
| TRAIN_SURROGATE       | If `True`, a new directory to store surrogate training plots will be created under `results/plots`folder. Kept `False` for agent training. |
| VARIABLES             | Top causal variables influencing the outputs.                                                                                              |
| OUTPUTS               | Variables to be considered as outputs during training. MUST include `B:VIMIN`                                                              |
| NSTEPS                | Data entries to be used for  training/ testing.                                                                                            |
| N_SPLITS              | Number of k-fold validation splits to train the surrogate.                                                                                 |
| EPOCHS                | Epochs for training the surrogate                                                                                                          |
| BATCHES               | Batch size for surrogate training                                                                                                          |
| SURROGATE_VERSION     | Version number for the current surrogate.                                                                                                  |
| CKPT_FREQ             | checkpoint frequency                                                                                                                       |
| SURROGATE_CKPT_DIR    | subfolder to store model checkpoints                                                                                                       |
| SURROGATE_DIR         | subfolder to store final trained surrogate                                                                                                 |
| SURROGATE_FILE_NAME   | Common prefix to append to surrogate file name.                                                                                            |
| DQN_CONFIG_FILE       | `.json` file that store training hyperparameters for the agent.                                                                            |
| ARCH_TYPE             | Type of model architecture. Can be 'MLP', 'MLP_Ensemble', or 'LSTM'.                                                                       |
| NMODELS               | Number of models in the architecture. Useful when creating a  'MLP_Ensemble'. Default set to 1 otherwise.                                  |
| LATEST_SURROGATE_MODEL | Absolute path address to the latest surrogate model.                                                                                       |
| ENV_TYPE              | Can be "discrete" or "continuous"                                                                                                          |
| ENV_VERSION           | Version of accelerator env to be used for agent training.                                                                                  |
| AGENT_EPISODES        | Training epochs for the policy model.                                                                                                      |
| AGENT_NSTEPS          | Steps per episode for the policy model.                                                                                                    |
| IN_PLAY_MODE          | If `False` the code run is for training the agent, otherwise the agent is being used in test mode                                          |
|   CORR_PLOTS_DIR                    | Directory to save the correlation plot for the predictions made using RL agent on the data.                                                |
|  EPISODES_PLOTS_DIR                     | Directory to save the `B:VIMIN` and `B:IMINER` as predicted by the RL agent.                                                               |
|   DQN_SAVE_DIR                    | Directory to save the final and best_episode models for the agent.                                                                         |
|  LATEST_AGENT_MODEL                     | Absolute path to the policy model to be used in the PLAY MODE.                                                                             |

