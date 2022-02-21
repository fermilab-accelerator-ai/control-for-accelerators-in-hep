## How to use the Dockerfile to setup a podman container:

**NOTE:** _The code in this repository can be used to setup a `podman` image (and container based on the new image)._

To install the library run the following steps from your terminal:

1. Pull the package from git via `git clone https://github.com/fermilab-accelerator-ai/control-for-accelerators-in-hep.git`. 
2. If `git` is not setup on the server, clone it to on the local machine and `scp` the folder to the server (provided you have the requisite privileges).
3. Once the directory is set up run ``podman build -t <IMAGE TAG> -v < ABSOLUTE PATH TO THE DIRECTORY ON THE SYSTEM>:<ABSOLUTE PATH TO THE DIRECTORY IN THE CONTAINER> -f Dockerfile .
``
4. After building the image you can run it using ``podman run -it -v < ABSOLUTE PATH TO THE DIRECTORY ON THE SYSTEM>:<ABSOLUTE PATH TO THE DIRECTORY IN THE CONTAINER>:z <IMAGE TAG>
``
5. If run successfully, you will see a new prompt, for example, ``[root@6ccffd0f6421 /]#``. Type `ls -l` to view the files in the container env. It'll show you all the files on the local directory as we have bind mount the volume of the container to the local.
6. Training for the agent can be run using ``python3 run_training.py`` and the results of the training will be stored in the local volume mount.
7. Container can be exported via ``podman export <CONTAINER ID> > <NAME FOR  THE .TAR.GZ FILE>`` to create a copy for file transfer on the local system.
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
