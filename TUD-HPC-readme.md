
# Table of contents
* [About this document](#About-this-document)
* [General information](#general-information)
* [Getting Started](#Getting-Started)
    * [Python3](#Python3)
    * [log in](#log-in)
    * [Allocating working space](#Allocating-working-space)
    * [Working space shortcut](#Working-space-shortcut)
    * [Cloning the code](#Cloning-the-code)
    * [Submitting Job](#Submitting-Job)
    * [job monitor](#job-monitor)
* [Things are different](#Things-are-different)
    * [Where are the console logs](#Where-are-the-console-logs)
    * [What about my python packages](#What-about-my-python-packages)
    * [Execution permission](#Execution-permission)
* [Job](#job)
    * [An example of a job file](#An-example-of-a-job-file)
* [Important Links](#Important-Links)


## About this document 
This readme file is created to ease and accelerate working with TU-Dresden HPCs for others. It describe the process of running a specific example on HPCs clusters rather than go deep and discuss why & how it works like this. Additional resources have been added to every point which needs more background information. Feel free to contact me in case of questions or corrections @ [shayan.shsh@yahoo.com](shayan.shsh@yahoo.com)

## General information 
Before working with HPCs you need to know that unlike other normal servers, which are normally being used at universities or privately, you cannot run your `bash` or `python`programs directly in the terminal console. Instead, every script needs to be submitted as a [Job](#job) in order to get executed. These jobs will then get queued depending on their resource usages and executed one after each other on the computing machines(Nodes). The system responsible for hardware allocation and queueing the jobs is called batch system. On TUD-HPCs the batch system [SLURM](https://doc.zih.tu-dresden.de/hpc-wiki/bin/view/Compendium/Slurm) is monitoring these tasks. 
 
 
 **NOTE That:** depending on your job and specific hardware you might need for it, it can take up to 2 days for the job to be started. 
 
 
 **NOTE That:** A job can also be submitted by `$ srun` command directly from the console but it is used more for testing purposes
 In order to be able follow also in the code, in this documentation I will be mostly talking about the following files:
 - `parallel_run_shayan.sh`
 - `utils/hpcJobFile.sh`
 
 ## Getting Started
 
 - ### Python3
 This  step is optional but I usually do this.
 run the following command in your user working directory `$ alias python=python3`.
 It helps you to uses `python` instead of `python3` every time you want to start a python3 program.
 
 - ### log in

In order to [login](https://doc.zih.tu-dresden.de/hpc-wiki/bin/view/Compendium/Login) to the TUD Servers, you need to have
1. a username which is provided to you by their Support team via email.
2. [VPN](https://tu-dresden.de/zih/dienste/service-katalog/arbeitsumgebung/zugang_datennetz/vpn) Connection to TUD servers.
3. A Terminal environment to ssh to the servers. (Putty does work for Windows users as well)

You can start a ssh connection with the following command to the TUD Servers after starting your VPN.
`$ ssh username@taurus.hrsk.tu-dresden.de` where `username` is the provided username by TUD Support team.
 
 - ### Allocating working space
 Compute nodes, does not have access to your user directory, that's why it is necessary to allocate a special directory which is accessible for the compute nodes.
 There are different type of those directories(in HPC documentation they are called workespaces). Different workspaces and their specs can be found in [this page](https://doc.zih.tu-dresden.de/hpc-wiki/bin/view/Compendium/WorkSpaces). Each one of these working spaces store files for a specific amount of time and after that they archive them and then delete them. No worries since you can extend the time. For this specific example `parallel_run_shayan.sh` the best workspace is `scratch`. It keeps data for 100 Days and it can be extended 10 times. In the next paragraph it is explained how to create one and access it.


 A scratch workspace can be allocated with the following command in your user directory.
 `$ ws_allocate -F scratch -r 10 -m name.lastname@tu-dresden.de NameOfTheDirectory01 100`
 
- `-r 10` means it should send a reminder Email 10 days before the directory expires
- `-m name.lastname@tu-dresden.de` is the email address to which the notification will be sent
- `NameOfTheDirectory01` is the name of the directory which is bein created. I suggest directory names have some numbers or details as postfix since it is possible that there are multiple directories are created for one project.
- `100` is the initial days scratch will keep this directory. min=1, max=100

**NOTE:** A directory can be extended with the following command:
`$ ws_extend -F scratch NameOfTheDirectory01 100`

**NOTE:** All working workspaces which are created by an account can be listed with the command: `$ ws_list` and it look like:
```bash
id: KGE_Rotate_rotate
     workspace directory  : /scratch/ws/1/username-KGE_Rotate_rotate
     remaining time       : 99 days 23 hours
     creation time        : Thu Aug 13 14:37:45 2020
     expiration date      : Sat Nov 21 13:37:45 2020
     filesystem name      : scratch
     available extensions : 9
id: KGE
     workspace directory  : /scratch/ws/1/username-KGE
     remaining time       : 91 days 2 hours
     creation time        : Tue Aug  4 17:19:52 2020
     expiration date      : Thu Nov 12 16:19:52 2020
     filesystem name      : scratch
     available extensions : 10
```
It is important to know where your workspaces are build since you have to submit you jobs from there.

- ### Working space shortcut
It is possible to create a shortcut to all you live workspaces into you user home directory with the following command: 
`ws_register DIR` Then all of your workspaces will be listed under `/DIR/` in your home directory. Bare in mind that every time these directory changes you need to run the command `ws_register DIR` again to update the actual directories as well.


- ### Cloning the code
After creating the [workspace](#Allocating-working-space), a git repository can be cloned into the workspace directory. To make the process easier a job file can be created into the git branch and only submitted in the workspace.

- ### Submitting Job
in this example, a job file is available in `/utils` directory with the name `hpcJobFile.sh`. This file can be submitted with `$ sbatch hpcJobFile.sh` in the following directory: `/scratch/ws/1/username-KGE/KGE_Pattern/utils` where:
- `username` is the TUD username
- `KGE` is the workspace name
- `KGE_Pattern` is the git repository name

To check your workspace path check end of [allocating working space](#Allocating-working-space) section.


 
 - ### job monitor
 Slowly you should be able to submit your first test job. There is a good option provided from TUD HPCs which let you to observe ongoing and every job which is executed before by a user. You can see the Job Monitoring page [here](https://selfservice.zih.tu-dresden.de/l/index.php/hpcportal/jobmonitoring//zih/jobs). You need to log in in order to visit the page. 


## Things are different

Yes, on a HPC cluster almost everything is different than the normal environment we used to work with. here I go through the point that might confuse one when starting to work with hpc:

#### Where are the console logs

It is common to use console logs/prints to debug a program. On HPC clusters though it is not possible to see a live console log. Everything your script writes on console will end up in your output file. Check `#SBATCH -o` flag in [job](#job)s for more information.

#### What about my python packages

Since the script, in this example `parallel_run_shayan.sh ` does not run within your user working directory, if you use any python packages, you need to install them in an virtual environment(venv) and activate this in your job file before your script gets executed. Check the job example file in [job](#job) section to see how a venv can get activate.

 ------------
 If you are not familiar to creating a venv and installing packages here is a short guide:
 1. After [logging in](#log-in) to your user account on HPC run the following command in the terminal: `$ python3 -m venv envs/env01`
 2. Activate the just created venv with: `$ source envs/env01/bin/activate`
 3. install all your package dependencies with: `$ pip install torch torchvision numpy sklearn` here `torch`, `torchvision`, `numpy` and `sklearn` are the packages which are needed to run `run.py` in this repository.
 ------------
 
Now that we have a venv under `/envs/env01` it can be activated for any job by being added as following in a job file: `source /home/username/envs/env01/bin/activate` where `username` is the given username from TUD with which you ssh to their servers.
 

 #### Execution permission
 Compute nodes need to be granted the permission to execute the scripts which are used in the job file. for the following [job](#An-example-of-a-job-file), permissions are needed to be adjusted for `parallel_run_shayan.sh` as follows:
 
 `$ chmod +x parallel_run_shayan.sh`

## Job
A slurm job defines how much resources an execution file, be it a `bash` or a `python` script, does need to work properly.
The slurm file I have been using is `hpcJobFile.sh` and looks like this:

#### An example of a job file 
```bash
#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=1:00:00   # walltime
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1      # limit to one node
#SBATCH --cpus-per-task=1  # number of processor cores (i.e. threads)
#SBATCH --mem-per-cpu=62000M   # memory per CPU core
#SBATCH -J "test-shayan"   # job name
#SBATCH -o test-shayan-slurm-%j.out
#SBATCH --mail-user=shayan.shahpasand@mailbox.tu-dresden.de   # email address
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT,TIME_LIMIT_90
#SBATCH -A p_ml_nimi


source /home/shsh829c/venv/env1/bin/activate

echo $PWD
/scratch/ws/1/shsh829c-KGE/KGE_Pattern/parallel_run_shayan.sh 

exit 0
```

A detailed description of all available options for a job file can be found [here](https://doc.zih.tu-dresden.de/hpc-wiki/bin/view/Compendium/Slurm#Job_Submission).
There are some options that I highly recommend to use:
- `#SBATCH -o test-shayan-slurm-%j.out` where `%j` is your job ID.
This option gives your output file a more readable name which makes it much easier to find after all those experiments. remember that you'll have many of those files and there should be good way to distinguish them fast from each other. If this option is not passed the output will have only the job ID. e.g: `3345678.out`
- `echo $PWD` which prints the current **working directory** where your job is submitted in the [output/console]((#Where-are-the-console-logs?)) file
- `#SBATCH --mail-user=firstname.lastname@mailbox.tu-dresden.de`  which helps you to get notification to your email about your job.
- `#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT,TIME_LIMIT_90` This let the job now on which events ot should send an email. Bare in mind you can also use `ALL` instead of all the types I've listed here if you want to get email notification for every event.

## Important Links

1. [HPC compendium](https://doc.zih.tu-dresden.de/hpc-wiki/bin/view/Compendium/WebHome)
2. [PDF- introduction to HPC](https://doc.zih.tu-dresden.de/hpc-wiki/pub/Compendium/WebHome/HPC-Introduction.pdf?t=1597323405)
3. [Available partitions on TUD-HPC](https://doc.zih.tu-dresden.de/hpc-wiki/bin/view/Compendium/SystemTaurus#Partitions)  