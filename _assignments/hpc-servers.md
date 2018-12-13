---
title: Using the course HPC servers
excerpt: How to connect to and run jobs on the course servers.
toc: true
toc_label: Contents
toc_sticky: true
date: 2018-10-02
---

Those of you who are officially enrolled to the course will have access to
dedicated high-performance computing servers (`rishon1-4`) provisioned by
Computer Science faculty IT department. Running on the faculty servers will give
you access to more computing power and also fast GPUs (which will greatly
accelerate your deep-learning tasks). This should significantly speed up your
workflow when performing the course homework assignments and when implementing
your final project.

These servers are mainly suited for running batch jobs which you can submit to
dedicated job queues and be notified upon completion.  We therefore recommend
you install and work on the assignments locally (on your own machine), and only
use the faculty servers when you need to run a long model training task (we will
specify in the assignment).

# Logging in 

Logging in is performed with your Technion Single Sign-On (SSO) credentials.
Usually this means the username and password of your `@campus` or `@technion`
email address.

If your username is e.g. `user`, login like so

```shell
ssh user@rishon.technion.ac.il
```

or, directly using the server's IP:

```shell
ssh user@132.68.39.36
```

Notes:
1. The `rishon` server is only directly accessible from **within the Technion networks**.
1. In some internal Technion networks the DNS lookup seems to not find
   the `rishon` hostname. If you get a `could not resolve hostname` error, use
   the second option (directly with IP).

## Connecting from home

If you need to connect from home, first SSH into a Technion server that’s
accessible from the outside (e.g. CSM) and from there you can SSH into `rishon`. 

You can do this in one command like so:

```shell
ssh -J user@csm.cs.technion.ac.il user@rishon.cs.technion.ac.il
```

This example will connect through the CSM server in the CS faculty. You should
be able to use other Technion servers that you have SSH access to.

This method (`-J`) has the useful advantage of forwarding the SSH public key
from your local machine (if available) to the target machine though the
intermediate machine.

Notes:
1. Unfortunately the `t2`/`lux` student server cannot be used to access the
   Technion network from outside due to CIS
   [policy](https://cis.technion.ac.il/news/t2-server-upgraded-to-lux/).
1. If you use CSM, note that the credentials for the CSM server and `rishon` are
   not the same: CSM uses the CS-faculty credentials while the `rishon` server
   uses the Technion SSO credentials.

# Usage

## General

The faculty HPC server cluster is composed of a gateway server (`rishon`) into which
you log in with SSH, and four compute nodes `rishon1-4` which run the actual
computations. The gateway server is relatively weak and has no attached GPUs, so
it should not be used for running computations.

Your home directory on the gateway server (e.g. `/home/user`) is automatically
mounted on all the computation nodes. This ensures that any programs you
install locally under your home folder (for example a `conda` environment) will
be available for jobs running on these nodes.

The computation tasks are manged by a job scheduling system called
[`slurm`](https://slurm.schedmd.com/).  The system manages the computation nodes
and resources and allocates them to jobs submitted by users into a queue (AKA
partition).  If you wish, you can read the `slurm` [quick start
guide](https://slurm.schedmd.com/quickstart.html) to get a better understanding
of the system and the available commands.

The most useful `slurm` commands for our needs are,
- [`sbatch`](https://slurm.schedmd.com/sbatch.html)
- [`srun`](https://slurm.schedmd.com/srun.html)
- [`squeue`](https://slurm.schedmd.com/squeue.html)
- [`scancel`](https://slurm.schedmd.com/scancel.html)

## Job queues

We have a dedicated job queue ("partition") for our course, `236605`. Jobs
submitted to this queue will be served on one of the `rishon3` or `rishon4`
nodes.

You can view the jobs currently in the course queue by running `squeue -p
236605` or an all queues with `squeue`.

Each job defines which computational resources it requires (nodes, CPU cores, number of
GPUs). Multiple jobs can run simultaneously on a compute node as long as
their computational requirements can be satisfied.

For example, if job1 requires 2 CPU cores and 1 GPU and job2 requires 4 CPU
cores and 2 GPUs then they can run together on the same compute node if that node has at
least 6 CPU cores and 3 GPUs.

## Priority and preemption

Students from our course have special priority for the compute nodes serving the
`236605` queue.  Other users in the system can also run jobs on these nodes, but
if resources become limited a job submitted by a student in this course will
**preempt** (in this context, stop and re-queue) a job submitted by an external user that's
currently running on one of these nodes.

Likewise if you submit your jobs without specifying a queue, they may end up
running on a compute node that gives priority to other users. This might be OK
if you're running an interactive job or a jupyter server (see sections below),
but it's not a good idea for long-running batch jobs. The reason is that the
system has the following **major limitation**: it can't guarantee GPU state when your
process starts running again after being preempted. In other words, if you get
preempted while training some model on a GPU, the process may resume later on
some other node with an inconsistent GPU state. Not fun. To prevent this, **always
use the course queue for running model training tasks**.

Specifying which queue to submit a job to is performed with the `-p` flag for
the `srun`/`sbatch` commands. For example, use `-p 236605` to make sure your job
is submitted to the course queue. If you don't specify it, it will be submitted
to the `all` queue which is served by all compute nodes, not just those that
give priority to course students.

The advantage of not specifying a specific queue is that your job may start
sooner since it can run on any node. For interactive jobs, this may allow you to
start immediately.

## Running interactive jobs

An interactive job allows you to view it's output and interact with it in
real time, as if it were running on the machine you're logged
in to.

Submitting an interactive job is performed with the `srun` command. Required resources
can be specified and if they're available the job starts running immediately.

### Example

Let's see how to run an `ipython` console session as an interactive job with an
allocated GPU.

```shell
(cs236605-hw) avivr@rishon:~/cs236605-hw1$ srun -c 2 --gres=gpu:1 --pty ipython
cpu-bind=MASK - rishon1, task  0  0 [15995]: mask 0x100000001 set
cpu-bind=MASK - rishon1, task  0  0 [15995]: mask 0x100000001 set
Python 3.7.0 (default, Oct  9 2018, 10:31:47)
Type 'copyright', 'credits' or 'license' for more information
IPython 7.1.1 -- An enhanced Interactive Python. Type '?' for help.

In [1]: import torch

In [2]: torch.cuda.is_available()
Out[2]: True

In [3]: t = torch.tensor([1,2,3], dtype=torch.float).cuda()

In [4]: t.dot(t)
Out[4]: tensor(14., device='cuda:0')
```

Here the `-c 2` and `--gres=gpu:1` options specify that we want to allocate 2 CPU
cores and one GPU to the job, the `--pty` option is required for the session
to be interactive and the last argument `ipython` is the command to run. You can
specify any command and also add command arguments after it. In this example we
didn't specify the queue to submit to (`-p`) which may or may not be right for
your use case. Indeed, you can see that the example ran on `rishon1`, which is not one
of the dedicated course nodes.

Notes:
1. You should use interactive jobs for debugging or running short one-off
   tasks.  If you need to run something long, submit a batch job instead.
1. When you submit an interactive job, your shell is blocked (by `srun`) until
   it completes. If you terminate `srun`, it will cancel your job.
   Crucially, this means that if you log out of the machine while running an
   interactive job, the job will terminate (as with regular processes you
   invoke from the shell). You can get around this by either,
    - Using terminal managers e.g. `screen` and `tmux`;
    - Running with `nohup`;
    - Running a batch job instead (preferred). See below.

   The reason the last method is preferred is that interactive jobs run with
   `srun` may be terminated after running for a few hours due to policy.
1. You should activate your `conda env` before running an interactive job if
   you need to run python.
   The shell environment variables will be passed to the process that will run
   your job on the compute node, so therefore the `conda env` will effectively
   also be active there.
1. You can specify `bash` as the command to run in an interactive job to get a
   shell on one of the compute nodes.

## Running batch jobs

A batch job is submitted to the queue with the `sbatch` command.
It runs non-interactively when resources are available and sends it's output to
files that you can specify. Additionally, it can notify you by email when the
job starts and finishes.

Running jobs with `sbatch` is useful for long-running processes such as training
models. While the job is running, it's not connected to any specific shell
session and thus it keeps running if you log out of the machine. To view output
from a batch job, you'll need to read it from the file it writes to.

To use `sbatch`, you need to create a script for it to run. It can be any script
with a valid shebang line (`#!`) at the top, e.g. a bash script or a python
script.

### Example

Lets create a file `~/myscript.sh` on the server with the following contents:
```bash
#!/bin/bash

# Setup env
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate cs236605-hw
echo "hello from $(python --version) in $(which python)"

# Run some arbitrary python
python -c 'import torch; print(f"i can haz gpu? {torch.cuda.is_available()}")'
```

Then we can run the script as a `slurm` batch job as follows:
```shell
avivr@rishon:~$ sbatch -c 2 --gres=gpu:1 -p 236605 -o slurm-test.out -J my_job  myscript.sh
Submitted batch job 114425

avivr@rishon:~$ squeue -p 236605
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
            114425    236605   my_job    avivr  R       0:01      1 rishon3

avivr@rishon:~$ tail -f slurm-test.out
cpu-bind=MASK - rishon3, task  0  0 [20442]: mask 0x100000001 set
cpu-bind=MASK - rishon3, task  0  0 [20442]: mask 0x100000001 set
hello from Python 3.7.0 in /home/avivr/miniconda3/envs/cs236605-hw/bin/python
i can haz gpu? True
```

Here the `-c 2` and `--gres=gpu:1` options specify that we want to allocate 2 CPU
cores and one GPU to the job, the `-p 236605` option specifies the name of the
job queue (partition) to use, the `-o slurm-test.out` option specifies where
to write the output from the process and `-J my_job` is an arbitrary name we can
assign to the job.

### Viewing status

After submitting a batch job, you can use `squeue -p 236605` to view it's
status in the queue, as shown in the example above. You can see the job name and
it's id there.

### Viewing output

Each job you submit causes a text file you be created in your current directory,
named `slurm-<jobid>.out`.

To view the output from a job in real time, you can use `tail -f` or `less -r
+F` on the output file for the relevant job. `less` allows you to also scroll
back.

### Canceling

To cancel a batch job you've submitted (whether it's running or waiting in the
queue), run `scancel <job-id>` where `<job-id>` is the id you received when
starting the batch job.

### Course helper script

To slightly simplify your workflow on the server, we provide you with a simple
script to run python code from the course `conda env` as a `slurm` batch job.

The homework assignment repos contain a script called `py-sbatch.sh`. You can
use this script as if it were the `python` command, and it will active the
`conda env` for you and execute your provided python code with `sbatch`.

For example, let's say we want to run all our notebooks with the `main.py`
script. Instead of
```shell
conda activate cs236605-hw
python main.py run-nb *.ipynb
```

which will run on the gateway server, do this
```shell
./py-sbatch.sh main.py run-nb *.ipynb
```

This will take care of activating the `conda env` and run the script on the more
powerful compute nodes as a batch job. The script has some declared variables
which you can edit to configure the `sbatch` parameters such as computational
resources, notification email address and others.

Note that for the above example it may have been more straightforward to use an
interactive job (`srun`). However this script may be useful when you need to
create a batch job running a python script, for example to run long training
tasks.

In any case, this script is completely optional since you can always use
`sbatch` directly as shown in the previous section.

## Running `jupyter`

You can run `jupyter` on a compute node by creating a script that exposes the IP
of the compute node as the jupyer server URL.

For example, if you create a script `jupyter-lab.sh` like so
```bash
#!/bin/bash
unset XDG_RUNTIME_DIR
jupyter lab --no-browser --ip=$(hostname -I) --port-retries=100
```

then you can start the jupyter lab server with `srun`, e.g.
```shell
srun -c 2 --gres=gpu:1 --pty jupyter-lab.sh
```

The connection URL in the console will show the IP of the compute node that the
server is actually running on.

We'll provide you with a similar script in the assignment repos.

Notes:
1. As mentioned previously, interactive jobs are not meant to be long-running.
   Please be considerate of other students and use the computing resources only
   as needed.
1. In the above example we didn't specify a specific queue. Read the
   above section about preemption to understand the implications of this.

### Accessing jupyter from home

Although the `rishon` servers are only accessible within the Technion networks,
it's possible to connect from home to a jupyter instance running on them by using a
combination of SSH port forwarding and using an intermediate server.

1. Follow the instructions above to start jupyter on one of the compute nodes.
2. Observe the IP and port of the jupyter server specified on the command line.
   Let's assume you got this line after jupyer started:

    ```shell
    [I 21:39:07.830 LabApp] http://132.68.39.38:8888/?token=abcdef0123...
    ```
3. Additionally, let's assume you can you have SSH access from home to another
   Technion server, such as CSM as in the previous examples.
4. Then you can run the following from your machine (from a different terminal):

   ```shell
   ssh -L 9999:132.68.39.38:8888 -J user@csm.cs.technion.ac.il user@rishon
   ```

   This creates a local port forwarding from port `9999` on your `localhost` to
   `132.68.39.38:8888` from the CSM machine through an SSH tunnel, and also
   gives you a new SSH session on `rison` which you can work from.

5. To connect to the jupyter lab server from home, you can now point your
   local browser to `localhost:9999`.

# Tips

## For windows users

Many people recommend [MobaXterm](https://mobaxterm.mobatek.net/) as
a good graphical ssh client for windows.

## Pubic-key based authentication

You can use a public-key based authentication to prevent the need for typing
your password when connecting to remote servers over SSH.

1. Generate an SSH key pair using the `ssh-keygen` tool. More detailed instructions for all platforms can be found
   [here](https://help.github.com/articles/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent/).
1. Copy the **public** key. By default it's in `~/.ssh/id_rsa.pub`. Make sure you copy it exactly
   without any extra spaces or newlines.
1. Connect to your user on the machine and paste the public key contents into a new line in `~/.ssh/authorized_keys`.

Notes:

1. On macOS and linux, there's a utility you can use to automate steps 2-3.
   After generating the key pair, copy the public key to the server like so:
   ```shell
   ssh-copy-id user@rishon.cs.technion.ac.il
   ```
1. If you use an intermediate server to connect from home, make sure to first
   also copy your public key to that server.

After generating your key pair, you should also [add it to your github
account](https://help.github.com/articles/adding-a-new-ssh-key-to-your-github-account/).
After that, you can use the SSH remote-URLs (instead of HTTPS) to clone repos
and prevent the need to specify your username and password when
`push`ing, `pull`ing and `fetch`ing.

## Git-based workflow

A very effective way to synchronize your local work with the server (and also with your
assignment partner) is with `git`. Note that `git` is already installed on the
server.

You can [fork](https://help.github.com/articles/fork-a-repo/) the assignment repo.
Then you can clone your fork both onto your local machine and to the server. This
will allow you to work on either machine while synching changes both with
yourself (on each machine) and with your partner.

If you decide to work this way you should also add the original assignment repo
as a new `git remote` so that you can pull updates from us.

## Transferring files to and from the server

The `rsync` tool can be your friend. For example, to send files or a directory
you can do

```
rsync -Cavz path/to/local/file_or_dir user@rishon:/home/user/path/to/remote/file_or_dir
```

To send files from home via an intermediate server (in this example CSM):
```
rsync -Cavz -e 'ssh -A -J user@csm.cs.technion.ac.il' path/to/local/file_or_dir user@rishon:/home/user/path/to/remote/file_or_dir
```

To download files from the server to your computer, simple change the order of
the last two arguments in the above examples.

