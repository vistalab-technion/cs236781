---
title: Using the course HPC servers
permalink: assignments/hpc-servers
excerpt: How to connect to and run jobs on the course servers.
toc: true
toc_label: Contents
toc_sticky: true
date: 2020-10-31
published: true
---

Those of you who are officially enrolled to the course and requested access
using the server registration form will have access to dedicated
high-performance computing servers (`lambda1-4`) provisioned by Computer
Science faculty IT department. Running on the faculty servers will give you
access to more computing power and also fast GPUs (which will greatly
accelerate your deep-learning tasks). This should significantly speed up your
workflow when performing the course homework assignments and when implementing
your final project.

These servers are mainly suited for running batch jobs which you can submit to
dedicated job queues and be notified upon completion.  We therefore recommend
you install and work on the assignments locally (on your own machine), and 
use the faculty servers mainly to run long model training tasks (we will
specify in the assignment).

# Logging in 

Logging in is performed with your Technion Single Sign-On (SSO) credentials.
Usually this means the username and password of your `@campus` or `@technion`
email address.

If your username is e.g. `user`, login like so

```shell
ssh user@lambda.cs.technion.ac.il
```

or, use the server's IP directly if you're connecting over a VPN or receive an
unknown hostname error:

```shell
ssh user@132.68.39.36
```

Notes:
1. Your credentials will only work after we pass a final list of
   registered students to the faculty IT department.
   This will happen during the first 2-3 weeks of the semester.
1. These servers are only directly accessible from **within the Technion networks**.
   If connecting over WiFi, do not use the `TechPublic` network, as it won't
   allow you to connect. The `TechSec` network will work, as well as other
   non-open faculty networks (e.g. `CS-WIFI`).
1. `lambda` is a gateway server that you connect to in order to run jobs on the
   actual compute nodes (`lambda1-4`) as explained below.
   You should not run any computations on `lambda` itself as it does not have a
   GPU and is limited in CPU resources.
1. In some internal Technion networks, and when connecting through VPN, the DNS
   lookup does not find the `lambda` hostname. If you get a `could not
   resolve hostname` error, use the second option (directly with IP).

## Connecting from home

To connect from home you'll need to configure a VPN connection to the Technion. See the
[instructions](https://cis.technion.ac.il/en/central-services/communication/off-campus-connection/ssl-vpn/)
on the Technion CIS website regarding how to set this up. After you connect
though the VPN, you can connect to the server as normal.

Note that we cannot provide you with technical support regarding how to
setup/use the VPN. You should contact CIS if you need VPN support.

# Server Usage

## General

### Storage and Environment

Your home directory on the gateway server `lambda` (e.g. `/home/user`) is
automatically mounted on all the computation nodes `lambda1-4`. This ensures
that any programs you install locally under your home folder (for example a
`conda` environment) will be available for jobs running on these nodes.

In fact, the **first thing** you should do after connecting for the first time
is to install `conda` and the course `conda` environment within your home
folder. Follow the instructions for linux on the [getting started page]({{
site.baseurl }}{% link _assignments/00-getting-started.md %}).

Note that your home folder will be deleted from the server at some point after
the semester ends, likely without prior notice. Please make sure to backup your files.

### Computation

The faculty HPC server cluster is composed of a gateway server, `lambda`, into which
you log in with SSH, and four compute nodes `lambda1-4` which run the actual
computations. The gateway server is relatively weak and has no attached GPUs, so
it should not be used for running computations.

Again, to be clear: **do not run any computations directly on `lambda`!** after
logging in, only run computations through `slurm` as described below.

The computation tasks are manged by a job scheduling system called
[`slurm`](https://slurm.schedmd.com/).  The system manages the computation
nodes and resources and allocates them to jobs submitted by users into a queue.
If you wish, you can read the `slurm` [quick start
guide](https://slurm.schedmd.com/quickstart.html) to get a better understanding
of the system and the available commands.

The most useful `slurm` commands for our needs are,
- [`sbatch`](https://slurm.schedmd.com/sbatch.html)
- [`srun`](https://slurm.schedmd.com/srun.html)
- [`squeue`](https://slurm.schedmd.com/squeue.html)
- [`scancel`](https://slurm.schedmd.com/scancel.html)

## Running interactive jobs

An interactive job allows you to view it's output and interact with it in
real time, as if it were running on the machine you're logged
in to.

Submitting an interactive job is performed with the `srun` command. Required resources
can be specified and if they're available the job starts running immediately.

### Example

Let's see how to run an `ipython` console session as an interactive job with an
allocated GPU (notice that the course `conda` env is already active).

```shell
(cs236781-hw) avivr@lambda:~/cs236781-hw$ srun -c 2 --gres=gpu:1 --pty ipython
Python 3.8.6 | packaged by conda-forge | (default, Oct  7 2020, 19:08:05)
Type 'copyright', 'credits' or 'license' for more information
IPython 7.19.0 -- An enhanced Interactive Python. Type '?' for help.

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
specify any command and also add command arguments after it.

Notes:
1. You should use interactive jobs for debugging or running short tasks like
   launching `jupyter`. If you need to run something long, it's better to
   submit a batch job instead (see below).
1. When you submit an interactive job, your shell is blocked (by `srun`) until
   it completes. If you terminate `srun`, it will cancel your job. Crucially,
   this means that if you log out of the machine while running an interactive
   job, the job will terminate (as with regular processes you invoke from the
   shell). You can get around this by either,
    - Using terminal managers e.g. `screen` and `tmux`;
    - Running with `nohup`;
    - Running a batch job instead (preferred). See below.

1. You should activate your `conda env` before running an interactive job if
   you need to run python.
   The shell environment variables will be passed to the process that will run
   your job on the compute node, so therefore the `conda env` will effectively
   also be active there.
1. You can specify `bash` as the command to run in an interactive job to get a
   shell on one of the compute nodes.
1. Jobs may be terminated after running for more than 24 hours due to policy.

## Running batch jobs

A batch job is submitted to the queue with the `sbatch` command.
It runs non-interactively when resources are available and sends it's output to
files that you can specify. Additionally, it can notify you by email when the
job starts and finishes.

Running jobs with `sbatch` is useful for long-running processes such as
training models or running experiments. While the job is running, it's not
connected to any specific shell session and thus it keeps running if you log
out of the machine. To view output from a batch job, you'll need to read it
from the file it writes to.

To use `sbatch`, you need to create a script for it to run. It can be any script
with a valid shebang line (`#!`) at the top, e.g. a bash script or a python
script.

### Example

Lets create a file `~/myscript.sh` on the server with the following contents:
```bash
#!/bin/bash

# Setup env
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate cs236781-hw
echo "hello from $(python --version) in $(which python)"

# Run some arbitrary python
python -c 'import torch; print(f"i can haz gpu? {torch.cuda.is_available()}")'
```

Then we can run the script as a `slurm` batch job as follows:
```shell
# Run a batch job with slurm
avivr@lambda:~$ sbatch -c 2 --gres=gpu:1 -o slurm-test.out -J my_job  myscript.sh
Submitted batch job 3550

# Check the job status in the queue
avivr@lambda:~$ squeue 
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
              3550       all   my_job    avivr  R       0:01      1 lambda2

# See the output from the job by following the output file contents
(cs236781-hw) avivr@lambda:~/cs236781-hw$ tail -f slurm-test.out
/bin/bash: warning: setlocale: LC_ALL: cannot change locale (en_US.UTF-8)
hello from Python 3.8.6 in /home/avivr/miniconda3/envs/cs236781-hw/bin/python
i can haz gpu? True
^C
```

Here the `-c 2` and `--gres=gpu:1` options specify that we want to allocate 2 CPU
cores and one GPU to the job, the `-o slurm-test.out` option specifies where
to write the output from the process and `-J my_job` is an arbitrary name we can
assign to the job.

### Viewing status

After submitting a batch job, you can use `squeue` to view it's
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
conda activate cs236781-hw
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

You can run `jupyter` on a compute node by running a small script what we
provide, `jupyter-lab.sh`.

You should run this script via `srun`, like so:
```shell
(cs236781-hw) avivr@lambda:~/cs236781-hw$ srun -c2 --gres=gpu:1 jupyter-lab.sh

[I 22:47:14.620 LabApp] JupyterLab extension loaded from /home/avivr/miniconda3/envs/cs236781-hw/lib/python3.8/site-packages/jupyterlab
[I 22:47:14.621 LabApp] JupyterLab application directory is /home/avivr/miniconda3/envs/cs236781-hw/share/jupyter/lab
[I 22:47:14.624 LabApp] Serving notebooks from local directory: /home/avivr/cs236781-hw
[I 22:47:14.624 LabApp] Jupyter Notebook 6.1.4 is running at:
[I 22:47:14.624 LabApp] http://132.68.39.38:8888/?token=5d426de30111ee82c7af4789077dbc4cf7a996f9e46476e6
[I 22:47:14.624 LabApp]  or http://127.0.0.1:8888/?token=5d426de30111ee82c7af4789077dbc4cf7a996f9e46476e6
[I 22:47:14.624 LabApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 22:47:14.632 LabApp]

    To access the notebook, open this file in a browser:
        file:///home/avivr/.local/share/jupyter/runtime/nbserver-20063-open.html
    Or copy and paste one of these URLs:
        http://132.68.39.38:8888/?token=5d426de...
     or http://127.0.0.1:8888/?token=5d426de...
```

The connection URL in the console will show the IP of the compute node that the
server is actually running on. Use the **first url** to open jupyter in your
browser.

Note: Please be considerate of other students and use the computing
resources only as needed. Do not leave unattended jupyter notebooks just
running without a reason.

# Tips

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
    ssh-copy-id user@lambda.cs.technion.ac.il
    ```
1. If you use an intermediate server to connect from home, make sure to first
   also copy your public key to that server.

After generating your key pair, you should also [add it to your github
account](https://help.github.com/articles/adding-a-new-ssh-key-to-your-github-account/).
After that, you can use the SSH remote-URLs (instead of HTTPS) to clone repos
and prevent the need to specify your username and password when
`push`ing, `pull`ing and `fetch`ing.

## Transferring files to and from the server

The `rsync` tool can be helpful. It can automatically sync between local and
remote folders, only uploading/downloading modified files.

For example, to send files or a directory you can do

```shell
rsync -Cavz path/to/local/file_or_dir user@lambda:/home/user/path/to/remote/file_or_dir
```

To send files from home via an intermediate server (in this example CSM):

```shell
rsync -Cavz -e 'ssh -A -J user@csm.cs.technion.ac.il' path/to/local/file_or_dir user@lambda:/home/user/path/to/remote/file_or_dir
```

To download files from the server to your computer, simply change the order of
the last two arguments in the above examples.

### GUI option for macOS users

[Cyberduck](https://cyberduck.io/) is a free remote file browser that you can
use to copy files to/from the server using a GUI.

### GUI option Windows users

Many people recommend [MobaXterm](https://mobaxterm.mobatek.net/) as
a good graphical ssh client for windows.


# Additional Information

Please also read the [Faculty lambda help
page](https://hpc.cswp.cs.technion.ac.il/2020/08/31/lambda-computational-cluster/).

It contains other important information you need to know:
- Limits on resources for jobs (number of hours, maximal GPUs, etc).
- How jobs are prioritized between different students of the course,
  and between different courses.
- Code of conduct. Failure to meed this code will cause revocation of your
  account.


