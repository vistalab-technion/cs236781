---
title: Getting started with the course assignments
permalink: assignments/getting-started
toc: true
toc_label: Contents
toc_sticky: true
date: 2019-03-17
---

This document will help you get started with the course homework assignments.
Please read it carefully as it contains crucial information.

## General

The course homework assignments are mandatory and a large part of the grade.
They entail writing code in python using popular third-party machine-learning
libraries and also theoretical questions.

The assignments are implemented in part on a platform called Jupyter notebooks.
[Jupyter](http://jupyter.org/) is a widely-used tool in the machine-learning
ecosystem which allows us to create interactive notebooks containing live code,
equations and text. We'll use jupyter notebooks to guide you through the
assignments, explain concepts, test your solutions and visualize their outputs.

To install and manage all the necessary packages and dependencies for the
assignments, we use [conda](https://conda.io), a popular package-manager for
python.  The homework assignments come with an `environment.yml` file which
defines what third-party libraries we depend on. Conda will use this file to
create a virtual environment for you. This virtual environment includes python
and all other packages and tools we specified, separated from any preexisting
python installation you may have. Detailed installation instructions are below.
We will not support any other installation method other than the one
described.

For working on the code itself, we recommend using
[PyCharm](https://www.jetbrains.com/pycharm/),
however you can use any other
editor or IDE if you prefer. Note that you can get the professional version of
PyCharm for free by using your Technion student email - see
[here](https://www.jetbrains.com/student/).


### Project structure

Each assignment's root directory contains the following files and folders:

- `cs236781`: Python package containing course utilities and helper functions.
  You do not need to edit anything here.
- `hwN` where `N` is the assignment number: Python package containing the
  assignment code. **All** your solutions will be implemented here.
- `tests`: A package containing tests that run all the assignment notebooks and
  fail if there are errors.
- `PartN_XYZ.ipynb` where `N` is a number and `XYZ` is some name:
  A set of jupyter notebooks that contain the instructions that will guide you
  through the assignment. You won't need to edit these except if you wish to
  play around and to write your name at the beginning.
- `main.py`: A script providing some utilities via a CLI.
  Mainly, you'll run it to create your submission after completing the
  assignment.
- `environment.yml`: A file for `conda`, specifying the third-party packages it
  should install into the virtual environment it creates.

## Environment set-up

1. Install the python3 version of [miniconda](https://conda.io/miniconda.html).
   Follow the [installation instructions](https://conda.io/docs/user-guide/install/index.html)
   for your platform.

   For example, on linux you should do:
   ```shell
   curl -fsSLO https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh
   # Accept EULA
   # Install in default directory
   # Select no for editing .bashrc

   # Update your bashrc like so:
   echo "source $HOME/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc
   ```

   On macOS it's similar but with a different script URL
   ```shell
   curl -fsSLO https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh 
   bash Miniconda3-latest-MacOSX-x86_64.sh
   # Rest is the same
   ```

2. Use conda to create a virtual environment for the assignment.
   From the assignment's root directory, run

   ```shell
   conda env create -f environment.yml
   ```

   This will install all the necessary packages into a new conda virtual
   environment named `cs236781-hw`.

3. Activate the new environment by running

   ```shell
   conda activate cs236781-hw
   ```

   *Activating* an environment simply means that the path to it's python binaries
   (and packages) is placed at the beginning of your `$PATH` shell variable.
   Therefore, running programs installed into the conda env (e.g. `python`) will
   run the version from the env since it appears in the `$PATH` before any other
   installed version.

   To check what conda environments you have and which is active, run

   ```shell
   conda env list
   ```

   or, you can run `which python` and you should see the python binary is in a
   subfolder of `~/miniconda3/envs/cs236781-hw/`.

   You can find more useful info about conda environments
   [here](https://conda.io/docs/user-guide/tasks/manage-environments.html).

Notes: 

- You only need to do steps 1 and 2 above once (not for each assignment).
  However, the third-party package dependencies (in the `environment.yml` file)
  might slightly change from one
  assignment to the next. To make sure you have the correct versions run
  ```shell
  conda env update
  ```
  from the assignment root directory every time a new assignment is published.

- Always make sure the correct environment is active. It will revert to it's
  default each new terminal session. If you want to change the default env you
  can add a `conda activate` in your `~/.bashrc`.

## Working on the assignment

### Running Jupyter

Make sure that the active conda environment is `cs236781-hw`, and run

```shell
jupyter lab
```

This will start a [jupyter lab](https://jupyterlab.readthedocs.io/en/stable/)
server and open your browser at the local server's url. You can now start working.
Open the first notebook (`Part0`) and follow the instructions.

If you're new to jupyter notebooks, you can get started by reading the
[UI guide](https://jupyter-notebook.readthedocs.io/en/stable/notebook.html#notebook-user-interface)
and also about how to use notebooks in
[JupyterLab](https://jupyterlab.readthedocs.io/en/latest/user/notebook.html).

Note that if you are familiar with and prefer the regular `jupyter notebook` you
can use that instead of `jupyter lab`.

### Implementing your solution and answering questions

- The assignment is comprised of a set of notebooks and accompanying code
  packages.
- You only need to edit files in the code package corresponding to the
  assignment number, e.g. `hw1`, `hw2`, etc.
- The notebooks contain material you need to know, instructions about what to do
  and also code blocks that will **test and visualize** your implementations.
- Within the notebooks, anything you need to do is marked with a **TODO** beside
  it. It will explain what to implement and in which file.
- Within the assignment code package, all locations where you need to write code
  are marked with a special marker (`YOUR CODE`). Additionally, implementation
  guidelines, technical details and hints are in some cases provided in a
  comment above.
- Sometimes there are open questions to answer. Your answers should also be
  written within the assignment package, not within the notebook itself. The
  notebook will specify where to write each answer.

Notes:

1. You should think of the code blocks in the notebooks as tests. They test your
   solutions and they will fail if something is wrong.  As such, if you
   implement everything and the notebook runs without error, you can be
   confident about your solution.

2. Please don't put other files in the assignment directory. If you do, they
   will be added to your submission which is automatically generated from the
   contents of the assignment folder.

3. Always make sure the active conda env is `cs236781-hw`. If you get strange
   errors or broken import statements, this is probably the reason.
   Note that if you close your terminal session you will need to re-activate
   since conda will use it's default `base` environment.

## Submitting the assignment

What you'll submit:
- All notebooks, after running them clean from start to end, with all outputs
  present.
- An html file containing the merged content of all notebooks.
- The code with all your solutions present.

You don't need to do this manually - we provide you with a helper CLI program to
run all the notebooks and combine them into a single file for submission.

### Generating your submission file

To generate your submission, run (obviously with different id's):

```shell
python main.py prepare-submission --id 123456789 --id 987654321
```

The above command will:
- Execute all the notebooks cleanly, from start to end, regenerating all outputs.
- Merge the notebook contents into a single html file.
- Create a zip file with all of the above and also with your code.

If there are errors when running your notebooks, it means there's a problem with
your solution or that you forgot to implement something.

Additionally, you can use the `--skip-run` flag to skip running your notebooks
(and just merge them) in case you already ran everything and you're sure that
all outputs are present:

```shell
python main.py prepare-submission --skip-run --id 123456789 --id 987654321
```

Note however that if some of the outputs are missing from your submission you'll
lose marks.

**Note**: The submission script must also be run from within the same `conda env` as
the assignment.

### Submitting a partial solution

If you are unable to solve the entire assignment and wish to submit a partial
solution you can create a submission with errors by adding an `allow-errors`
flag, like so:

```shell
python main.py prepare-submission --allow-errors --id 123456789 --id 987654321
```

### Uploading the solution

The `.zip` file you generate should be uploaded using the assignments tab in the
[webcourse](https://webcourse.cs.technion.ac.il/236781/) system.

Grades will also be reported there.
