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

## Logging in 

Logging in is performed with your Technion Single Sign-On (SSO) credentials.
Usually this means the username and password of your `@campus` or `@technion`
email address.

If your username is e.g. `user`, login like so

```shell
ssh user@rishon.technion.ac.il
```

### Connecting with pubic-key based authentication

If you wish to not use a password to connect every time, you can use a public-key based authentication.

1. Generate an SSH key pair using the `ssh-keygen` tool. More detailed instructions for all platforms can be found
   [here](https://help.github.com/articles/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent/).
1. Copy the **public** key. By default it's in `~/.ssh/id_rsa.pub`. Make sure you copy it exactly
   without any extra spaces or newlines.
1. Connect to your user on the machine and paste the public key contents into a new line in `~/.ssh/authorized_keys`.

Notes:

1. On macOS and linux, there's a utility you can use to do automate steps 2-3.
   After generating the key pair, copy the public key to the server like so:
   ```shell
   ssh-copy-id user@rishon.cs.technion.ac.il
   ```
1. If you use an intermediate server to connect from home, make sure to first
   also copy your public key to that server.


### Connecting from home

The `rishon` server is only accessible from within the Technion networks.
If you need to connect from home, first SSH into a Technion server that’s
accessible from the outside (e.g. CSM) and from there you can SSH into `rishon`. 

You can do this in one command like so:

```shell
ssh -J user@csm.cs.technion.ac.il user@rishon.cs.technion.ac.il
```

This example will connect through the CSM server in the CS faculty. You should
be able to use any other Technion server that you have SSH access to.

Note that this method (`-J`) has the useful advantage of forwarding the SSH public key
from your local machine (if available) to the target machine though the
intermediate machine.

## Usage

Instructions on how to run jobs on the server will be added soon.

