---
title: "Homework 2"
excerpt: Backpropagarion, optimization, training and convolutional architectures
author: aviv
published: true
date: 2018-12-01
---

**Submission date**: December ~~22th~~ 30th, 2018

## Topics

- Backpropagation on arbitrarily deep nets
- Optimization algorithms for deep learning
- Training
- Convolutional networks
- Effect of network architecture on accuracy

## Downloading

The assignment code is available
[here](https://github.com/vistalab-technion/cs236605-hw2).

We recommend you use `git` to clone the repo:
```shell
git clone https://github.com/vistalab-technion/cs236605-hw2.git
```
This will allow you to pull updates from us in the event that they are needed.

Note that there are some updates in the `environment.yml` file since the
previous assignment. From within the hw2 directory, please run

```shell
conda env update
```

To update your conda environment (only new dependencies will be installed).

## Updates

**2018-12-09**
\[[`14f9d5c`](https://github.com/vistalab-technion/cs236605-hw2/commit/14f9d5c291c53415e40713d26449cef5448152c6)\]  
In Part 2, the expression for the regularization term of the
Vanilla SGD was corrected with an additional factor of $\frac{1}{2}$ to make the
notebook consistent with the reference implementation (and the unit test).

## FAQ

Make sure to read the [getting started page]({{ site.baseurl }}{% link _assignments/getting-started.md %})
and the [guide for using course servers]({{ site.baseurl }}{% link _assignments/hpc-servers.md %}) (relevant for part 3).

**Q**: In Part 2, should the regularization term factor be $\frac{1}{2}\lambda$?  
**A**: Yes, see updates.

**Q**: Should we comment out the `raise NotImplementedError()` in the
 `YourCodeNet` class in `models.py` while working on Part 1/2?  
**A**: Yes.
