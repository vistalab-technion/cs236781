---
title: "Lecture 2: Supervised Learning"
date: 2018-10-01
excerpt: Ingredients of supervised learning, linear regression,
         binary classification, logistic regression, why learning works
author: alex
---

The goal of this lecture is to formalize the supervised regime of learning
problems, which is by far the most common used type of learning.
Let us start with a concrete learning problem that we will use as an
illustration throughout this lecture. A financial institution issues credit
cards to its customers. Each customer has a profile with a bunch of numeric
values indicating, e.g., their salary, total outstanding debt, credit score,
years in the last residence, etc.  Based on this information, the institution
has to decide whether to approve the credit and decide on the size of the credit
line. The former problem of assigning each customer a discrete decision (yes/no
in this case) is known as *classification*. The latter problem of assigning a
continuous value (dollar amount) is called *regression*.

In what follows, we formalize the supervised learning problem.

## Ingredients of Supervised Learning

### Instance space

Let us represent the user data as an $n$-dimensional column vector
$\bb{x} = (x_1,\dots,x_n)^\Tr$ and denote the space in which these
data reside as $\mathcal{X}$. $\mathcal{X}$ is ofter called the *input* or
*instance space*, and a point in it is called an *instance*.  We will further
define a probability measure $P(\bb{X})$ on $\mathcal{X}$ and think
of an instance as a realization of the random vector $\bb{X}$
distributed with $P$.

### Label space

Let us denote by $\mathcal{Y}$ the *target* or *label space* in which the
possible decisions about a customer reside. For example, in the case of our
binary classification, $\mathcal{Y}$ is simply $\{0,1\}$; in the case of
regression, this may be $\mathcal{Y}=[0,\infty)$. In general, this space may
contain continuous and vector-valued objects. We will still refer to a point in
$\mathcal{Y}$ as to a *label* even in the case of regression problems.

### Target function

We assume that there exists some *target function* $f : \mathcal{X} \rightarrow
\mathcal{Y}$ assigning to each instance $\bb{x}$ a label $y$. This
function is, obviously, unknown; yet, we can think of it as a black box into
which we can throw an instance $\bb{x}$ and get the corresponding
label $y = f(\bb{x})$. In practice, it often happens that the target
“function” is not a function at all. For example, it might not be deterministic,
so that two customers with exactly the same profiles receive different
decisions. A way to correctly model this situation is by defining a *target
distribution* $P(Y | \bb{X})$, a conditional probability measure
telling us how the label (a random variable) is distributed given an instance
$\bb{X} = \bb{x}$. We can therefore think of another black
box generating pairs $(\bb{x},y)$ from the *joint* distribution
$P(\bb{X},Y) = P(\bb{X}) P(Y|\bb{X})$.
Alternatively, we can think of a noisy target function of the form
$f(\bb{x}) = \mathbb{E} (Y|\bb{X}=\bb{x})$ plus
noise $Y - f(\bb{x})$. The first term in this definition is
deterministic accounting for the part of $y$ that can be explained in the terms
of $\bb{X}$, while the second term is stochastic accounting for
whatever cannot be explaind by $\bb{X}$ (e.g., the bad mood of a
financial officer making the final decision about credit approvals).

### Training set

While the target function $f$ (alternatively, the conditional distribution
$P(Y|\bb{X})$) is latent, we assume to be given a *finite sample*

$$\{ (\bb{x}_i,y_i) \}_{i=1}^N$$

of labeled instances with each
$\bb{x}_i$ drawn from $P(\bb{X})$ and $y_i =
f(\bb{x}_i)$ (alternatively, $(\bb{x}_i,y_i)$ are drawn
from the joint distribution $P(\bb{X},Y) = P(\bb{X})
P(Y|\bb{X})$).

**The goal of supervised learning is to estimate the target function (or the
target distribution) from the training examples.**

### Hypothesis class

Since the target function can be arbitrarily complex, in order to make the
learning problem manageable, we will restrict our estimates to some (usually,
parametric) family of functions which we will refer to as a *hypothesis class*
$\mathcal{H}$. A function $h \in \mathcal{H}$ is a map $h : \mathcal{X}
\rightarrow \mathcal{Y}$ assigning each instance in $\mathcal{X}$ a label in
$\mathcal{Y}$.

For example, the class of *linear* hypotheses that we will encounter in the
sequel is defined as

$$
\mathcal{H} = \{ h : \mathcal{X} \rightarrow \mathcal{Y} \, | \,
h(\bb{x} = \bb{w}^\Tr \bb{x} + w_0 :
\bb{w} \in \mathbb{R}^{n+1}  \}.
$$

A concrete choice of the weights $\bb{w}$ gives a specific hypothesis
from the class $\mathcal{H}$.

### Loss functions

The task of supervised learning consists therefore of picking a single
hypothesis function from $\mathcal{H}$ that best estimates the target function
$f$. But "best" in what sense? A very standard way of quantifying the quality of
the fit is by defining a functional of the form $L(h,f)$ accepting a hypothesis
$h$ and the target function $f$ and returning a numerical value indicating the
deviation of $h$ from $f$.  $L$ is usually referred to as the *loss*, *risk* or
*cost* function and should be as small as possible. In these terms, the task of
machine learning can be formulated as the minimization of the loss over the
hypothesis class,

$$
h^\ast = \mathrm{arg} \min_{h \in \mathcal{H}} L(h,f).
$$

Loss functions are (almost) always constructed from a pointwise definition,
i.e., we actually define $\ell(\hat{y},y)$ accepting the estimated label
$\hat{y}=h(\bb{x})$ of an instance $\bb{x}$ and its true
label given by the target function $y=f(\bb{x})$. For example, in
regression problems the squared error

$$
\ell( \hat{y},y ) = ( \hat{y} - y )^2
$$

is often used as the loss, while in binary classification, the choice could be
the binary error

$$
\ell( \hat{y},y ) = [ \hat{y} \ne  y],
$$

with $[a]$ denoting the indicator function of the condition $a$ accepting the
value of $1$ when $a$ is satisfied and $0$ otherwise.

As a general rule and a matter of good practice, the choice of the loss function
is application-specific and should be provided by the user. In the real world,
however, this rarely happens. Two alternatives are possible as the second
choice. The first one is to use a *plausible* loss function that has a merit and
can be rigorously defined asserting a certain assumption (the validity of which
can be typically debated ad infinitum). For example, the assumption of white
additive Gaussian noise leads to the squared error. While the latter fact can be
proved mathematically, the former assumption is rarely perfectly (or at all)
correct. Similarly, the cross-entropy error can plausibly replace the binary
error.

The second alternative is to *pragmatically* choose a loss function not based on
some particular merit or justification, but rather because it is easy to use.
For example, the squared error in combination with the linear hypothesis class
leads to a simple closed form expression; in the same setting, the cross-entropy
loss, while not leading to a closed-form expression yields a convex objective
that can be minimized globally.

Just to reiterate: the choice of the loss function is crucial to the success of
learning and should be done with as much domain-specific understanding of the
problem as possible.

#### In-sample loss

Once the point-wise loss function has been fixed, we can define the overall loss
$L$ by averaging the pointwise losses. Averaging on the training set leads to
the *empirical* or *in-sample loss*

$$
L_{\mathrm{in}}(h) =  \frac{1}{N} \sum_{i=1}^N \ell( h(\bb{x}_i),
f(\bb{x}_i) ) =  \frac{1}{N} \sum_{i=1}^N \ell(
h(\bb{x}_i), y_i ),
$$

where we omitted the dependence on $f$ for convenience. Note that the loss only
depends on the instances in the training set and their corresponding labels.
This is the only type of loss that we can actually compute without knowing the
target function! So let us state our learning problem more precisely: in
consists of minimizing the in-sample loss,

$$
h^\ast = \mathrm{arg} \min_{h \in \mathcal{H}} L_{\mathrm{in}}(h) =
\mathrm{arg} \min_{h \in \mathcal{H}} \frac{1}{N} \sum_{i=1}^N \ell(
h(\bb{x}_i), y_i).
$$

## Simple examples

In what follows, we will examine several very simple examples of supervised
learning based on the *linear* hypothesis class. To simplify notation, we will
assume that every instance has an additional dimension $x_0 = 1$, such that the
affine term in $\bb{w}^\Tr \bb{x} + w_0$ can be absorbed
into the inner product (in some literature this trick goes by the name of
*homogeneous coordinates*). In this notation, the linear class is defined as

$$
\mathcal{H} = \{ h : \mathcal{X} \rightarrow \mathcal{Y} \, | \, h(\bb{x} = \bb{w}^\Tr \bb{x} : \bb{w} \in \mathbb{R}^{n+1}  \}.
$$

In binary classification, this class of models in known as *linear perceptron*.

### Linear regression

Let us examine first the regression problem using the linear regressor
$h(\bb{x}) = \bb{w}^\Tr \bb{x}$.  Assuming the
squared error loss, we obtain

$$
L_{\mathrm{in}} =  \frac{1}{N} \sum_{i=1}^N (
h(\bb{x}_i) - y_i )^2 = \frac{1}{N} \sum_{i=1}^N (
\bb{w}^\Tr \bb{x}_i - y_i )^2.
$$

Arranging the training instances into the columns of an $(n+1) \times N$ matrix
$\bb{X}$ and the training labels into an $N$-dimensional vector
$\bb{y}$ yields

$$
L_{\mathrm{in}} =  \frac{1}{N} \|  \bb{X}^\Tr \bb{w} -
\bb{y} \|^2.
$$

Differentiating w.r.t. $\bb{w}$ and requiring vanishing gradient
yields

$$
0 = \bb{X}
(\bb{X}^\Tr \bb{w} - \bb{y}) =
\bb{X} \bb{X}^\Tr \bb{w} - \bb{X}
\bb{y},
$$

from where

$$
\bb{w}^\ast =(
\bb{X}^\Tr)^\dagger \bb{y} =
(\bb{X}\bb{X}^\Tr )^{-1} \bb{X}
\bb{y}.
$$

### Linear binary classification

Linear classification is very similar to linear regression, with the exception
that a classifier only retains the sign of the linear function
$h(\bb{x}) = \mathrm{sign}(\bb{w}^\Tr \bb{x})$.
Geometrically, this corresponds to splitting the space into two regions by a
hyperplane whose normal is defined by $(w_1,\dots,w_n)$. We could naively ignore
the sign function and learn the classifier using the squared error applied to
$y_i \in  \{\pm 1\}$. However, because of the linear form of the function under
the sign, this loss will artificially penalize correcy hypotheses at point
distant from the decision boundary.

### Logistic regression

A better alternative is to model directly the conditional probability
$P(Y|\bb{X})$ as

$$
P(Y=1|\bb{X}=\bb{x}) = \frac{1}{1+e^{ \bb{w}^\Tr \bb{x}}}
$$

and

$$
P(Y=0|\bb{X}=\bb{x}) = 1- P(Y=1|\bb{X}=\bb{x}) = \frac{e^{ \bb{w}^\Tr \bb{x}} }{1+e^{ \bb{w}^\Tr \bb{x}}}
$$

Label $y=0$ is assigned if

$$
\frac{ P(Y=0|\bb{X}=\bb{x}) }{ P(Y=1|\bb{X}=\bb{x}) } = e^{ \bb{w}^\Tr \bb{x}} > 1,
$$

which is equivalent to
$e^{ \bb{w}^\Tr \bb{x}} > 0$. The label $y=1$ is
assigned otherwise.

In order to solve the regression problem, we maximize the likelihood of the
observed labels in the training data given the corresponding instances,

$$
\bb{w}^\ast = \mathrm{arg}\max_{ \bb{w}} \prod_{i=1}^n P(y_i | \bb{x}_i) = \mathrm{arg}\min_{ \bb{w}} \sum_{i=1}^n -\log P(y_i | \bb{x}_i)
$$

The negative likelihood can be written as the loss function

$$
L(\bb{w}) = \sum_{i=1}^n -\log P(y_i | \bb{x}_i)  = \sum_{i=1}^n -y_i \log P(y_i=1 | \bb{x}_i) - (1-y_i) \log P(y_i=0 | \bb{x}_i),
$$

utilizing the fact that $y_i$ can only assume binary values.  Substituting the
linear hypothesis yields

$$
L(\bb{w}) =   \sum_{i=1}^n(y_i-1)\bb{w}^\Tr\bb{x}_i + \log ( 1+e^{ \bb{w}^\Tr \bb{x}_i} ).
$$

Note that if training labels are expressed as delta-distributions, the logistic
regression loss is nothing but the cross-entropy between the true labels and the
logistic model for $P(Y|\bb{X})$.

While the minimizer of this loss does not admit a closed-form expression, it can
be found using an iterative solver. The convexity of the loss warrants global
convergence.

### Non-linear data transformations vs. non-linear models

The utility of linear models is limited since they can only express linear
decision boundaries. However, by transforming the instance data by some
non-linear map $\Phi$ and applying a linear model (regressor or classifier) on
the obtained feature space can create arbitrarily complex decision boundaries.
Note that while a regressor of the form $h(\bb{x}) =
\bb{w}^\Tr \Phi(\bb{x})$ is non-linear in
$\bb{x}$, it is still linear in $\bb{w}$ and, therefore,
can be solved for as before.

However, while non-linear data transformations allow designing complex
decision boundaries, the choice of the map $\Phi$ is hand-crafted rather
than learned. Alternatively, a class of parametric hypothesis functions
that depend *non-linearly* on the parameters allow to learn the features
themselves direclty from the training examples. In this course, such
non-linear models will be deep neural networks.

## Why does learning work?

Why does learning work at all? What prevents us from choosing a sufficiently
rich hypothesis class that will simply memorize each training sample in a huge
table and whenever $h$ is queried with the instance $\bb{x}\_i$, it
will return the memorized $y_i$.  Obviously, such a hypothesis will result in
$L_{\mathrm{in}} = 0$, but will be completely useless on previously unseen
instances.

### Generalization error

In order to perform well on unseen data, the hypothesis should to *generalize*
over the underlying probability distribution. This can be formalized by defining
the *generalization error* (aka *out-of-sample loss*) as

$$
L_{\mathrm{out}}(h)  = \mathbb{E} \, \ell( h(\bb{X}), f(\bb{X}) ),
$$

where the expectation is taken over $P(\bb{X})$, or,
alternatively, in terms of the joint distribution,

$$
L_{\mathrm{out}}(h)  = \mathbb{E} \, \ell( h(\bb{X}), Y ),
$$

where the expectation is over $P(\bb{X},Y)$.

In order to generalize well, our learning process should pick up
$h \in \mathcal{H}$ with the smallest out-of-sample loss. However, we
cannot really solve this problem, since we cannot compute
$L_{\mathrm{out}}$ as $P(\bb{X},Y)$ is unknown.

### Hoeffding inequality

Is the learning problem solvable at all? We are stuck with the
minimization of the empirical (in-sample) loss, while we want to
minimize the out-of-sample loss. Can we say anything useful about the
*generalization gap*, i.e., the difference between the generalization
error and the training error? Note that the difference between
$L_{\mathrm{in}}$ and $L_{\mathrm{out}}$ is that while in the former we
use a finite-sample average, the latter is defined with a true
expectation. Given a hypothesis, the in-sample loss is a random variable
(each realization of the sample will give it a different value), while
the out-of-sample loss is a deterministic quantity.

If we assume that the training samples are drawn i.i.d. from the
underlying distribution, by the law of large number the finite-sample
average will asymptotically concentrate about the expected value. One of
the strong forms of the law of large numbers is known as the *Hoeffding
inequality* setting a bound on the probability of the finite-sample
average deviating from the expected value by more than $\epsilon$. In
our terms, *for a given hypothesis*, it can be expressed

$$
\mathbb{P}( | L_{\mathrm{out}}(h) - L_{\mathrm{in}}(h) | > \epsilon) \le 2 e^{-2\epsilon^2 N }.
$$

This concentration inequality states that
$L_{\mathrm{out}}(h) = L_{\mathrm{in}}(h)$ is *probably approximately
correct* (PAC). *Probably* in the sense that it can be violated with a
negligibly small probability (that decays exponentially in the sample
size $N$); *approximately* in the sense that $L_{\mathrm{out}}(h)$ is
allowed to deviate from $L_{\mathrm{in}}(h)$ by a specified tolerance
$\epsilon$. The smaller is this constant, the more training samples are
required to maintain the same level of certainty about the result. For
example, decreasing $\epsilon$ by $10$ times requires to increase $N$ by
$100$ times.

### A naïve generalization bound

The bound we derived so far applies to a single hypothesis. So it is
useful to verfiy whether a given hypothesis will generalize well.
However, recall that the learning process actually involves a search
over many hypotheses. In order to accomodate for this, let us derive the
following worst-case bound

$$
\mathbb{P}( | L_{\mathrm{out}}(h^\ast) - L_{\mathrm{in}}(h^\ast) | > \epsilon) \le \mathbb{P}( \sup_{h \in \mathcal{H} } | L_{\mathrm{out}}(h) - L_{\mathrm{in}}(h) | > \epsilon) = \mathbb{P}\left( \bigcup_{h \in \mathcal{H} } \{ | L_{\mathrm{out}}(h) - L_{\mathrm{in}}(h) | > \epsilon \} \right).
$$

Using the union bound inequality yields

$$
\mathbb{P}( | L_{\mathrm{out}}(h^\ast) - L_{\mathrm{in}}(h^\ast) | > \epsilon) \le \sum_{ h \in \mathcal{H}  } \mathbb{P}\left(  | L_{\mathrm{out}}(h) - L_{\mathrm{in}}(h) | > \epsilon \right) \le  2 | \mathcal{H} | e^{-2\epsilon^2 N }.
$$


However, using the cardinality of $\mathcal{H}$ is a very lousy idea.
First of all, it clearly fails even for such a simple infinite class of
hypotheses as the linear class we discussed earlier. This happens
because given two hypotheses $h_1,h_2 \in \mathcal{H}$, we treat the
events $| L_{\mathrm{out}}(h_1) - L_{\mathrm{in}}(h_1) | > \epsilon$ and
$| L_{\mathrm{out}}(h_2) - L_{\mathrm{in}}(h_2) | > \epsilon$ as
*independent*, while, clearly, the dependence of these events should
somehow depend on how $h_1$ and $h_2$ are close to each other in some
sense.

### Hypothesis class complexity

More delicate tools exist to get a better grasp of the complexity (or
*capacity* of the hypothesis class, leading to tighter and more useful
generalization error bounds. The first such analytic tool derived in the
70’s was the Vapnik–Chervonenkis (VC) dimension. A more modern and
powerful tool is the Rademacher complexity that measures how well a
hypothesis class fits random noise.

We will not get into the details of these tools, for the mere reason
that they are not very helpful when applied to deep learning.
Practically useful deep neural networks architectures have a huge VC
capacity, yet practice shows that they can generalize really well when
trained on comparatively tiny training sets (compared to the size that
would satisfy the bounds). Currently, there is no theoretical answer
explaining the successful generalization of DNNs, and there is strong
evidence ruling out the classical explanation based on the VC dimension
(and similar reasoning). For example, the effects of explicit
regularization (weight decay, dropout, data augmentation) as well as
implicit regularization produced by the use of stochastic gradient
descent are presently poorly understood.

