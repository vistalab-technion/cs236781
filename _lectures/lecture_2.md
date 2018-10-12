---
title: "Lecture 2: Supervised Learning"
date: 2018-10-01
excerpt: Ingredients of supervised learning, linear regression,
         binary classification, logistic regression, why learning works
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
${\bm{\mathrm{x}}} = (x_1,\dots,x_n)^\Tr$ and denote the space in which these
data reside as $\mathcal{X}$. $\mathcal{X}$ is ofter called the *input* or
*instance space*, and a point in it is called an *instance*.  We will further
define a probability measure $P({\bm{\mathrm{X}}})$ on $\mathcal{X}$ and think
of an instance as a realization of the random vector ${\bm{\mathrm{X}}}$
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
\mathcal{Y}$ assigning to each instance ${\bm{\mathrm{x}}}$ a label $y$. This
function is, obviously, unknown; yet, we can think of it as a black box into
which we can throw an instance ${\bm{\mathrm{x}}}$ and get the corresponding
label $y = f({\bm{\mathrm{x}}})$. In practice, it often happens that the target
“function” is not a function at all. For example, it might not be deterministic,
so that two customers with exactly the same profiles receive different
decisions. A way to correctly model this situation is by defining a *target
distribution* $P(Y | {\bm{\mathrm{X}}})$, a conditional probability measure
telling us how the label (a random variable) is distributed given an instance
${\bm{\mathrm{X}}} = {\bm{\mathrm{x}}}$. We can therefore think of another black
box generating pairs $({\bm{\mathrm{x}}},y)$ from the *joint* distribution
$P({\bm{\mathrm{X}}},Y) = P({\bm{\mathrm{X}}}) P(Y|{\bm{\mathrm{X}}})$.
Alternatively, we can think of a noisy target function of the form
$f({\bm{\mathrm{x}}}) = \mathbb{E} (Y|{\bm{\mathrm{X}}}={\bm{\mathrm{x}}})$ plus
noise $Y - f({\bm{\mathrm{x}}})$. The first term in this definition is
deterministic accounting for the part of $y$ that can be explained in the terms
of ${\bm{\mathrm{X}}}$, while the second term is stochastic accounting for
whatever cannot be explaind by ${\bm{\mathrm{X}}}$ (e.g., the bad mood of a
financial officer making the final decision about credit approvals).

### Training set

While the target function $f$ (alternatively, the conditional distribution
$P(Y|{\bm{\mathrm{X}}})$) is latent, we assume to be given a *finite sample*
$$\{ ({\bm{\mathrm{x}}}_i,y_i) \}_{i=1}^N$$ of labeled instances with each
${\bm{\mathrm{x}}}_i$ drawn from $P({\bm{\mathrm{X}}})$ and $y_i =
f({\bm{\mathrm{x}}}_i)$ (alternatively, $({\bm{\mathrm{x}}}_i,y_i)$ are drawn
from the joint distribution $P({\bm{\mathrm{X}}},Y) = P({\bm{\mathrm{X}}})
P(Y|{\bm{\mathrm{X}}})$).

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
h({\bm{\mathrm{x}}} = {\bm{\mathrm{w}}}^\Tr {\bm{\mathrm{x}}} + w_0 :
{\bm{\mathrm{w}}} \in \mathbb{R}^{n+1}  \}.
$$

A concrete choice of the weights ${\bm{\mathrm{w}}}$ gives a specific hypothesis
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
$\hat{y}=h({\bm{\mathrm{x}}})$ of an instance ${\bm{\mathrm{x}}}$ and its true
label given by the target function $y=f({\bm{\mathrm{x}}})$. For example, in
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
L_{\mathrm{in}}(h) =  \frac{1}{N} \sum_{i=1}^N \ell( h({\bm{\mathrm{x}}}_i),
f({\bm{\mathrm{x}}}_i) ) =  \frac{1}{N} \sum_{i=1}^N \ell(
h({\bm{\mathrm{x}}}_i), y_i ),
$$

where we omitted the dependence on $f$ for convenience. Note that the loss only
depends on the instances in the training set and their corresponding labels.
This is the only type of loss that we can actually compute without knowing the
target function! So let us state our learning problem more precisely: in
consists of minimizing the in-sample loss,

$$
h^\ast = \mathrm{arg} \min_{h \in \mathcal{H}} L_{\mathrm{in}}(h) =
\mathrm{arg} \min_{h \in \mathcal{H}} \frac{1}{N} \sum_{i=1}^N \ell(
h({\bm{\mathrm{x}}}_i), y_i).
$$

## Simple examples

In what follows, we will examine several very simple examples of supervised
learning based on the *linear* hypothesis class. To simplify notation, we will
assume that every instance has an additional dimension $x_0 = 1$, such that the
affine term in ${\bm{\mathrm{w}}}^\Tr {\bm{\mathrm{x}}} + w_0$ can be absorbed
into the inner product (in some literature this trick goes by the name of
*homogeneous coordinates*). In this notation, the linear class is defined as

$$
\mathcal{H} = \{ h : \mathcal{X} \rightarrow \mathcal{Y} \, | \, h({\bm{\mathrm{x}}} = {\bm{\mathrm{w}}}^\Tr {\bm{\mathrm{x}}} : {\bm{\mathrm{w}}} \in \mathbb{R}^{n+1}  \}.
$$

In binary classification, this class of models in known as *linear perceptron*.

### Linear regression

Let us examine first the regression problem using the linear regressor
$h({\bm{\mathrm{x}}}) = {\bm{\mathrm{w}}}^\Tr {\bm{\mathrm{x}}}$.  Assuming the
squared error loss, we obtain

$$
L_{\mathrm{in}} =  \frac{1}{N} \sum_{i=1}^N (
h({\bm{\mathrm{x}}}_i) - y_i )^2 = \frac{1}{N} \sum_{i=1}^N (
{\bm{\mathrm{w}}}^\Tr {\bm{\mathrm{x}}}_i - y_i )^2.
$$

Arranging the training instances into the columns of an $(n+1) \times N$ matrix
${\bm{\mathrm{X}}}$ and the training labels into an $N$-dimensional vector
${\bm{\mathrm{y}}}$ yields

$$
L_{\mathrm{in}} =  \frac{1}{N} \|  {\bm{\mathrm{X}}}^\Tr {\bm{\mathrm{w}}} -
{\bm{\mathrm{y}}} \|^2.
$$

Differentiating w.r.t. ${\bm{\mathrm{w}}}$ and requiring vanishing gradient
yields

$$
0 = {\bm{\mathrm{X}}}
({\bm{\mathrm{X}}}^\Tr {\bm{\mathrm{w}}} - {\bm{\mathrm{y}}}) =
{\bm{\mathrm{X}}} {\bm{\mathrm{X}}}^\Tr {\bm{\mathrm{w}}} - {\bm{\mathrm{X}}}
{\bm{\mathrm{y}}},
$$

from where

$$
{\bm{\mathrm{w}}}^\ast =(
{\bm{\mathrm{X}}}^\Tr)^\dagger {\bm{\mathrm{y}}} =
({\bm{\mathrm{X}}}{\bm{\mathrm{X}}}^\Tr )^{-1} {\bm{\mathrm{X}}}
{\bm{\mathrm{y}}}.
$$

### Linear binary classification

Linear classification is very similar to linear regression, with the exception
that a classifier only retains the sign of the linear function
$h({\bm{\mathrm{x}}}) = \mathrm{sign}({\bm{\mathrm{w}}}^\Tr {\bm{\mathrm{x}}})$.
Geometrically, this corresponds to splitting the space into two regions by a
hyperplane whose normal is defined by $(w_1,\dots,w_n)$. We could naively ignore
the sign function and learn the classifier using the squared error applied to
$y_i \in  \{\pm 1\}$. However, because of the linear form of the function under
the sign, this loss will artificially penalize correcy hypotheses at point
distant from the decision boundary.

### Logistic regression

A better alternative is to model directly the conditional probability
$P(Y|{\bm{\mathrm{X}}})$ as

$$
P(Y=1|{\bm{\mathrm{X}}}={\bm{\mathrm{x}}}) = \frac{1}{1+e^{ {\bm{\mathrm{w}}}^\Tr {\bm{\mathrm{x}}}}}
$$

and

$$
P(Y=0|{\bm{\mathrm{X}}}={\bm{\mathrm{x}}}) = 1- P(Y=1|{\bm{\mathrm{X}}}={\bm{\mathrm{x}}}) = \frac{e^{ {\bm{\mathrm{w}}}^\Tr {\bm{\mathrm{x}}}} }{1+e^{ {\bm{\mathrm{w}}}^\Tr {\bm{\mathrm{x}}}}}
$$

Label $y=0$ is assigned if

$$
\frac{ P(Y=0|{\bm{\mathrm{X}}}={\bm{\mathrm{x}}}) }{ P(Y=1|{\bm{\mathrm{X}}}={\bm{\mathrm{x}}}) } = e^{ {\bm{\mathrm{w}}}^\Tr {\bm{\mathrm{x}}}} > 1,
$$

which is equivalent to
$e^{ {\bm{\mathrm{w}}}^\Tr {\bm{\mathrm{x}}}} > 0$. The label $y=1$ is
assigned otherwise.

In order to solve the regression problem, we maximize the likelihood of the
observed labels in the training data given the corresponding instances,

$$
{\bm{\mathrm{w}}}^\ast = \mathrm{arg}\max_{ {\bm{\mathrm{w}}}} \prod_{i=1}^n P(y_i | {\bm{\mathrm{x}}}_i) = \mathrm{arg}\min_{ {\bm{\mathrm{w}}}} \sum_{i=1}^n -\log P(y_i | {\bm{\mathrm{x}}}_i)
$$

The negative likelihood can be written as the loss function

$$
L({\bm{\mathrm{w}}}) = \sum_{i=1}^n -\log P(y_i | {\bm{\mathrm{x}}}_i)  = \sum_{i=1}^n -y_i \log P(y_i=1 | {\bm{\mathrm{x}}}_i) - (1-y_i) \log P(y_i=0 | {\bm{\mathrm{x}}}_i),
$$

utilizing the fact that $y_i$ can only assume binary values.  Substituting the
linear hypothesis yields

$$
L({\bm{\mathrm{w}}}) =   \sum_{i=1}^n(y_i-1){\bm{\mathrm{w}}}^\Tr{\bm{\mathrm{x}}}_i + \log ( 1+e^{ {\bm{\mathrm{w}}}^\Tr {\bm{\mathrm{x}}}_i} ).
$$

Note that if training labels are expressed as delta-distributions, the logistic
regression loss is nothing but the cross-entropy between the true labels and the
logistic model for $P(Y|{\bm{\mathrm{X}}})$.

While the minimizer of this loss does not admit a closed-form expression, it can
be found using an iterative solver. The convexity of the loss warrants global
convergence.

### Non-linear data transformations vs. non-linear models

The utility of linear models is limited since they can only express linear
decision boundaries. However, by transforming the instance data by some
non-linear map $\Phi$ and applying a linear model (regressor or classifier) on
the obtained feature space can create arbitrarily complex decision boundaries.
Note that while a regressor of the form $h({\bm{\mathrm{x}}}) =
{\bm{\mathrm{w}}}^\Tr \Phi({\bm{\mathrm{x}}})$ is non-linear in
${\bm{\mathrm{x}}}$, it is still linear in ${\bm{\mathrm{w}}}$ and, therefore,
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
table and whenever $h$ is queried with the instance ${\bm{\mathrm{x}}}\_i$, it
will return the memorized $y_i$.  Obviously, such a hypothesis will result in
$L_{\mathrm{in}} = 0$, but will be completely useless on previously unseen
instances.

### Generalization error

In order to perform well on unseen data, the hypothesis should to *generalize*
over the underlying probability distribution. This can be formalized by defining
the *generalization error* (aka *out-of-sample loss*) as

$$
L_{\mathrm{out}}(h)  = \mathbb{E} \, \ell( h({\bm{\mathrm{X}}}), f({\bm{\mathrm{X}}}) ),
$$

where the expectation is taken over $P({\bm{\mathrm{X}}})$, or,
alternatively, in terms of the joint distribution,

$$
L_{\mathrm{out}}(h)  = \mathbb{E} \, \ell( h({\bm{\mathrm{X}}}), Y ),
$$

where the expectation is over $P({\bm{\mathrm{X}}},Y)$.

In order to generalize well, our learning process should pick up
$h \in \mathcal{H}$ with the smallest out-of-sample loss. However, we
cannot really solve this problem, since we cannot compute
$L_{\mathrm{out}}$ as $P({\bm{\mathrm{X}}},Y)$ is unknown.

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

