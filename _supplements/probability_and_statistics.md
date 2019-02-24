---
title: "Probability and statistics: a survival guide"
excerpt: "Random variables and vectors, statistical estimation"
author: alex
copyright: alex
---

Random variables
================

Probability measure
-------------------

We start with a few elementary (and simplified) definitions from the
theory of probability. Let us fix a *sample space* $\Omega = [0,1]$. A
*Borel set* on $\Omega$ is a set that can be formed from open intervals
of the form $(a,b), 0 \le a<b \le 1$, through the operations of
countable union, countable intersection, and set difference. We will
denote the collection of all Borel sets in $\Omega$ as $\Sigma$. It is
pretty straightforward to show that $\Sigma$ contains the empty set, is
closed under complement, and is closed under countable union. Such a set
is known as *$\sigma$-algebra* and its elements (subsets of
$\mathbb{R}$) are referred to as *events*.

A *probability measure* $P$ on $\Sigma$ is a function
$P : \Sigma \rightarrow [0,1]$ satisfying $P(\emptyset) = 0$,
$P(\mathbb{R}) = 1$ and additivity for every countable collection
$\{ E _n \in \Sigma \}$,


$$P\left(  \bigcup _n E _n \right) = \sum _{n} P(E _n).$$



Random variables
----------------

A *random variable* $\mathpzc{X}$ is a *measurable* map
$\mathpzc{X} : \Omega \rightarrow \mathbb{R}$, i.e., a function such
that for every $a$,
$\{ \mathpzc{X} \le a \} = \{ \alpha : \mathpzc{X}(\alpha) \le a  \} \in \Sigma$.
The map $\mathpzc{X}$ pushes forward the probability measure $P$; the
*pushforward measure* $\mathpzc{X} _\ast P$ is given by


$$(\mathpzc{X} _\ast P)(A) = P(\mathpzc{X}^{-1}(A)),$$

where
$\mathpzc{X}^{-1}(A) = \{ \alpha  : X(\alpha) \in A \}$ is the preimage
of $A \subseteq \mathbb{R}$. (In short, we can write
$\mathpzc{X} _\ast P = P\mathpzc{X}^{-1}$). This pushforward probability
measure $\mathpzc{X} _\ast P$ is usually referred to as the *probability
distribution* (or the *law*) of $\mathpzc{X}$.

When the range of $\mathpzc{X}$ is finite or countably infinite, the
random variable is called *discrete* and its distribution can be
described by the *probability mass function* (PMF):


$$f _{\mathpzc{X}}(x) = P(\mathpzc{X}=x),$$

which is a shorthand for
$P(  \{\alpha : \mathpzc{X}(\alpha) = x \} )$. Otherwise, $\mathpzc{X}$
is called a *continuous* random variable. Any random variable can be
described by the *cumulative distribution function* (CDF)


$$F _{\mathpzc{X}}(x) = P({\mathpzc{X} \le x}),$$

which is a shorthand
for
$F _{\mathpzc{X}}(x) = P(  \{\alpha : \mathpzc{X}(\alpha) \le x \} )$. If
$X$ is absolutely continuous, the CDF can be described by the integral


$$F _{\mathpzc{X}}(x) = \int _{-\infty}^x f _{\mathpzc{X}}(x') dx',$$

where
the integrand $f _{\mathpzc{X}}$ is known as the *probability density
function* (PDF)[^1].

Uniform distribution and uniformization
---------------------------------------

A random variable $\mathpzc{U}$ is said to be *uniformly distributed* on
$[0,1]$ (denoted as $\mathpzc{U} \sim \mathcal{U}[0,1]$) if


$$P(\mathpzc{U} \in [a,b]) = b-a = \lambda([a,b]).$$

In other words, the
map $\mathpzc{U}$ pushes forward the standard Lebesgue measure on
$[0,1]$, $\mathpzc{U} _\ast P = \lambda$. The corresponding CDF is
$F _\mathpzc{U}(u) = \max\{ 0, \min\{ 1, u \} \}$. Let $\mathpzc{X}$ be
some other random variable characterized by the CDF $F _\mathpzc{X}$. We
define $\mathpzc{U} = F _\mathpzc{X}(\mathpzc{X})$. Let us pick an
arbitrary $x \in \mathbb{R}$ and let $u = F _\mathpzc{X}(x) \in [0,1]$.
From monotonicity of the CDF, it follows that $\mathpzc{U} \le u$ if and
only if $\mathpzc{X} \le x$. Hence,
$F _\mathpzc{U}(u) = P(\mathpzc{U} \le u) = P(\mathpzc{X} \le x) = F _\mathpzc{X}(x) = u$.
We conclude that by transforming a random variable with its own CDF
uniformizes it on the interval $[0,1]$.

Applying the relation in inverse direction, let
$\mathpzc{U} \sim \mathcal{U}[0,1]$ and let $F$ be a valid CDF. Then,
the random variable $\mathpzc{X} = F^{-1}(\mathpzc{U})$ is distributed
with the CDF $F _\mathpzc{U} = F$.

Expectation
-----------

The *expected value* (a.k.a. the *expectation* or *mean*) of a random
variable $\mathpzc{X}$ is given by


$$\mathbb{E} \mathpzc{X} = \int _{\mathbb{R}} \mathrm{id}\, d(\mathpzc{X} _\ast P) =  \int _{\Omega} \mathpzc{X}(\alpha) d\alpha,$$


where the integral is the Lebesgue integral w.r.t. the measure $P$;
whenever a probability density function exists, the latter can be
written as


$$\mathbb{E} \mathpzc{X} = \int _{\mathbb{R}} x  f _{\mathpzc{X}}(x) dx.$$


Note that due to the linearity of integration, the expectation operator
$\mathbb{E}$ is linear. Using the Lebesgue integral notation, we can
write for $E \in \Sigma$


$$P(E) = \int _E dP = \int _\mathbb{R} \ind _E \, dP = \mathbb{E}  \ind _E,$$


where

$$
\ind _E(\alpha) = \left\{
 \begin{array}{ccc}
 1 & : & \alpha \in E \\
 0 & : & \mathrm{otherwise}
 \end{array}
   \right.
$$

is the *indicator function* of $E$, which is by itself a
random variable. This relates the expectation of the indicator of an
event to its probability.

Moments
-------

For any measurable function $g : \mathbb{R} \rightarrow \mathbb{R}$,
$\mathpzc{Z} = g(\mathpzc{X})$ is also a random variable with the
expectation


$$\mathbb{E} \mathpzc{Z} = \mathbb{E} g(\mathpzc{X}) =  \int _{\mathbb{R}} g\, dP =  \int _{\mathbb{R}} g(x) f _{\mathpzc{X}}(x) dx.$$


Such an expectation is called a *moment* of $\mathpzc{X}$. Particularly,
the $k$-th order moment is obtained by setting $g(x) = x^k$,


$$\mu _{k}(\mathpzc{X}) = \mathbb{E} \mathpzc{X}^k.$$

The expected value
itself is the first-order moment of $\mathpzc{X}$, which is often
denoted simply as $\mu _\mathpzc{X} = \mu _{1}(\mathpzc{X})$. The
*central* $k$-th order moment is obtained by setting
$g(x) = (x - \mu _\mathpzc{X})^k$,


$$m _{k}(\mathpzc{X}) = \mathbb{E} ( \mathpzc{X}  - \mathbb{E}  \mathpzc{X})^k.$$


A particularly important central second-order moment is the *variance*


$$\sigma _\mathpzc{X}^2 = \mathrm{Var}\, \mathpzc{X} = m _2 = \mathbb{E} ( \mathpzc{X}  - \mathbb{E}  \mathpzc{X})^2 = \mu _2  ( \mathpzc{X} ) - \mu^2 _\mathpzc{X}.$$



Random vectors
==============

Joint and marginal distributions
--------------------------------

A vector $\mathpzcb{X} = (\mathpzc{X} _1, \dots, \mathpzc{X} _n)$ of
random variables is called a *random vector*. Its probability
distribution is defined as before as the pushforward measure
$P = \mathpzcb{X} _\ast \lambda$ Its is customary to treat $\mathpzcb{X}$
as a collection of $n$ random variables and define their *joint CDF* as

$$F _{\mathpzcb{X}}(\bb{x}) = P({\mathpzcb{X} \le \bb{x}}) = P(\mathpzc{X} _1 \le x _1, \dots, \mathpzc{X} _n \le x _n) 
= P(\{ \mathpzc{X} _1 \le x _1 \} \times  \dots \times \{ \mathpzc{X} _n \le x _n \}).$$

As before, whenever the following holds


$$F _{\mathpzcb{X}}(\bb{x}) = \int _{-\infty}^{x _1} \cdots \int _{-\infty}^{x _n}  f _{\mathpzcb{X}}(x _1',\dots, x _n') dx' _1 \cdots dx' _n,$$


the integrand $f _{\mathpzcb{X}}$ is called the *joint PDF* of
$\mathpzcb{X}$. The more rigorous definition as the Radon-Nikodym
derivative


$$f _{\mathpzcb{X}} = \frac{d(\mathpzc{X} _\ast P) }{ d\lambda}$$

stays
unaltered, only that now $\lambda$ is the $n$-dimensional Lebesgue
measure.

Note that the joint CDF of the sub-vector
$(\mathpzc{X} _2, \dots, \mathpzc{X} _n)$ is given by

$$
\begin{aligned}
F _{\mathpzc{X} _2, \cdots, \mathpzc{X} _n } (x _2,\dots,x _n) &=& P(\mathpzc{X} _2 \le x _2, \dots, \mathpzc{X} _n \le x _n) 
= P(\mathpzc{X} _1 \le \infty, \mathpzc{X} _2 \le x _2, \dots, \mathpzc{X} _n \le x _n)  \\
&=& F _{\mathpzc{X} _1, \cdots, \mathpzc{X} _n } (\infty, x _2,\dots,x _n).
\end{aligned}
$$

Such a distribiution is called *marginal* w.r.t. $\mathpzc{X} _1$ and the
process of obtaining it by substituting $x _1 = \infty$ into
$F _{\mathpzcb{X}}$ is called *marginalization*. The corresponding action
in terms of the PDF consists of integration over $x _1$,

$$
\begin{aligned}
f _{\mathpzc{X} _2, \cdots, \mathpzc{X} _n } (x _2,\dots,x _n) &=& 
\int _\mathbb{R} f _{\mathpzc{X} _1, \cdots, \mathpzc{X} _n } (x _1, x _2,\dots,x _n) dx _1.
\end{aligned}
$$

Statistical independence
------------------------

A set $\mathpzc{X} _1, \dots, \mathpzc{X} _n$ of random variables is
called *statistically independent* if their joint CDF is
coordinate-separable, i.e., can be written as the following tensor
product


$$F _{\mathpzc{X} _1, \cdots, \mathpzc{X} _n} = F _{\mathpzc{X} _1} \otimes \cdots \otimes F _{\mathpzc{X} _n}.$$


An alternative definion can be given in terms of the PDF (whenever it
exists):


$$f _{\mathpzc{X} _1, \cdots, \mathpzc{X} _n} = f _{\mathpzc{X} _1} \otimes \cdots \otimes f _{\mathpzc{X} _n}.$$


We will see a few additional alternative definitions in the sequel. Let
$\mathpzc{X}$ and $\mathpzc{Y}$ be statistically-independent random
variables with a PDF and let $\mathpzc{Z} = \mathpzc{X}+\mathpzc{Y}$.
Then,

$$
\begin{aligned}
F _\mathpzc{Z}(z) &=& P(\mathpzc{Z} \le z) = P(X+Y \le z) = \int _{\mathbb{R}} \int _{\infty}^{z-y} f _{\mathpzc{X}\mathpzc{Y}}(x,y) dxdy \\
&=& \int _{\mathbb{R}} \int _{\infty}^{z} f _{\mathpzc{X}\mathpzc{Y}}(x'-y,y) dx' dy,
\end{aligned}
$$

where we changed the variable $x$ to $x' = x+y$. Differentiating w.r.t.
$z$ yields

$$\begin{aligned}
f _\mathpzc{Z}(z)  &=& \frac{dF _\mathpzc{Z}(z)}{dz} = \int _{\mathbb{R}} \frac{\partial}{\partial z} \int _{\infty}^{z} f _{\mathpzc{X}\mathpzc{Y}}(x'-y,y) dx' dy = \int _{\mathbb{R}}  f _{\mathpzc{X}\mathpzc{Y}}(z-y,y)  dy.\end{aligned}
$$

Since $\mathpzc{X}$ and $\mathpzc{Y}$ are statistically-independent, we
can substitute
$f _{\mathpzc{X}\mathpzc{Y}} = f _{\mathpzc{X}} \otimes f _{\mathpzc{Y}}$
yielding

$$\begin{aligned}
f _\mathpzc{Z}(z)  &=& \int _{\mathbb{R}}  f _{\mathpzc{X}}(z-y) f _{\mathpzc{Y}}(y)  dy = (f _{\mathpzc{X}} \ast f _{\mathpzc{Y}} )(z).\end{aligned}
$$

This result is known as the *convolution theorem*.

Limit theorems
--------------

Given independent identically distributed (i.i.d.) variables
$\mathpzc{X} _1, \dots, \mathpzc{X} _n$ with mean $\mu$ and variance
$\sigma^2$, we define their *sample average* as


$$\mathpzc{S} _n = \frac{1}{n}( \mathpzc{X} _1 + \cdots + \mathpzc{X} _n ).$$


Note that $\mathpzc{S} _n$ is also a random variable with
$\mu _{\mathpzc{S} _n} = \mu$ and
$\displaystyle{\sigma^2 _{\mathpzc{S} _n} = \frac{\sigma^2}{n}}$. It is
straightforward to see that the variance decays to zero in the limit
$n \rightarrow \infty$, meaning that $\mathpzc{S} _n$ approaches a
deterministic variable $\mathpzc{S} = \mu$. However, a much stronger
result exists: the (strong) *law of large numbers* states that in the
limit $n \rightarrow \infty$, the sample average converges in
probability to the expected value, i.e.,


$$P\left(  \lim _{n \rightarrow \infty} \mathpzc{S} _n = \mu  \right) = 1.$$


This fact is often denoted as
$\mathpzc{S} _n \mathop{\rightarrow}^P \mu$. Furthermore, defining the
normalized deviation from the limit
$\mathpzc{D} _n = \sqrt{n}(\mathpzc{S} _n - \mu)$, the *central limit
theorem* states that $\mathpzc{D} _n$ converges in distribution to
$\mathcal{N}(0,\sigma^2)$, that is, its CDF converges pointwise to that
of the normal distribution. This is often denoted as
$\mathpzc{D} _n \mathop{\rightarrow}^D \mathcal{N}(0,\sigma^2)$.

A slightly more general result is known as the *delta method* in
statistics: if $g :  \mathbb{R} \rightarrow \mathbb{R}$ is a
$\mathcal{C}^1$ function with non-vanishing derivative, then by the
Taylor theorem,


$$g(\mathpzc{S} _n) = g(\mu) + g'(\nu)(\mathpzc{S} _n-\mu) + \mathcal{O}(| \mathpzc{S} _n-\mu |^2),$$


where $\nu$ lies between $\mathpzc{S} _n$ and $\mu$. Since by the law of
large numbers $\mathpzc{S} _n \mathop{\rightarrow}^P \mu$, we also have
$\nu \mathop{\rightarrow}^P \mu$; since $g'$ is continuous,
$g'(\nu) \mathop{\rightarrow}^P g'(\mu)$. Rearranging the terms and
multiplying by $\sqrt{n}$ yields


$$\sqrt{n}( g(\mathpzc{S} _n) - g(\mu) ) = g'(\nu) \sqrt{n}( \mathpzc{S} _n) - \mu ) = g'(\nu) \mathpzc{D} _n,$$


from where (formally, by invoking the Slutsky theorem):


$$\sqrt{n}( g(\mathpzc{S} _n) - g(\mu) ) \mathop{\rightarrow}^D \mathcal{N}(0,g^{\prime} (\mu)^2 \sigma^2).$$



Joint moments
-------------

Given a measurable function
$\bb{g} : \mathbb{R}^n \rightarrow \mathbb{R}^m$, a (joint) moment of a
random vector $\mathpzcb{X} = (\mathpzc{X} _1, \dots, \mathpzc{X} _n)$ is

$$
\mathbb{E} \bb{g}(\mathpzcb{X}) = \int \bb{g}(\bb{x}) dP = 
\left(\begin{array}{c} 
 \int g _1(\bb{x}) dP \\
\vdots \\
 \int g _m(\bb{x}) dP
\end{array}
\right)
=
\left(\begin{array}{c} 
\int _{\mathbb{R}^n} g _1(\bb{x}) f _{\mathpzcb{X}}(\bb{x}) d\bb{x}  \\
\vdots \\
\int _{\mathbb{R}^n} g _m(\bb{x}) f _{\mathpzcb{X}}(\bb{x}) d\bb{x}
\end{array}
\right);
$$

the last term migh be undefined if the PDF does not exist.
The mean of a random vector is simply
$\bb{\mu} _\mathpzcb{X}  = \mathbb{E} \mathpzcb{X}$. Of particular
importance are the second-order joint moments of pairs of random
variables,


$$r _{\mathpzc{X}\mathpzc{Y}} = \mathbb{E} \mathpzc{X}\mathpzc{Y}$$

and
its central version


$$\sigma^2 _{\mathpzc{X}\mathpzc{Y}} = \mathrm{Cov}(\mathpzc{X},\mathpzc{Y}) = \mathbb{E} \left( (\mathpzc{X} - \mathbb{E} \mathpzc{X} )(\mathpzc{Y}  - \mathbb{E} \mathpzc{Y}) \right) = r _{\mathpzc{X}\mathpzc{Y}} - \mu _\mathpzc{X} \mu _\mathpzc{Y}.$$


The latter quantity is known as the *covariance* of $\mathpzc{X}$ and
$\mathpzc{Y}$.

Two random variables $\mathpzc{X}$ and $\mathpzc{Y}$ with
$r _{\mathpzc{X}\mathpzc{Y}} = 0$ are called *orthogonal*[^2]

The variables with $\sigma^2 _{\mathpzc{X}\mathpzc{Y}} = 0$ are called
*uncorrelated*. Note that for a statistically independent pair
$(\mathpzc{X},\mathpzc{Y})$,

$$
\begin{aligned}
\sigma^2 _{\mathpzc{X}\mathpzc{Y}} &=& \int _{\mathbb{R}^2} (x-\mu _\mathpzc{X}) (y-\mu _\mathpzc{Y}) d((\mathpzc{X} \times \mathpzc{Y}) _\ast P) = \int _{\mathbb{R}} (x-\mu _\mathpzc{X}) d(\mathpzc{X} _\ast P) \, \int _{\mathbb{R}} (y-\mu _\mathpzc{Y}) d(\mathpzc{Y} _\ast P) \\
&=& \mathbb{E} (\mathpzc{X} - \mathbb{E} \mathpzc{X} ) \cdot \mathbb{E} (\mathpzc{Y}  - \mathbb{E} \mathpzc{Y}) = 0.
\end{aligned}
$$

However, the converse is not true, i.e., lack of correlation does not
generally imply statistical independence (with the notable exception of
normal variables). If $\mathpzc{X}$ and $\mathpzc{Y}$ are uncorrelated
and furthermore one of them is zero-mean, then they are also orthogonal
(and the other way around).

In general, the *correlation matrix* of a random vector
$\mathpzcb{X} = (\mathpzc{X} _1, \dots, \mathpzc{X} _n)$ is given by


$$\bb{R} _{\mathpzcb{X}} = \mathbb{E}  \mathpzcb{X} \mathpzcb{X}^\Tr;$$


its $(i,j)$-th element is
$(\bb{R} _{\mathpzcb{X}}) _{ij} = \mathbb{E} \mathpzc{X} _i \mathpzc{X} _j$.
Similarly, the *covariance matrix* is defined as the central counterpart
of the above moment,


$$\bb{C} _{\mathpzcb{X}} = \mathbb{E}  (\mathpzcb{X} - \bb{\mu} _\mathpzcb{X} ) (\mathpzcb{X} - \bb{\mu} _\mathpzcb{X} )^\Tr;$$


its $(i,j)$-th element is
$(\bb{C} _{\mathpzcb{X}}) _{ij} =\mathrm{Cov}( \mathpzc{X} _i , \mathpzc{X} _j)$.
Given another random vector
$\mathpzcb{Y} = (\mathpzc{Y} _1, \dots, \mathpzc{Y} _m)$, the
*cross-correlation* and *cross-covariance* matrices are defined as
$\bb{R} _{\mathpzcb{X}\mathpzcb{Y}} = \mathbb{E}  \mathpzcb{X} \mathpzcb{Y}^\Tr$
and
$\bb{C} _{\mathpzcb{X}\mathpzcb{Y}} = \mathbb{E}  (\mathpzcb{X} - \bb{\mu} _\mathpzcb{X} ) (\mathpzcb{Y} - \bb{\mu} _\mathpzcb{Y} )^\Tr$,
respectively.

Linear transformations
----------------------

Let $\mathpzcb{X} = (\mathpzc{X} _1, \dots, \mathpzc{X} _n)$ be an
$n$-dimensional random vector, $\bb{A}$ and $m \times n$ deterministic
matrix, and $\bb{b}$ and $m$-dimensional deterministic vector. We define
a random vector $\mathpzcb{Y} = \bb{A} \mathpzcb{X} + \bb{b} $ as the
affine transformation of $\mathpzcb{X}$. Using linearity of the
expectation operator, it is straightforward to show that

$$\begin{aligned}
\bb{\mu} _\mathpzcb{Y}  &=& \mathbb{E}(\bb{A} \mathpzcb{X} + \bb{b}) = \bb{A} \bb{\mu} _\mathpzcb{X} + \bb{b} \\
\bb{C} _\mathpzcb{Y}  &=& \mathbb{E}(\bb{A} \mathpzcb{X} - \bb{A} \bb{\mu} _\mathpzcb{X} ) (\bb{A} \mathpzcb{X} - \bb{A} \bb{\mu} _\mathpzcb{X} )^\Tr = \bb{A} \bb{C} _\mathpzcb{X} \bb{A}^\Tr \\
\bb{C} _{\mathpzcb{X} \mathpzcb{Y}}  &=& \mathbb{E}(\mathpzcb{X} - \bb{\mu} _\mathpzcb{X} ) (\bb{A} \mathpzcb{X} - \bb{A} \bb{\mu} _\mathpzcb{X} )^\Tr  = \bb{C} _\mathpzcb{X} \bb{A}^\Tr.\end{aligned}$$

Estimation
==========

Let $\mathpzcb{X}$ be a latent $n$-dimensional random vector, and let
$\mathpzcb{Y}$ be a statistically related $m$-dimensional observation
(measurement). For example $\mathpzcb{Y}$ can be a linearly transformed
version of $\mathpzcb{X}$ corrupted by additive random noise,
$\mathpzcb{Y} = \bb{A}\mathpzcb{X} + \mathpzcb{N}$. We might attempt
using the information $\mathpzcb{Y}$ contains about $\mathpzcb{X}$ in
order to *estimate* $\mathpzcb{X}$. For that purpose, let us construct a
deterministic function $\bb{h} : \RR^m \rightarrow \RR^n$ that we are
going to call an *estimator*. Supplying a realization
$\mathpzcb{Y} = \bb{y}$ to this estimator will produce a deterministic
vector $\hat{\bb{x}} = \bb{h}(\bb{y})$, which is referred to as the
estimate of $\mathpzcb{X}$ given the measurement $\bb{y}$. With some
abuse of notation, we will henceforth denote $\bb{h}(\bb{y})$ as
$\hat{\bb{x}}(\bb{y})$. Note that supplying the random observation
vector $\mathpzcb{Y}$ to $\hat{\bb{x}}$ produces the random vector
$\hat{\mathpzcb{X}} = \hat{\bb{x}}(\mathpzcb{Y})$; here the
deterministic function $\hat{\bb{x}}$ acts as a random variable
transformation.

Ideally, $\hat{\mathpzcb{X}}$ and $\mathpzcb{X}$ should coincide;
however, unless the measurement is perfect, there will be a discrepancy
$\mathpzcb{E} = \hat{\mathpzcb{X}} - \mathpzcb{X}$ which we will refer
to as the *error* vector.

Maximum likelihood
------------------

For the sake of simplicity of exposition, let us focus on a very common
estimation setting with a linear forward model and an additive
statisticaly independent noise, i.e.,


$$\mathpzcb{Y} = \bb{A}\mathpzcb{X} + \mathpzcb{N},$$

where $\bb{A}$ is
a deterministic $m \times n$ matrix and $\mathpzcb{N}$ is independent of
$\mathpzcb{X}$. In this case, we can assert that the distribution of the
measurement $\mathpzcb{Y}$ given the latent signal $\mathpzcb{X}$ is
simply the distribution of $\mathpzcb{N}$ at
$\mathpzcb{N} = \mathpzcb{Y}- \bb{A} \mathpzcb{X}$,


$$P _{\mathpzcb{Y} | \mathpzcb{X}}( \bb{y} | \bb{x}  ) = P _{\mathpzcb{N}}( \bb{y} - \bb{A} \bb{x} ).$$


Assuming i.i.d. noise (i.e., that the $N _i$’s are distributed
identically and independently of each other), the latter simplifies to a
product of one-dimensional measures. Note that this is essentially a
parametric family of distributions – each choice of $\bb{x}$ yields a
distribution $P _{\mathpzcb{Y} | \mathpzcb{X} = \bb{x}}$ of
$\mathpzcb{Y}$. For the time being, let us treat the notation
$\mathpzcb{Y} | \mathpzcb{X}$ just as a funny way of writing.

Given an estimate $\hat{\bb{x}}$ of the true realization $\bb{x}$ of
$\mathpzcb{X}$, we can measure its “quality” by measuring some distance
$D$ from $P _{\mathpzcb{Y} | \mathpzcb{X}=\hat{\bb{x}}}$ to the true
distribution $P _{\mathpzcb{Y} | \mathpzcb{X}=\bb{x}}$ that created
$\mathpzcb{Y}$, and try to minimize it. Our estimator of $\bb{x}$ can
therefore be written as


$$\hat{\bb{x}} = \mathrm{arg}\min _{\hat{\mathpzcb{X}}} D(P _{\mathpzcb{Y} | \mathpzcb{X}=\bb{x}}  ||  P _{\mathpzcb{Y} | \mathpzcb{X}=\hat{\bb{x}}} ).$$


Note that we treat the quantity to be estimated as a deterministic
parameter rather than a stochastic quantity.

A standard way of measuring distance[^3] between distributions is the
so-called *Kullback-Leibler (KL) divergence*. To define it, let $P$ and
$Q$ be two probability measures (such that $P$ is absolutely continuous
w.r.t. $Q$). Then, the KL divergence from Q to P is defined as


$$D(P || Q) = \int _{} \, \log \frac{dP}{dQ} \, dP.$$

In other words, it
is the expectation of the logarithmic differences between the
probabilities $P$ and $Q$ when the expectation is taken over $P$. The
divergence can be thought of as an (asymmetric) distance between the two
distributions.

Let us have a closer look at the minimization objective

$$
D(P _{\mathpzcb{Y} | \mathpzcb{X}=\bb{x}}  ||  P _{\mathpzcb{Y} | \mathpzcb{X}=\hat{\bb{x}}}  ) = \mathbb{E} _{ \mathpzcb{Y} \sim P _{\mathpzcb{Y} | \mathpzcb{X}=\bb{x}}   }   \log\left(  \frac{P _{\mathpzcb{Y} | \mathpzcb{X}=\bb{x}}  }{ P _{\mathpzcb{Y} | \mathpzcb{X}=\hat{\bb{x}} }  } \right) =
\mathbb{E} _{ \mathpzcb{Y} \sim P _{\mathpzcb{Y} | \mathpzcb{X} = \bb{x}}   }   \log P _{\mathpzcb{Y} | \mathpzcb{X} = \bb{x}}  
-\mathbb{E} _{ \mathpzcb{Y} \sim P _{\mathpzcb{Y} | \mathpzcb{X} = \bb{x}}   }   \log P _{\mathpzcb{Y} | \mathpzcb{X}=\hat{\bb{x}} }.
$$

Note that the first term (that can be recognized as the entropy of
$\log P _{\mathpzcb{Y} | \mathpzcb{X}=\bb{x}}$) does not depend on the
minimization variable; hence, we have


$$\hat{\bb{x}} = \mathrm{arg}\min _{\hat{\bb{x}}} \, \mathbb{E} _{ \mathpzcb{Y} \sim P _{\mathpzcb{Y} | \mathpzcb{X}=\bb{x}}   }   \left( - \log P _{\mathpzcb{Y} | \mathpzcb{X}=\hat{\bb{x}}} \right).$$



Let us now assume that $N$ realization $\{ \bb{y} _1, \dots, \bb{y} _N \}$
of $\mathpzcb{Y}$ are observed. In this case, we can express the joint
p.d.f of the observations as the product of
$f _\mathpzcb{N} (\bb{y} _i - \bb{A} \bb{x})$ or, taking the negative
logarithm,


$$-\frac{1}{N} \log f _{\mathpzcb{Y} | \mathpzcb{X}=\bb{x}} (\bb{y} _1,\dots, \bb{y} _N ) = - \frac{1}{N} \sum _{i=1}^N \log f _\mathpzcb{N} (\bb{y} _i - \bb{A} \bb{x} ) = L(\bb{y} _1,\dots,\bb{y} _N | \bb{x}).$$


This function is known as the *negative log likelihood* function. By the
law of large numbers, when $N$ approaches infinity,


$$L(\bb{y} _1,\dots,\bb{y} _N | \bb{x}) \rightarrow \mathbb{E} _{ \mathpzcb{Y} \sim P _{\mathpzcb{Y} | \mathpzcb{X}=\bb{x}}   }   \left( - \log P _{\mathpzcb{Y} | \mathpzcb{X}=\hat{\bb{x}}} \right).$$


Behold our minimization objective!

To recapitulate, recall that we started with minimizing the discrepancy
between the latent parametric distribution that generated the
observation and that associated with our estimator. However, a closer
look at the objective revealed that it is the limit of the negative log
likelihood when the sample size goes to infinity. The minimization of
the Kullback-Leibler divergence is equivalent to maximization of the
likelihood of the data coming from a specific parametric distribution,


$$\hat{\bb{x}} = \mathrm{arg}\max _{\hat{\bb{x}}} \, P(  \mathpzcb{Y}=\bb{y} |  \mathpzcb{X}=\bb{x} ).$$


For this reason, the former estimator is called *maximum likelihood*
(ML).

Conditioning
------------

Before treating maximum *a posteriori* estimation, we need to briefly
introduce the important notion of conditioning and conditional
distributions. Recall our construction of a probability space comprising
the triplet $\Omega$ (the sample space), $\Sigma$ (the Borel sigma
algebra), and $P$ (the probability measure). Let $X$ be a random
variable and $B \subset \Sigma$ a sub sigma-algebra of $\Sigma$. We can
then define the *conditional expectation of $\mathpzc{X}$ given $B$* as
a random variable $\mathpzc{Z} = \mathbb{E} \mathpzc{X} | B$ satisfying
for every $E \in B$

$$\int _E \mathpzc{Z} dP = \int _E \mathpzc{X} dP.$$


(we are omitting some technical details such as, e.g., integrability
that $\mathpzc{X}$ has to satisfy).

Given another random variable $\mathpzc{Y}$, we say that it generates a
sigma algebra $\sigma(\mathpzc{Y})$ as the set of pre-images of all
Borel sets in $\mathbb{R}$,


$$\sigma(\mathpzc{Y}) = \{ \mathpzc{Y}^{-1}(A) : A \in \mathbb{B}(\mathbb{R}) \}.$$


We can then use the previous definition to define the conditional
expectation of *$\mathpzc{X}$ given $\mathpzc{Y}$* as


$$\mathbb{E} \mathpzc{X} | \mathpzc{Y} = \mathbb{E} \mathpzc{X} | \sigma(\mathpzc{Y}).$$



### Conditional distribution

Recall that expectation applied to indicator functions can be used to
define probability measures. In fact, for every $E \in \Sigma$, we may
construct the random variable $\ind _E$, leading to
$P(E) = \mathbb{E} \ind _E$. We now repeat the same, this time replacing
$\mathbb{E} $ with $\mathbb{E} \cdot | \mathpzc{Y}$. For every
$E \in \Sigma$,

$$\varphi(\mathpzc{Y}) = \mathbb{E} \, E | \mathpzc{Y}$$


is a random variable that can be thought of as a transformation of the
random variable $\mathpzc{Y}$ by the function $\varphi$. We denote this
function as $P(E |\mathpzc{Y})$ and refer to it as the (regular)
*conditional probability of event $E$ given $\mathpzc{Y}$*. It is easy
to show that for every measurable set $B \subset \mathbb{R}$,


$$\int _B P(E | \mathpzc{Y}=y) (\mathpzc{Y} _\ast P)(dy) = P(E \cap \{ \mathpzc{Y} \in B \});$$


Substituting $E = \{ \mathpzc{X} \in B\}$ yields the *conditional
distribution of random variable $X$ given $\mathpzc{Y}$*,


$$P _{\mathpzc{X} | \mathpzc{Y}} (  B  | \mathpzc{Y}=y) = P(\mathpzc{X} \in B | \mathpzc{Y}=y).$$


It can be easily shown that $P _{\mathpzc{X} | \mathpzc{Y}}$ is a valid
probability measure on $\Sigma$ and for every pair of measurable sets
$A$ and $B$,


$$\int _B P _{\mathpzc{X} | \mathpzc{Y}} (A | \mathpzc{Y}=y) (\mathpzc{Y} _\ast P)(dy) = P(\{ \mathpzc{X} \in A  \} \cap \{ \mathpzc{Y} \in B \}).$$


If density exists, $P _{\mathpzc{X} | \mathpzc{Y}}$ can be described
using the *conditional p.d.f.* $f _{\mathpzc{X} | \mathpzc{Y}}$ and the
latter identity can be rewritten in the form


$$\int _A \left( \int _B f _{\mathpzc{X} | \mathpzc{Y}} (x | y) f _\mathpzc{Y}(y) dy  \right)  dx = P(\{ \mathpzc{X} \in A  \} \cap \{ \mathpzc{Y} \in B \}) = \int _A \int _B f _{\mathpzc{XY} } (x, y) dxdy.$$


This essentially means that
$f _{\mathpzc{XY} } (x, y) = f _{\mathpzc{X} | \mathpzc{Y}} (x | y)  f _\mathpzc{Y}(y)$.
Integrating w.r.t. $y$ yields the so-called *total probability formula*


$$f _{\mathpzc{X} } (x) = \int _\mathbb{R} f _{\mathpzc{XY} } (x, y) dy = \int _\mathbb{R} f _{\mathpzc{X|Y} } (x|y)  f _\mathpzc{Y}(y) dy.$$


We can also immediately observe that if $\mathpzc{X}$ and $\mathpzc{Y}$
are statistically independent, we have


$$f _{\mathpzc{XY} } (x, y) = f _{\mathpzc{X} }(x) f _{\mathpzc{Y} } (y) =  f _{\mathpzc{X} | \mathpzc{Y}} (x | y)  f _\mathpzc{Y}(y),$$


from where $f _{\mathpzc{X} | \mathpzc{Y}} = f _{\mathpzc{X}}$. In this
case, conditioning on $\mathpzc{Y}$ does not change our knowledge of
$\mathpzc{X}$.

### Bayes’ theorem

One of the most celebrate (and useful) results related to conditional
distributions is the following theorem named after Thomas Bayes.
Exchanging the roles of $\mathpzc{X}$ and $\mathpzc{Y}$, we have


$$f _{\mathpzc{XY} }  = f _{\mathpzc{X} | \mathpzc{Y}}   f _\mathpzc{Y} = f _{\mathpzc{Y} | \mathpzc{X}}   f _\mathpzc{X};$$


re-arranging the terms, we have


$$f _{\mathpzc{Y} | \mathpzc{X}} = f _{\mathpzc{X} | \mathpzc{Y}} \, \frac{  f _\mathpzc{X} }{  f _\mathpzc{Y} };$$


in terms of probability measures, the equivalent form is


$$P _{\mathpzc{Y} | \mathpzc{X}} = P _{\mathpzc{X} | \mathpzc{Y}} \, \frac{  dP _\mathpzc{X} }{  dP _\mathpzc{Y} }.$$


### Law of total expectation

Note that treating the conditional density
$f _{\mathpzc{X}|\mathpzc{Y}}(x|y)$ just as a funnily-decorated p.d.f.
with the argument $x$, we can write the following expectation integral


$$\mathbb{E} (\mathpzc{X}|\mathpzc{Y}=y) = \int _{\RR} x f _{\mathpzc{X}|\mathpzc{Y}}(x|y) dx.$$


With (a very accepted) abuse of notation, we denote it as
"$\mathbb{E} (\mathpzc{X}|\mathpzc{Y}=y)$”. Note, however, that this is
a very different object from $\mathbb{E} \, \mathpzc{X}|\mathpzc{Y}$ –
while the former is a deterministic value, the latter is a random
variable (a transformation of $\mathpzc{Y}$). In order to construct
$\mathbb{E} \, \mathpzc{X}|\mathpzc{Y}$ out of
$\mathbb{E} (\mathpzc{X}|\mathpzc{Y}=y)$, we define the map
$\varphi : y \mapsto \mathbb{E} (\mathpzc{X}|\mathpzc{Y}=y)$ and apply
it to the random variable $\mathpzc{Y}$, obtaining
$\mathbb{E} \, \mathpzc{X}|\mathpzc{Y} = \varphi(Y)$. Again, with a
slight abuse of notation, we can write this as


$$\varphi(Y) = \mathbb{E} \, \mathpzc{X}|\mathpzc{Y} = \int _{\RR} x f _{\mathpzc{X}|\mathpzc{Y}}(x|Y) dx.$$


Let us now take a regular expectation of the transformed variable
$\varphi(Y)$, which can be viewed as a generalized moment of
$\mathpzc{Y}$,

$$\mathbb{E}\, \varphi(\mathpzc{Y}) = \int _{\RR} \varphi(y) f _{\mathpzc{Y}} (y) dy =  \int _{\RR}  \left(
\int _{\RR} x f _{\mathpzc{X}|\mathpzc{Y}}(x|y) dx  \right)  f _{\mathpzc{Y}} (y) dy.$$

Rearranging the integrands and using
$f _{\mathpzc{XY} }  = f _{\mathpzc{X} | \mathpzc{Y}}   f _\mathpzc{Y}$, we
obtain


$$\mathbb{E}\, \varphi(\mathpzc{Y}) = \int _{\RR^2} x f _{\mathpzc{XY} }(x,y) dx dy = \mathbb{E}\, X.$$


Stated differently,


$$\mathbb{E}\left(  \mathbb{E}\, X|Y \right)  = \mathbb{E}\, X.$$

This
result is known as the *smoothing theorem* or the *law of total
expectation* and can be thought of as an integral version of the law of
total probability.

Maximum *a posteriori* 
-----------------------

Recall that in maximum likelihood estimation we treated $\mathpzcb{X}$
as a deterministic parameter and tried to maximize the conditional
probability $P(\mathpzcb{Y} | \mathpzcb{X})$. Let us now think of
$\mathpzcb{X}$ as of a random vector and maximize its probability given
the data,


$$\hat{\bb{x}}(\bb{y}) = \mathrm{arg}\max _{ \hat{\bb{x}} } P _{\mathpzcb{X} | \mathpzcb{Y} } ( \mathpzcb{X} =  \hat{\bb{x}} | \mathpzcb{Y} = \bb{y}).$$


Invoking the Bayes theorem yields


$$P _{\mathpzcb{X} | \mathpzcb{Y}} = P _{\mathpzcb{Y} | \mathpzcb{X} } \, \frac{ dP _{\mathpzcb{X}} }{dP _{\mathpzcb{Y}} }$$


In the Bayesian jargon, $P _{\mathpzcb{X}}$ is called the *prior*
probability, that is, our initial knowledge about $\mathpzcb{X}$ before
any observation thereof was obtained; $P _{\mathpzcb{X} | \mathpzcb{Y}}$
is called the *posterior* probability having accounted for the
measurement $\mathpzcb{Y}$. Note that the term
$P _{\mathpzcb{Y} | \mathpzcb{X}}$ is our good old likelihood. Since we
are maximizing the posterior probability, the former estimator is called
*maximum a posteriori* (MAP).

Taking negative logarithm, we obtain


$$-\log P _{\mathpzcb{X} | \mathpzcb{Y}} = -\log P _{\mathpzcb{Y} | \mathpzcb{X} } -\log P _\mathpzcb{X} +\log P _{\mathpzcb{Y}} = L(\mathpzcb{Y} | \mathpzcb{X}) - \log P _{\mathpzcb{X}} + \mathrm{const}.$$


This yields the following expression for the MAP estimator


$$\bb{h}(\bb{Y}) = \mathrm{arg}\min _{ \hat{\bb{x}} } L(\mathpzcb{Y} |  \hat{\bb{x}} ) - \log P _\mathpzcb{X} ( \hat{\bb{x}} ).$$


The minimization objective looks very similar to what we had in the ML
case; the only difference is that now a *prior* term is added. In the
absence of a good prior, a uniform prior is typically assumed, which
reduces MAP estimation to ML estimation.

Minimum mean squared error
--------------------------

Another sound way of constructing the estimator function $\hat{\bb{x}}$
is by minimizing some error criterion related to the error vector
$\mathcal{E}(\mathpzcb{E})$. A very common pragmatic choice is the *mean
squared error* (MSE) criterion,


$$\mathcal{E}(\mathpzcb{E}) = \mathbb{E} \, \|  \mathpzcb{E} \| _2^2,$$


leading to the following optimization problem:


$$\hat{\bb{x}}^{\mathrm{MMSE}}  = \mathrm{arg} \min _{ \bb{h} : \RR^m \rightarrow \RR^n }  \mathbb{E} \, \| \bb{h}( \mathpzcb{Y} ) - \mathpzcb{X}  \| _2^2.$$


The resulting estimator is called *minimum mean squared error* (or MMSE)
estimator. Since the squared norm is coordinate separable, we can
effectively solve for each dimension of $\hat{\bb{x}}^{\mathrm{MMSE}}$
independently, finding the best (in the MSE sense) estimator of $X _i$
given $\mathpzcb{Y}$,


$$\hat{x} _i^{\mathrm{MMSE}} = \mathrm{arg} \min _{ h : \RR^m \rightarrow \RR }  \mathbb{E} \, ( h( \mathpzcb{Y} ) - X _i  )^2.$$


The minimization objective can be written explicitly as

$$
\begin{aligned}
 \mathbb{E} \, ( h( \mathpzcb{Y} ) - X _i  )^2 &=&   \mathbb{E} \, \left( \mathbb{E}\, (  h( \mathpzcb{Y} ) - X _i  )^2 | \mathpzcb{Y} \right) =
  \mathbb{E} \, \left( \mathbb{E}\,  h^2 ( \mathpzcb{Y} ) | \mathpzcb{Y}  - 2 \mathbb{E}\,  h ( \mathpzcb{Y} ) X _i | \mathpzcb{Y}   + \mathbb{E}\, X^2 _i | \mathpzcb{Y}  \right)    \\
  &=& 
    \mathbb{E} \, \left(  h^2 ( \mathpzcb{Y} )  - 2   \mathbb{E}\, X _i | \mathpzcb{Y} \cdot h ( \mathpzcb{Y} )   + \mathbb{E}\, X^2 _i | \mathpzcb{Y}  \right) \\
    &=& \int _{\RR^m}  (  h^2 ( \bb{y} )  - 2   \mathbb{E} (X _i | \mathpzcb{Y}=\bb{y}) \cdot h ( \mathpzcb{Y} )   + \mathbb{E} ( X^2 _i | \mathpzcb{Y}=\bb{y})    )  f _{\mathpzcb{Y}} (\bb{y}) d\bb{y}.
\end{aligned}
$$

The latter integral is minimized iff its non-negative integrand is
minimized at every point $\bb{y}$. Let us fix $\bb{y}$ and define
$a = h(\bb{y})$. The expression to minimize is


$$\varphi(a) = a^2  - 2a\,   \mathbb{E} (\mathpzc{X} _i | \mathpzcb{Y}=\bb{y})   + \mathbb{E} ( \mathpzc{X}^2 _i | \mathpzcb{Y}=\bb{y});$$


note that this is a convex quadratic function with the minimizer given
by

$$a^\ast = \mathbb{E} (\mathpzc{X} _i | \mathpzcb{Y}=\bb{y}).$$

From
here we conclude that


$$h(\bb{Y}) =  \mathbb{E} \mathpzc{X} _i | \mathpzcb{Y};$$

consequently,
the MMSE estimator of $\mathpzcb{X}$ given $\mathpzcb{Y}$ is given by
the conditional expectation


$$\hat{\mathpzcb{X}}^{\mathrm{MMSE}}  =  \mathbb{E} \, \mathpzcb{X} | \mathpzcb{Y}.$$



The error vector produced by the MMSE estimator is given by
$\mathpzcb{E} =  \mathbb{E} \, \mathpzcb{X} | \mathpzcb{Y} - \mathpzcb{X}$.
Taking the expectation yields


$$\mathbb{E}\, \mathpzcb{E} =  \mathbb{E} \left( \mathbb{E} \, \mathpzcb{X} | \mathpzcb{Y} \right) - \mathbb{E}\, \mathpzcb{X} =  \mathbb{E}\, \mathpzcb{X} -  \mathbb{E}\, \mathpzcb{X}  = \bb{0}.$$


In other words, the estimation error is zero mean – a property often
stated by saying that the MMSE estimator is *unbiased*.

Since the MSE is equivalent (isomorphic) to Euclidean length, MMSE
estimation can be viewed as the minimization of the length of the vector
$\mathpzcb{E}$ over the subspace of vectors of the form
$\hat{\mathpzcb{X}} = \bb{h}( \mathpzcb{Y}  )$ with
$\bb{h} : \RR^m \rightarrow \RR^m$. We known from Euclidean geometry
that the minimum length is obtained by the orthogonal projection of
$\mathpzcb{X}$ onto the said subspace, meaning that $\hat{\mathpzcb{X}}$
is an MMSE estimator iff its error vector $\mathpzcb{E}$ is *orthogonal*
to every $\bb{h}( \mathpzcb{Y}  )$, that is,


$$\mathbb{E}\, \left( (\hat{\mathpzcb{X}} - \mathpzcb{X} )  \bb{h}^\Tr ( \mathpzcb{Y}  ) \right)  = \bb{0}$$


for every $\bb{h} : \RR^m \rightarrow \RR^m$.

Best linear estimator
---------------------

Sometimes the functional dependence of
$\hat{\mathpzcb{X}}^{\mathrm{MMSE}} $ on $\mathpzcb{Y}$ might be too
complicated to compute. In that case, it is convenient to restrict the
family of functions to some simple class such as that of linear (more
precisely, affine) functions of the form
$\bb{h}(\bb{y}) = \bb{A} \bb{y} + \bb{b}$. The MMSE estimator restricted
to such a subspace of functions is known as the *best linear estimator*
(BLE), and its optimal parameters $\bb{A}$ and $\bb{b}$ are found by
minimizing


$$\min _{ \bb{A}, \bb{b} }  \mathbb{E} \, \| \bb{A} \mathpzcb{Y} + \bb{b} - \mathpzcb{X}  \| _2^2.$$


Note that since
$\mathbb{E} \mathpzcb{E} =   \bb{A} \mathbb{E}\, \mathpzcb{Y} + \bb{b} - \mathbb{E}\, \mathpzcb{X}$,
we can always zero the estimator bias by setting
$\bb{b} = \bb{\mu} _{\mathpzcb{X}} -  \bb{A}\bb{\mu} _{\mathpzcb{Y}}$.
With this choice, the problem reduces to


$$\min _{ \bb{A} }  \mathbb{E} \, \| \bb{A} (\mathpzcb{Y} - \bb{\mu} _{\mathpzcb{Y}} ) - ( \mathpzcb{X}  - \bb{\mu} _{\mathpzcb{X}} )  \| _2^2$$


or, equivalently,

$$\min _{ \bb{A} }  \, \mathbb{E} \, \mathrm{tr} \left( (\mathpzcb{Y} - \bb{\mu} _{\mathpzcb{Y}} )^\Tr \bb{A}^\Tr \bb{A} (\mathpzcb{Y} -
 \bb{\mu} _{\mathpzcb{Y}} ) \right) - 2 \mathbb{E} \,\mathrm{tr} \left(      (\mathpzcb{Y} - \bb{\mu} _{\mathpzcb{Y}} )^\Tr \bb{A}^\Tr 
  (\mathpzcb{X} -
 \bb{\mu} _{\mathpzcb{X}} )     \right).$$

Manipulating the order of
multiplication under the trace, exchaging its order with that of the
expectation operator, and moving the constants outside the expectation
yields the following minimization objective:

$$\begin{aligned}
\varphi(\bb{A}) &=& \mathrm{tr} \left(   \bb{A} \mathbb{E}   (\mathpzcb{Y} -
 \bb{\mu} _{\mathpzcb{Y}} )  (\mathpzcb{Y} -
 \bb{\mu} _{\mathpzcb{Y}} )^\Tr \bb{A}^\Tr 
 -2 \bb{A} \mathbb{E}   (\mathpzcb{X} -
 \bb{\mu} _{\mathpzcb{X}} )  (\mathpzcb{Y} -
 \bb{\mu} _{\mathpzcb{Y}} )^\Tr 
   \right) \\
   &=&  \mathrm{tr} \left(   \bb{A} \bb{C} _{\mathpzcb{Y}}  \bb{A}^\Tr 
 -2 \bb{A}  \bb{C} _{\mathpzcb{X} \mathpzcb{Y}}
   \right).
\end{aligned}
$$

Note that this is a convex (since
$\bb{C} _{\mathpzcb{Y}}  \succ 0$) quadratic function. In order to find
its minimizer, we differentiate w.r.t. the parameter $\bb{A}$ and equate
the gradient to zero:


$$0 = \nabla \varphi(\bb{A}) = 2\bb{A} \bb{C} _{\mathpzcb{Y}} - 2\bb{C} _{\mathpzcb{X} \mathpzcb{Y}}.$$


The optimal parameter is obtained as
$\bb{A} = \bb{C} _{\mathpzcb{X} \mathpzcb{Y}}\bb{C} _{\mathpzcb{Y}}^{-1}$.

Combining this result with the expression for $\bb{b}$, the best linear
estimator is

$$\hat{\mathpzcb{X}}^{\mathrm{BLE}} =  \bb{C} _{\mathpzcb{X} \mathpzcb{Y}}\bb{C} _{\mathpzcb{Y}}^{-1} (  \mathpzcb{Y} -
 \bb{\mu} _{\mathpzcb{Y}}  )  + \bb{\mu} _{\mathpzcb{X}} .$$
 
As the more
general MMSE estimator, BLE is also unbiased and enjoys the
orthogonality property, meaning that $\hat{\mathpzcb{X}}$ is an MMSE
estimator iff its error vector $\mathpzcb{E}$ is *orthogonal* to every
affine function of $\mathpzcb{Y}  )$, that is,

$$\mathbb{E}\, \left( (\hat{\mathpzcb{X}} - \mathpzcb{X} )  ( \bb{A}\mathpzcb{Y}  + \bb{b} )^\Tr \right)  = \bb{0}$$

for every $\bb{A} \in \RR^{m \times n}$ and $\bb{b} \in \RR^m$.

[^1]: To be completely rigorous, the proper way to define the PDF is by
    first equipping the image of the map $\mathpzc{X}$ with the Lebesgue
    measure $\lambda$ that assigns to every interval $[a,b]$ its length
    $b-a$. Then, we invoke the Radon-Nikodym theorem saying that if
    $\mathpzc{X}$ is absolutely continuous w.r.t. $\lambda$, there
    exists a measurable function $f : \mathbb{R} \rightarrow [0,\infty)$
    such that for every measurable $A \subset \mathbb{R}$,
    $\displaystyle{(\mathpzc{X} _\ast P)(A) =P(\mathpzc{X}^{-1}(A)) = \int _A f d\lambda}$.
    $f$ is called the *Radon-Nikodym derivative* and denoted by
    $\displaystyle{f = \frac{d(\mathpzc{X} _\ast P)}{d\lambda}} $. It is
    exactly our PDF.

[^2]: In fact, $r _{\mathpzc{X}\mathpzc{Y}}$ can be viewed as an inner
    product on the space of random variables. This creates a geometry
    isomorphic to the standard Euclidean metric in $\mathbb{R}^n$. Using
    this construction, the Cauchy-Schwarz inequality immediately
    follows:
    $| r _{\mathpzc{X}\mathpzc{Y}} | \le \sigma _\mathpzc{X} \sigma _\mathpzc{Y}$.

[^3]: Actually, not a true *metric* (which what the term distance
    implies, but rather an asymmetric form thereof, formally termed a
    *divergence*.
