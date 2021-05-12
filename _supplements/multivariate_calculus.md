---
title: Multivariate Calculus
excerpt: "Gradient, directional derivative, Hessian, Taylor expansion"
author: alex
copyright:
  name: alex
  icon: far fa-copyright
---


## Introduction

The purpose of this document is to quickly refresh (presumably) known
notions in multivariate differential calculus such as differentials,
directional derivatives, the gradient and the Hessian. These notions
will be used heavily in our course. Even though this quick reminder may
seem redundant or trivial to most of you (I hope), I still suggest at
least to skim through it, as it might present less common ways of
interpretation of very familiar definitions and properties. And even if
you discover nothing new in this document, it will at least be useful to
introduce notation.

## Notation

In our course, we will deal exclusively with real functions. A scalar
function will be denoted as $f : \RR^n \rightarrow \RR$, $f(\bb{x})$, or
simply $f$. A vector-valued function will be denoted in bold, as
$\bb{f} : \RR^n \rightarrow \RR^m$, or component-wise as
$\bb{f}(\bb{x}) = (f_1(\bb{x}), \dots, f_m(\bb{x}))^\Tr$. A scalar
function of a matrix variable, $f : \RR^{m \times n} \rightarrow \RR$,
will be denoted as $f(\bb{A})$, and a matrix-valued function of a
vector, $f : \RR^n \rightarrow \RR^{m \times k}$ as $\bb{F}(\bb{x})$.
Derivatives of a scalar function of one variable will be denoted as
$f'(x)$, $f''(x)$, etc. An $n$-times continuously differentiable
function will be said $\mathcal{C}^n$ ($f \in \mathcal{C}^n$). In most
cases, we will tacitly assume that a function is sufficiently smooth for
at least the first-order derivative to exist.

## First-order derivative of a function of one variable

Before proceeding to multivariate functions, let us remind ourselves a few basic
notions of univariate calculus. A $\mathcal{C}^1$ function $f(x)$ can be
approximated linearly around some point $x=x_0$.  Incrementing the argument by
$dx$, the function itself changes by the amount that we denote by
$\Delta f = f(x_0+dx) - f(x_0)$, while the linear approximation changes by the
amount denoted by $df$. For a sufficiently small $dx$ (more formally, in the
limit $|dx| \rightarrow 0$), it can be shown that $\Delta f = df + o(dx)$[^1].
This means that for an infinitesimally small increment $dx$, the linear
approximation of the function becomes exact. In this limit, $df$ is called the
*differential* of $f$, and the slope of the linear approximation, is called the
*first-order derivative* of $f$, denoted $\displaystyle{\frac{df}{dx} =
f'(x_0)}$.  Another way to express this fact is through the first-order *Taylor
expansion* of $f$ around $x_0$:

$$f(x_0+dx) = f(x_0) + f'(x_0) dx + O(dx^2),$$

which essentially says that a linear function whose value at $x_0$ matches that
of $f(x_0)$, and whose slope matches that of $f$ (expressed by $f'(x_0)$)
approximates $f$ around $x_0$ up to some second-order error.

## Gradient

We can extend the previous discussion straightforwardly to the
$n$-dimensional case. Let $f$ now be a $\mathcal{C}^1$ function on
$\RR^n$. The surface the function creates in $\RR^{n+1}$ can be
approximated by an $n$-dimensional tangent plane (the multidimensional
analog of linear approximation). Fixing a point $\bb{x}_0$ and making a
small step $\dx = (dx_1,\dots,dx_n)^\Tr$ (note that now $\dx$ is a
vector), it can be shown that the change in the value of the linear
approximation is given by

$$df = \frac{\partial f}{\partial x_1} dx_1 + \cdots + \frac{\partial f}{\partial x_n} dx_n,$$

where $\frac{\partial f}{\partial x_i}$ denotes the *partial derivative*
of $f$ at $\bb{x}_0$. The latter formula is usually known as the *total
differential*. Arranging the partial derivatives into a vector
$\displaystyle{\bb{g} = \left( \frac{\partial f}{\partial x_1}, \dots, \frac{\partial f}{\partial x_n} \right)^\Tr }$,
the total differential can be expressed as the inner product
$df = \langle \bb{g}, \dx \rangle$. The object $\bb{g}$ appearing in the
inner product is called the *gradient* of $f$ at point $\bb{x}_0$, and
will be denoted by $\nabla f (\bb{x}_0)$ (the symbol $\nabla$,
graphically a rotated capital Delta, is pronounced “nabla”, from the
grecized Hebrew “nevel” for “harp”; $\nabla$ is sometimes called the
*del operator*). While we can simply define the gradient as the vector
of partial derivatives, we will see that the definition through the
inner product can often be more useful.

## Directional derivative

In this course, we will often encounter situations where we are
interested in the behavior of a function along a line (formally, we say
that $f(\bb{x})$ is restricted to the one-dimensional linear subspace
$\mathcal{L} = \{ \bb{x}_0 + \alpha \bb{r} : \alpha \in \RR \}$, where
$\bb{x}_0$ is some fixed point, and $\bb{r}$ is a fixed direction). Let
use define a new function of a single variable $\alpha$,
$\varphi(\alpha) = f(\bb{x}_0 + \alpha \bb{r})$. Note that we can find
the first-order derivative of $\varphi$, arriving at the following
important notion:

$$
f_{\bb{r}}'(\bb{x}_0) = \left. \frac{d}{d\alpha} f(\bb{x}_0 + \alpha \bb{r}) \right|_{\alpha=0}  =\varphi'(0)
$$

which is called the *directional derivative* of $f$ at $\bb{x}_0$ in the
direction $\bb{r}$.

The same way a derivative measures the rate of change of a function, a
directional derivative measures the rate of change of a multivariate
function when we make a small step in a particular direction.

Denoting $\bb{g} = \nabla f(\bb{x}_0)$ and using our definition of the
gradient as the inner product, we can write

$$d\varphi = df = \bb{g}^\Tr\dx = \bb{g}^\Tr (d\alpha \bb{r}) = d\alpha (\bb{g}^\Tr \bb{r}).$$

Identifying in the latter quantity an inner product of $d\alpha$ with
the scalar $\bb{g}^\Tr \bb{r}$, we can say that $\bb{g}^\Tr \bb{r}$ is
the gradient of $\varphi(\alpha)$ at $\alpha=0$, which coincides with
the first-order derivative, $\varphi'(0) = \bb{g}^\Tr \bb{r}$, as
$\varphi$ is a function of a single variable. We can summarize this
result as the following:

**Property**. The directional derivative of $f$ at $\bb{x}_ {0}$ in the direction
$\bb{r}$ is obtained by projecting the gradient at $\bb{x}_ {0}$ onto the
direction $\bb{r}$, $f'_ {\bb{r}} = {\bb{r}}^\Tr \nabla f(\bb{x}_ {0})$.

## Hessian

In the case of a function of a single variable, we saw that the
differential of $f$ was given by $df = f'(x) dx$. However, the
first-order derivative $f'(x)$ is also a function of $x$, and we can
again express its differential as $df' = f''(x) dx$, where $f''(x)$
denotes the second-order derivative. This notion can be extended to the
multivariate case. Recall our definition of the gradient through the
inner product, $$df = \bb{g}^\Tr \dx.$$ Thinking of the gradient as of a
vector-valued function on $\RR^n$,
$\bb{g}(\bb{x}) = (g_1(\bb{x}),\dots,g_n(\bb{x}))^\Tr$, we can write

$$
\left\{ 
\begin{array}{ccc}
  dg_1 & = & \bb{h}^\Tr_1 \dx \\
  \vdots &   & \vdots \\
  dg_n & = & \bb{h}^\Tr_n \dx, 
  \end{array}
\right.
$$

with each $\bb{h}_i$ being the gradient of the $i$-th
component of the gradient vector $\bb{g}$,

$$
\bb{h}_i = \left( \frac{\partial g_i }{\partial x_1}, \dots, \frac{\partial g_i }{\partial x_n} \right)^\Tr =
\left( \frac{\partial^2 f }{\partial x_1 \partial x_i}, \dots, \frac{\partial^2 g_i }{\partial x_n \partial x_i} \right)^\Tr.
$$

Denoting by $\bb{H} = (\bb{h}_ 1,\dots,\bb{h}_ n)$, we can write compactly
$\dg = \bb{H}^\Tr \bb{dx}$. The $n\times n$ matrix $\bb{H}$ containing
all the second-order partial derivatives of $f$ as its elements is
called the *Hessian* of $f$ at point $\bb{x}$, and is also denoted[^2]
as $\nabla^2 f(\bb{x})$. We tacitly assumed that $f$ is $\mathcal{C}^2$
in order for the second-order derivatives to exist. A nice property of
$\mathcal{C}^2$ functions is that partial derivation is commutative,
meaning that the order of taking second-order partial derivatives can be
interchanged:
$\displaystyle{h_{ij} = \frac{\partial^2 f }{\partial x_i \partial x_j} = \frac{\partial^2 f}{\partial x_j \partial x_i} = h_{ji} }$.
Algebraically, this implies that the Hessian matrix is symmetric, and we
can write

$$\dg = \bb{H} \bb{dx}.$$

## Second-order directional derivative

Recall that we have previously considered the restriction of a
multivariate function $f$ to a line,
$\varphi(\alpha) = f(\bb{x}_ 0 + \alpha \bb{r})$. This gave rise to the
first-order directional derivative $f_{\bb{r}}(\bb{x}_ 0) = \varphi'(0)$.
In a similar way, we define the *second-order directional derivative* at
$\bb{x}_ 0$ in the direction $\bb{r}$ as

$$
\begin{aligned}
f''_{\bb{rr}}(\bb{x}_0) &=& \varphi''(0) = \left. \frac{d^2}{d\alpha^2} f(\bb{x}_0 + \alpha\bb{r}) \right|_{\alpha=0}
= \left. \frac{d}{d\alpha} f'_{\bb{r}}(\bb{x}_0 + \alpha\bb{r}) \right|_{\alpha=0}.
\end{aligned}
$$

Considering $f'_{\bb{r}}(\bb{x}) = \bb{r}^\Tr \bb{g}(\bb{x})$ as a
function of $\bb{x}$, we can write its differential as

$$df'_{\bb{r}} = \bb{r}^\Tr \dg = \bb{r}^\Tr \bb{H}(\bb{x}_0) \dx = \bb{r}^\Tr \bb{H}(\bb{x}_0) \bb{r} d\alpha,$$

from where

$$f''_{\bb{rr}} = \bb{r}^\Tr \bb{H} \bb{r}.$$

In other words, in order to get the second-order directional derivative in the
direction $\bb{r}$, one has to evaluate the quadratic form $\bb{r}^\Tr \bb{H}
\bb{r}$.

## Derivatives of linear and quadratic functions

Let $\bb{y} = \bb{A}\bb{x}$ be a general linear operator defined by an
$m \times n$ matrix. Its differential is given straightforwardly by

$$\dy = \bb{A}(\bb{x} + \dx) - \bb{A}\bb{x} = \bb{A}\dx.$$

Using this result, we will do a small exercise deriving gradients and Hessians
of linear and quadratic functions. As we will see, it is often convenient to
start with evaluating the differential of a function.

Our first example is a linear function of the form
$f(\bb{x}) = \bb{b}^\Tr \bb{x}$, where $\bb{b}$ is a constant vector.
Note that this function is a particular case of the previous result
(with $\bb{A} = \bb{b}^\Tr$), and we can write $df = \bb{b}^\Tr \dx$.
Comparing this to the general definition of the gradient,
$df = \bb{g}^\Tr(\bb{x}) \dx$, we deduce that the gradient of $f$ is
given by $\nabla f(\bb{x}) = \bb{b}$. Note that the gradient of a linear
function is constant – this generalizes the case of a linear function of
one variable, $f(x)= bx$, which has a constant derivative $f'(x) = b$.

Our second example is a quadratic function of the form
$f(\bb{x}) = \bb{x}^\Tr \bb{A} \bb{x}$, where $\bb{A}$ is an
$n \times n$ matrix. We again compute the differential by definition,

$$
\begin{aligned}
df &=& f(\bb{x}+\dx) -f(\bb{x}) = (\bb{x}+\dx)^\Tr \bb{A} (\bb{x}+\dx)- \bb{x}^\Tr \bb{A} \bb{x} \nonumber\\
   &=& \bb{x}^\Tr \bb{A} \bb{x} + \dx^\Tr \bb{A} \bb{x} + \bb{x}^\Tr \bb{A} \dx + \dx^\Tr \bb{A} \dx - \bb{x}^\Tr \bb{A} \bb{x} \nonumber\\
   &=& \dx^\Tr \bb{A} \bb{x} + \bb{x}^\Tr \bb{A} \dx + \dx^\Tr \bb{A} \dx.
\end{aligned}
$$

Note that in the limit $\| \dx \| \rightarrow 0$, the third term
(quadratic in $\|\dx\|$) goes to zero much faster than the first two
terms (linear in $\dx$), and can be therefore neglected[^3], leading to

$$
df = \dx^\Tr \bb{A} \bb{x} + \bb{x}^\Tr \bb{A} \dx =
\dx^\Tr \bb{A} \bb{x} + (\bb{x}^\Tr \bb{A} \dx)^\Tr =
\dx^\Tr(\bb{A}^\Tr + \bb{A})\bb{x}.
$$

Again, recognizing in the latter expression an inner product with $\dx$,
we conclude that $\nabla f(\bb{x}) = (\bb{A}^\Tr + \bb{A})\bb{x}$. For a
symmetric $\bb{A}$, the latter simplifies to
$\nabla f(\bb{x}) = 2\bb{A} \bb{x}$. Note that the gradient of a
quadratic function is a linear function; furthermore, the latter
expression generalizes the univariate quadratic function $f(x) = ax^2$,
whose first-order derivative $f'(x) = 2ax$ is linear.

Since the gradient $\bb{g}(\bb{x}) = (\bb{A}^\Tr +\bb{A})\bb{x}$ of the
quadratic function is linear, its differential is immediately given by
$\dg = (\bb{A}^\Tr +\bb{A})\dx$, from where we conclude that the Hessian
of $f$ is $\bb{H}(\bb{x}) = \bb{A}^\Tr +\bb{A}$ (or $2\bb{A}$ in the
symmetric case). Note that the Hessian of a quadratic function is
constant, which coincides with the univariate case $f'' (x) = 2a$.

In the sequel, we will see more complicated examples of gradients and
Hessians. For a comprehensive reference on derivatives of matrix and
vector expressions, the Matrix Cookbook[^4] is highly advisable.

## Multivariate Taylor expansion

We have seen the Taylor expansion of a function of one variable as a way
to obtain a linear approximation. This construction can be generalized
to the multivariate case, as we show here, limiting the expansion to
second order.


**Theorem: Second-order Taylor expansion**.
Let $f$ be a $\mathcal{C}^2$ function on $\RR^n$, $\bb{x}$ some point,
and $\bb{r}$ a sufficient small vector. Then,

$$
f(\bb{x}+\bb{r}) =
f(\bb{x}) + \bb{g}^\Tr (\bb{x}) \bb{r} + \frac{1}{2} \bb{r}^\Tr \bb{H}(\bb{x}) \bb{r} + O(\|\bb{r}\|^3).
$$

The theorem say that up to a third-order error term, the function can be
approximated around $\bb{x}$ by a quadratic function
$q(\bb{r}) = f + \bb{g}^\Tr \bb{r} + \frac{1}{2} \bb{r}^\Tr \bb{H} \bb{r}$
(note that the function is quadratic in $\bb{r}$, as $\bb{x}$ is
constant, and so are $f=f(\bb{x})$, $\bb{g}$, and $\bb{H}$). Out of all
possible quadratic approximations of $f$, the approximation described by
$q(\bb{r}) \approx f(\bb{x} + \bb{r})$ is such that its value, slope,
and curvature at $\bb{x}$ (equivalently, at $\bb{r} = \bb{0}$) match
those of $f$. The latter geometric quantities are captured,
respectively, by the values of the function, its gradient, and its
Hessian; in order to match the value, slope, and curvature of $f$, $q$
has to satisfy $q(\bb{0}) = f(\bb{x})$,
$\nabla q(\bb{0}) = \nabla f(\bb{x})$, and
$\nabla^2 q(\bb{0}) = \nabla^2 f(\bb{x})$ (note that the gradient and
the Hessian of $q$ are w.r.t $\bb{r}$, whereas the derivatives of $f$
are w.r.t. $\bb{x}$). To see that the later equalities hold, we first
observe that $q(\bb{0}) = f(\bb{x})$. Next, using the fact that
$q(\bb{r})$ is quadratic, its gradient and Hessian (w.r.t. $\bb{r}$) are
given by $\nabla q(\bb{r}) = \bb{g} + \bb{H} \bb{r}$ and
$\nabla^2 q(\bb{r}) = \bb{H} \bb{r}$. Substituting $\bb{r} = \bb{0}$
yields $\nabla q(\bb{0}) = \bb{g}$ and $\nabla^2 q(\bb{r}) = \bb{H}$.

## Gradient of a function of a matrix

The notion of gradient can be generalized to functions of matrices. Let
$f : \RR^{m \times n} \rightarrow \RR$ be such function evaluated at
some $\bb{X}$. We can think of an equivalent function on $\RR^{mn}$
evaluated at $\bb{x} = \vec(\bb{X})$, for which the gradient is defined
simply as the $mn$-dimensional vector of all partial derivatives. We can
therefore think of the gradient of $f(\bb{X})$ at $\bb{X}$ as of the
$m \times n$ matrix

$$
\bb{G}(\bb{X}) = \left(
                   \begin{array}{ccc}
                     \frac{\partial f }{\partial x_{11} } & \cdots & \frac{\partial f }{\partial x_{1n} } \\
                     \vdots & \ddots & \vdots \\
                     \frac{\partial f }{\partial x_{m1} } & \cdots & \frac{\partial f }{\partial x_{mn} } \\
                   \end{array}
                 \right).
$$

Previously, we have seen that an “external” definition of the gradient
through an inner product is often more useful. Such a definition is also
valid for matrix arguments. Recall our definition of the standard inner
product on the space of $m\times n$ matrices as
$$\langle \bb{A}, \bb{B} \rangle = \sum_{ij} a_{ij} b_{ij} = \trace(\bb{A}^\Tr \bb{B}),$$
for $\bb{A},\bb{B} \in \RR^{m \times n}$. Using the total differential
formula yields

$$df = \sum_{ij} \frac{\partial f }{\partial x_{ij}} dx_{ij} = \langle \bb{G}, \dX \rangle,$$

where $\dX$ is now an $m\times n$ matrix. The matrix $\bb{G}$ appearing
in the above identity can be *defined* as the gradient of $f$.

## Gradient of a nonlinear function

We finish this brief introduction by deriving the gradient of a more
complicated function of the form

$$f(\bb{X}) = \bb{c}^\Tr \varphi( \bb{X}^\Tr \bb{a} + \bb{b}),$$

where
$\bb{X} \in \RR^{m\times n}$, $\bb{a} \in \RR^n$,
$\bb{b},\bb{c} \in \RR^m$, and $\varphi$ is a $\mathcal{C}^1$ function
applied element-wise. We will encounter such functions during the course
when dealing with nonlinear regression and classification applications.
In machine learning, functions of this form constitute building
blocks of more complicated functions called artificial neural networks.
As before, we proceed by computing differentials and using the chain
rule. Denoting $\bb{u} = \bb{X}^\Tr \bb{a} + \bb{b}$, we have

$$\varphi(\bb{u}) = \left(
                    \begin{array}{c}
                      \varphi(u_1) \\
                      \vdots \\
                      \varphi(u_m) \\
                    \end{array}
                  \right).
$$

Since $\varphi$ is applied element-wise to
$\bb{u}$, the differential of $\bb{\varphi} = \varphi(\bb{u})$ is given
by

$$\dphi = \left(
                    \begin{array}{c}
                      \varphi'(u_1) du_1\\
                      \vdots \\
                      \varphi'(u_m) du_m \\
                    \end{array}
                  \right) = 
                  \underbrace{\left(
                    \begin{array}{ccc}
                      \varphi'(u_1) &  &  \\
                       & \ddots &  \\
                       &  & \varphi'(u_m) \\
                    \end{array}
                  \right)}_{\bb{\Phi}'} \du = \bb{\Phi}' \du.
$$

Next, we consider the function $\bb{u}(\bb{X}) = \bb{X}^\Tr \bb{a} + \bb{b}$;
since it is linear in $\bb{X}$, its differential is given by $\du = \dX^\Tr
\bb{a}$. Finally, we consider the function $f(\bb{\varphi}) = \bb{c}^\Tr
\bb{\varphi}$, which is linear in $\bb{\varphi}$ and has the differential
$df = \bb{c}^\Tr \dphi$.

Combining these results and using simple properties of the matrix trace
yields

$$
\begin{aligned}
df &=& \bb{c}^\Tr \dphi = \bb{c}^\Tr \bb{\Phi}' \du = \bb{c}^\Tr\bb{\Phi}' \dX^\Tr \bb{a} \\
&=& \trace\left( \bb{c}^\Tr\bb{\Phi}' \dX^\Tr \bb{a} \right) = \trace\left( \dX^\Tr \bb{a}\bb{c}^\Tr\bb{\Phi}'  \right) \\
&=& \langle \dX, \bb{a}\bb{c}^\Tr\bb{\Phi}'\rangle.
\end{aligned}
$$

In the latter expression, we recognize in the second argument of the inner
product the gradient of $f$ w.r.t. $\bb{X}$,

$$\nabla f(\bb{X}) = \bb{a}\bb{c}^\Tr\bb{\Phi}'.$$

[^1]: The little-$o$ notation means that there exists some function of
    $dx$, $o(dx)$, going faster to zero than $dx$ (i.e.,
    $\displaystyle{\frac{o(dx)}{dx}} \rightarrow 0$), but the exact form
    of this function is unimportant. On the other hand, the big-$O$
    notation, as in $O(dx^2)$, stands for some function that grows with
    the same rate as $dx^2$ (i.e.,
    $\displaystyle{\lim_{|dx|\rightarrow 0} \frac{dx^2}{O(dx^2)} < \infty }$).

[^2]: Some people find the following abuse of notation helpful: Thinking
    of the gradient of $f$ as of a differential operator of the form
    “$\displaystyle{\nabla = \left(
                                \begin{array}{c}
                                \frac{\partial }{\partial x_1} \\
                                \vdots \\
                                \frac{\partial }{\partial x_n} \\
                                \end{array}
                            \right)}$”
    applied to $f$, the Hessian can be expressed by applying the
    operator “$\displaystyle{
    \nabla^2 = \nabla \nabla^\Tr =
    \left(
        \begin{array}{c}
        \frac{\partial }{\partial x_1} \\
        \vdots \\
        \frac{\partial }{\partial x_n} \\
        \end{array}
    \right)
    \left(\textstyle{
        \frac{\partial }{\partial x_1}},\dots, \textstyle{\frac{\partial }{\partial x_n}}
    \right) =
    \left(
        \begin{array}{ccc}
            \frac{\partial^2 }{\partial x_1 \partial x_1} & \cdots & \frac{\partial^2 }{\partial x_1 \partial x_n} \\
            \vdots &  \ddots & \vdots \\
            \frac{\partial^2 }{\partial x_n \partial x_1} & \cdots & \frac{\partial^2 }{\partial x_n \partial x_n} \\
        \end{array}
    \right)
    }$”.

[^3]: This “explanation” can be written rigorously using limits. Another
    way of getting the same result is the well-known rule of
    “differential of a product”, $d(fg) = df\, g + f \, dg$, which can
    be generalized to the multivariate case as follows: Let $h$ be a
    scalar function given as the inner product of two vector-valued
    functions, $h(\bb{x}) = \bb{f}^\Tr(\bb{x}) \bb{g}(\bb{x})$. Then,
    $dh = \df^\Tr \bb{g} + \bb{f}^\Tr \dg$.

[^4]: <http://www2.imm.dtu.dk/pubdb/views/edoc_download.php/3274/pdf/imm3274.pdf>
