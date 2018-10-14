---
title: "Lecture 3: Multi-layer Perceptron"
date: 2018-10-01
excerpt: MLP, Backpropagation, Gradient Descent, CNNs
---

## The Perceptron

In the last lecture, we discussed supervised learning with a linear hypothesis
class of the form

$$
y = {\bm{\mathrm{w}}}^\Tr {\bm{\mathrm{x}}}+b
$$

parametrized by $n$ weights ${\bm{\mathrm{w}}} = (w_0,w_1,\dots,w_n)$ and a
bias $b$. In the machine learning literature, this family of functions (or
“architecture” as we shall call it in the sequel) is known as a (linear)
*perceptron*.

We have seen that in the case of binary logistic regression (which, despite the
name, is a binary *classification* problem) the scalar output $y$ of the
hypothesis was further fed into the logistic (a.k.a.  *sigmoid* function 

$$
\psi(t) = \frac{e^t}{1+e^t}.
$$

This can be viewed as a *two-dimensional* output of the form

$$
{\bm{\mathrm{y}}} = \left(  \frac{e^{ {\bm{\mathrm{w}}}^\Tr {\bm{\mathrm{x}}}+b} }{1+e^{ {\bm{\mathrm{w}}}^\Tr {\bm{\mathrm{x}}} + b}}, \frac{1}{1+e^{ {\bm{\mathrm{w}}}^\Tr {\bm{\mathrm{x}}}+b}} \right)
$$

which can be interpreted as the vector of probabilities of the instance
${\bm{\mathrm{x}}}$ belonging to each of the two classes.

Using this perspective, the linear perceptron model can be generalized
to the $k$ class cases according to

$$
{\bm{\mathrm{y}}} = \frac{ e^{ {\bm{\mathrm{W}}}^\Tr
{\bm{\mathrm{x}}}+{\bm{\mathrm{b}}} }  }{ {\bm{\mathrm{1}}}^\Tr e^{
{\bm{\mathrm{W}}}^\Tr {\bm{\mathrm{x}}} + {\bm{\mathrm{b}}}}  } =
\displaystyle{\left(  \frac{ e^{ {\bm{\mathrm{w}}}_1^\Tr {\bm{\mathrm{x}}} + b_1
} }{\sum_{i=1}^k e^{ {\bm{\mathrm{w}}}_i^\Tr {\bm{\mathrm{x}}} + b_i  } } ,
\dots,  \frac{ e^{ {\bm{\mathrm{w}}}_n^\Tr {\bm{\mathrm{x}}}  + b_n }
}{\sum_{i=1}^k e^{ {\bm{\mathrm{w}}}_i^\Tr {\bm{\mathrm{x}}} + b_i  } }  \right)
},
$$

where ${\bm{\mathrm{W}}}$ is a $k \times n$ weight matrix whose rows are
denoted as ${\bm{\mathrm{w}}}_i$, ${\bm{\mathrm{b}}}$ is a
$k$-dimensional bias vector, and ${\bm{\mathrm{1}}}$ is an
appropriately-sized vector of ones. This generalization of the logistic
function used to normalize the output intot the form of a vector of
probabilities is known as *softmax*. Softmax is a function of the form

$$
\psi({\bm{\mathrm{z}}}) =  \frac{ e^{ {\bm{\mathrm{z}}}}  }{
{\bm{\mathrm{1}}}^\Tr e^{ {\bm{\mathrm{z}}}}  }
$$

that highlights the maximal value in the vector ${\bm{\mathrm{z}}}$ and
suppresses other elements that are significantly lower than the maximum.

### Adding layers

The linear perceptron model is rather limited due to its linearity. For example,
it cannot produce the XOR function. A much more powerful family of functions is
obtained by applying a non-linearity to the output of a linear perceptron and
concatenating several such models. We define the $i$-th *layer* as

$$
{\bm{\mathrm{y}}}_i = \varphi_{i} ( {\bm{\mathrm{W}}}_i {\bm{\mathrm{y}}}_{i-1} + {\bm{\mathrm{b}}}_i ),
$$

for $i=1,\dots,L$, where ${\bm{\mathrm{y}}}\_{i-1}$ is an
$n\_{i-1}$-dimensional input, ${\bm{\mathrm{y}}}\_i$ is an
$n\_{i}$-dimensional output, ${\bm{\mathrm{W}}}\_i$ is an
$n\_i \times n\_{i-1}$ matrix of weights (whose columns are denoted as
${\bm{\mathrm{w}}}^{i}\_1,\dots, {\bm{\mathrm{w}}}^{i}\_{n\_{i-1}}$),
${\bm{\mathrm{b}}}\_i$ is an $n\_i$-dimensional bias vector, and
$\varphi\_i : \RR \rightarrow \RR$ is is a non-linear function applied
element-wise. Setting ${\bm{\mathrm{y}}} = {\bm{\mathrm{y}}}\_L$ and
${\bm{\mathrm{y}}}\_0 = {\bm{\mathrm{x}}}$, a *multi-layer perceptron*
(MLP) with $L$ layers is obtained. MLP can be described by the following
input-to-output map

$$
{\bm{\mathrm{y}}}=\varphi_L \left(   {\bm{\mathrm{W}}}_{L} \varphi_{L-1}( {\bm{\mathrm{W}}}_{L-1}  \varphi_{L-2}(  \cdots  \varphi_1({\bm{\mathrm{W}}}_1 {\bm{\mathrm{x}}})   \cdots  )    ) \right).
$$

parametrized by the weight matrices
$\{ {\bm{\mathrm{W}}}_1,\dots,{\bm{\mathrm{W}}}_L \}$ and bias vectors
$\{ {\bm{\mathrm{b}}}_1,\dots,{\bm{\mathrm{b}}}_L \}$ which we will
collectively denote as a pseudo-vector ${\bm{\mathrm{\Theta}}}$.

Graphically, the $i$-th layer can be thought of a weighted directed graph
connecting each of the $n_{i-1}$ inputs to $n_i$ sum nodes with the
weights given by elements of ${\bm{\mathrm{W}}}\_i$. The output of each
sum node undergoes a non-linearity and together the $n_i$ outputs form
the input of the following layer. Because of its (deliberate)
resemblance to biological neural networks, MLP is called an (artificial)
neural network. In the jargon of artificial neural networks, each
sub-graph of the form
$y^i_j = \varphi_i (  {\bm{\mathrm{y}}}_{i-1}^\Tr {\bm{\mathrm{w}}}^{i}_j + b_j  )$
is called a *neuron* (the $j$-th neuron in $i$-th layer), its
non-linearity $\varphi_i$ is called an *activation function*, and its
output $y^i_j$ an *activation*. MLP is a *feedforward* neural network,
since the graph is acyclic – the data flow forward from the input to the
output without feedback loops.

Unlike their single-layered linear counterparts, MLPs constitute a
potent hypothesis class. In fact, even with just two layers, MLPs were
shown to be *universal approximators* – their weights can be selected to
approximate any function under mild technical conditions, provided they
have enough degrees of freedom (sufficiently large number of weights).

### Non-linearity

Various functions can be used as the element-wise nonlinearities
(activation function) of the MLP. Older neural networks used the
logistic function (a.k.a. sigmoid) 

$$
\varphi(t) = \frac{1}{1+e^{-t}}
$$

saturating the input in $\RR$ between $0$ and $1$, or its shifted and
scaled version

$$
\varphi(t) =  \frac{e^t  - e^{-t}}{e^t +e^{-t}} = \mathrm{tanh}\, t.
$$

The arctangent function also has a sigmoid-like behavior.

However, due to numerical issues that will be discussed in the sequel, these
functions were nowadays almost universally replaced by the *rectifier* function
(a.k.a. *rectified linear unit* or ReLU)

$$
\varphi(t) = [t]_+ = \max\{t,0\}.
$$

Note that this function has the derivative of exactly $0$ on $(-\infty,0)$,
exactly $1$ on $(0,\infty)$, and is non-smooth at $0$. These facts justifying
its choice will be discussed in the sequel.

In addition to element-wise non-linearities, modern neural networks sometimes
use “horizontal” non-linearities acting on the entire activation vector. One
typical choice of such a non-linearity adopted in classification networks is a
softmax function applied to the activation of the last (output) layer. Other
non-linearities of this kind are pooling operations that will be discussed in
the sequel.

## Supervised training

Now equipped with a new richer hypothesis class, let us zoom out to see the
whole picture. In the supervised learning problem, we are given a finite sample
of labeled training instances $\{  ({\bm{\mathrm{x}}}\_i, y\_i) \}_{i=1}^N$. We
then select a hypothesis that minimizes the empirical (in-sample) loss function,

$$
h^\ast =  \mathrm{arg} \min_{h \in \mathcal{H}} \frac{1}{N} \sum_{i=1}^N \ell( h({\bm{\mathrm{x}}}_i), y_i).
$$

In our terms, this minimization problem can be written as

$$
{\bm{\mathrm{\Theta}}}^\ast =  \mathrm{arg} \min_{ {\bm{\mathrm{\Theta}}} }
\frac{1}{N} \sum_{i=1}^N \ell_i ( h_{ {\bm{\mathrm{\Theta}}}}
({\bm{\mathrm{x}}}_i) ),
$$

where $h_{ {\bm{\mathrm{\Theta}}}}$ is the MLP parametrized by the pseudo-vector
${\bm{\mathrm{\Theta}}}$. Note that to simplify notation we dropped the
dependence of the $i$-th pointwise loss term on $y_i$, denoting it by $\ell_i$.
We will henceforth denote the loss function as

$$
L({\bm{\mathrm{\Theta}}}) = \frac{1}{N} \sum_{i=1}^N \ell_i ( h_{ {\bm{\mathrm{\Theta}}}} ({\bm{\mathrm{x}}}_i) )
$$

emphasizing that we are interested in its dependence on the model parameters
${\bm{\mathrm{\Theta}}}$. Let us now discuss how to minimize it with respect to
${\bm{\mathrm{\Theta}}}$.

### Global and local minima

Let us assume that $L$ is a function of an $m$-dimensional argument
${\bm{\mathrm{\theta}}}$ defined on all $\RR^m$ (we can always parse all the
degrees of freedom of our neural network into an $m$-dimensional vector). A
point ${\bm{\mathrm{\theta}}}^\ast$ is called a *global minimizer* of $L$ if for
any ${\bm{\mathrm{\theta}}}$, $L({\bm{\mathrm{\theta}}}) \ge
L({\bm{\mathrm{\theta}}}^\ast)$. The corresponding value of the function, $
L({\bm{\mathrm{\theta}}}^\ast)$, is called a *global minimum*. The latter term
is often (strictly speaking, erroneously) used to denote the minimizer as well.
A point ${\bm{\mathrm{\theta}}}^\ast$ is called a *local minimizer* of $L$ if
there exists $\epsilon > 0$ such that ${\bm{\mathrm{\theta}}}^\ast$ is a global
minimizer of $L$ on the ball $B_\epsilon({\bm{\mathrm{\theta}}}^\ast)$.

Unless $L$ satisfied special properties (such as convexity), finding its global
minimizer is an unsolvable problem. On the other hand, finding a local minimizer
is a much easier task, since local minimizers can be characterized using local
information (i.e., derivatives). Assuming $L$ is $\mathcal{C}^1$, from
elementary multivariate calculus we should recollect the first-order necessary
condition for ${\bm{\mathrm{\theta}}}^\ast$ being a local minimizer:

$$
\nabla_{ {\bm{\mathrm{\theta}}}} L({\bm{\mathrm{\theta}}}^\ast) = {\bm{\mathrm{0}}}.
$$

Obviously, this is not a sufficient condition – in fact, a local maximum and a
saddle point also satisfy it. However, the latter two types of extremal points
(characterized by negative curvature) are *unstable*, which will allow methods
such as stochastic gradient descent not to remain stuck at such points.

As a reminder, the *gradient* of a multi-variate function is an operator
$\nabla L : \RR^m \rightarrow \RR^m$. At a given point
${\bm{\mathrm{\theta}}}$, it produces a vector
${\bm{\mathrm{g}}} = \nabla L({\bm{\mathrm{\theta}}})$ satisfying

$$
dL = \langle {\bm{\mathrm{g}}}, {\bm{\mathrm{dx}}} \rangle = {\bm{\mathrm{g}}}^\Tr {\bm{\mathrm{d\theta}}};
$$

in other words, an inner product of the argument change
${\bm{\mathrm{d\theta}}}$ with the gradient yields the differential $dL$.

### Gradient descent

We can therefore suggest a very simple iterative strategy for finding a local
minimum, which can be summarized as the following “algorithm”:

Starting with some *initial guess* ${\bm{\mathrm{\theta}}}_0$, repeat for
$k=1,2,\dots$

1.  Select a *descent direction* ${\bm{\mathrm{d}}}_k$
2.  Select a *step size* $\eta_k$
3.  Update
    ${\bm{\mathrm{\theta}}}\_k = {\bm{\mathrm{\theta}}}\_{k-1} + \eta\_k {\bm{\mathrm{d}}}\_k$
4.  Check optimality condition at ${\bm{\mathrm{\theta}}}_k$ and stop if
    minimum is reached

(In practice, rather than checking the optimality condition, we will run the
algorithm for a fixed number of iterations and stop it prematurely based on the
value of cross-validation loss – these details will be discussed further in the
course.)

The main ingredient of the above “algorithm” is the choice of the descent
direction, i.e., a direction a (small) step in which decreases the value of the
function. Let ${\bm{\mathrm{\theta}}}$ be our current iterate (we drop the
iteration subscript) and let ${\bm{\mathrm{d}}}$ be a direction. Once a
direction is choses, we can consider a one-dimensional “section” of the
$m$-dimensional function $L$,

$$
f(\eta) = L({\bm{\mathrm{\theta}}}+\eta {\bm{\mathrm{d}}}).
$$

The quantity

$$
f'(0) = \left. \frac{d L({\bm{\mathrm{\theta}}}+\eta {\bm{\mathrm{d}}})
}{d\eta} \right|_{\eta = 0} = \nabla L({\bm{\mathrm{\theta}}})^\Tr
{\bm{\mathrm{d}}}
$$

is known as the *directional derivative* of $L$ at point
${\bm{\mathrm{\theta}}}$ in the direction ${\bm{\mathrm{d}}}$. A negative
directional derivative indicates that a small step in the direction
${\bm{\mathrm{d}}}$ decreases the value of the function.  Geometrically, this
means that a descent direction forms an *obtuse angle* with the gradient (or an
acute angle with the negative gradient).

Let us now approximate our function linearly around
${\bm{\mathrm{\theta}}}$,

$$
L({\bm{\mathrm{\theta}}}+{\bm{\mathrm{d}}}) \approx L({\bm{\mathrm{\theta}}}) +
\nabla L({\bm{\mathrm{\theta}}})^\Tr {\bm{\mathrm{d}}}
$$

and ask ourselves what direction minimizes the difference
$L({\bm{\mathrm{\theta}}}+{\bm{\mathrm{d}}}) -  L({\bm{\mathrm{\theta}}})
\approx \nabla L({\bm{\mathrm{\theta}}})^\Tr {\bm{\mathrm{d}}}$ – we could call
such a direction the *steepest* descent direction.  Obviously, this linear
approximation is unbounded, so we need to normalize the length of
${\bm{\mathrm{d}}}$. Different choices of the norm lead to different answers (so
there are many steepest directions); in the $\ell_2$ sense we obtain

$$
{\bm{\mathrm{d}}} = -\nabla {\bm{\mathrm{d}}}.
$$

This choice of the descent direction leads to a family of algorithms known as
*gradient descent*.

Our next goal is to select the step size $\eta$. Ideally, once we have the
direction ${\bm{\mathrm{d}}}$, we would like to solve for

$$
\eta = \mathrm{arg}\min_{\eta} L({\bm{\mathrm{\theta}}}+\eta {\bm{\mathrm{d}}}).
$$

While there exist various methods known as *line search* to solve such a
one-dimensional minimization problem, usually they come at the expense of
unaffordable extra complexity. In deep learning, a much more common choice is to
use a vanishing sequence of weights that start with some initial $\eta_0$ which
is kept for a certain number of iterations and then gradually reduced as $1/k$.
Using the statistical mechanics metaphor, such a reduction in the step size
resembles a decrease in temperature and is therefore referred to as *annealing*.

Gradient descent can be thus summarized as

Starting with some *initial guess* ${\bm{\mathrm{\theta}}}_0$, repeat for
$k=1,2,\dots$

1.  Select a *step size* $\eta_k$
2.  Update
    ${\bm{\mathrm{\theta}}}\_k = {\bm{\mathrm{\theta}}}\_{k-1} - \eta\_k \nabla L({\bm{\mathrm{\theta}}}\_{k-1})$
3.  Check optimality condition at ${\bm{\mathrm{\theta}}}_k$ and stop if
    minimum is reached

We will discuss variants of the gradient descent algorithm that are used
in practice in the sequel.

### Error backpropagation

The main computation ingredient in the gradient descent algorithm is the
gradient of the loss function w.r.t. the network parameters
${\bm{\mathrm{\theta}}}$. Obviously, since an MLP is just a composition of
multi-variate functions, the gradient can be simply computed invoking the chain
rule. However, recall that the output of the network is usually a
$k$-dimensional vector, whereas the parameters are a collection of
$n_i \times n_{i-1}$ weight matrices and $n_i$-dimensional bias vectors. The
gradient of a vector with respect to a matrix (formally termed the Jacobian) is
a third-order tensor, which is not exactly nice to work with.

A much more elegant approach to apply the chain rule takes advantage of the
layered structure of the network. As an illustration, we start with a two-layer
MLP of the form

$$
{\bm{\mathrm{y}}} = \varphi( {\bm{\mathrm{A}}}  \phi({\bm{\mathrm{B}}} {\bm{\mathrm{x}}} ) ),
$$

where $\varphi$ and $\phi$ are the two non-linearities, and
${\bm{\mathrm{A}}}$ and ${\bm{\mathrm{B}}}$ are the two weight matrices.
We are ignoring the bias terms for the sake of exposition clarity. To
analyze the influence of the last (second) layer, we denote its input as
${\bm{\mathrm{y}}}' =  \phi({\bm{\mathrm{B}}} {\bm{\mathrm{x}}} )$, and
the input to the second layer activation function as
${\bm{\mathrm{z}}} = {\bm{\mathrm{A}}}{\bm{\mathrm{y}}}'$. In this
notation, we have
${\bm{\mathrm{y}}} = \varphi({\bm{\mathrm{A}}} {\bm{\mathrm{y}}}')$.
According to the chain rule,

$$
\frac{\partial L}{\partial {\bm{\mathrm{A}}}} =
\frac{\partial {\bm{\mathrm{y}}} }{\partial {\bm{\mathrm{A}}}} \frac{\partial L}{\partial {\bm{\mathrm{y}}}} =
\sum_{j=1}^k \frac{\partial y_j }{\partial {\bm{\mathrm{A}}}} \frac{\partial L}{\partial y_j}.
$$

For convenience, let us adopt the standard deep learning notation,
according to which the derivative of the loss w.r.t. to a parameter
${\bm{\mathrm{*}}}$ is denoted as $\delta {\bm{\mathrm{\ast}}}$. In our
case,

$$
\delta {\bm{\mathrm{y}}} = \frac{\partial L}{\partial {\bm{\mathrm{y}}}} =
\left( \frac{\partial L}{\partial y_1},\dots,  \frac{\partial L}{\partial y_k} \right)^\Tr
$$

is the gradient of the loss w.r.t. its input, and
$\delta {\bm{\mathrm{A}}}$ is a matrix whose elements are
$\frac{\partial L}{\partial a_{ij} }$, etc. In this notation, we can
rewrite

$$
\delta {\bm{\mathrm{A}}} =
\sum_{j=1}^k \frac{\partial y_j }{\partial {\bm{\mathrm{A}}}} \, \delta  y_j.
$$

We can write $\frac{\partial y_j }{\partial {\bm{\mathrm{A}}}}$ as a
matrix of the size of ${\bm{\mathrm{A}}}$, filled with zeros except the
$j$-th row, which is given by
$\varphi'(z_j) {\bm{\mathrm{y}}}^{\prime \Tr}$. Substituting this result
into the former sum yields

$$
\delta {\bm{\mathrm{A}}} =
\left(
    \begin{array}{c}
        \delta y_1 \varphi'(z_1)  {\bm{\mathrm{y}}}^{\prime \Tr} \\ \vdots \\ \delta y_k \varphi'(z_k)
        {\bm{\mathrm{y}}}^{\prime \Tr}
    \end{array}
\right) =
\mathrm{diag}\{ \delta {\bm{\mathrm{y}}} \}  \, \mathrm{diag}\{ \varphi'({\bm{\mathrm{z}}}) \} 
\left(
    \begin{array}{c}
        {\bm{\mathrm{y}}}^{\prime \Tr} \\
        \vdots \\
        {\bm{\mathrm{y}}}^{\prime \Tr}
    \end{array}
\right) =
\mathrm{diag}\{ \delta {\bm{\mathrm{y}}} \}  \, \mathrm{diag}\{
\varphi'({\bm{\mathrm{z}}}) \} {\bm{\mathrm{1}}} {\bm{\mathrm{y}}}^{\prime \Tr}.
$$

To analyze the influence of the first layer, we denote ${\bm{\mathrm{z}}}' =
{\bm{\mathrm{B}}}{\bm{\mathrm{x}}}$. To derive the gradient of the loss w.r.t.
the first layer parameter ${\bm{\mathrm{B}}}$, we again invoke the chain rule

$$
\frac{\partial L}{\partial {\bm{\mathrm{B}}}} = \frac{\partial
{\bm{\mathrm{y}}}' }{\partial {\bm{\mathrm{B}}}} \frac{\partial L}{\partial
{\bm{\mathrm{y}}}'} = \frac{\partial {\bm{\mathrm{y}}}' }{\partial
{\bm{\mathrm{B}}}} \frac{\partial {\bm{\mathrm{y}}} }{\partial
{\bm{\mathrm{y}}}'}  \frac{\partial L}{\partial {\bm{\mathrm{y}}}} = \sum_{j=1}
\frac{\partial y'_j }{\partial {\bm{\mathrm{B}}}} \, \delta y'_j.
$$

As before, $\frac{\partial y'_j }{\partial {\bm{\mathrm{B}}}}$ is a matrix of
the size of ${\bm{\mathrm{B}}}$, filled with zeros except the $j$-th row, which
is given by $\phi'(z'_j) {\bm{\mathrm{x}}}^\Tr$, so

$$
\delta {\bm{\mathrm{B}}} =
\mathrm{diag}\{ \delta {\bm{\mathrm{y}}}' \}  \, \mathrm{diag}\{
\phi'({\bm{\mathrm{z}}}') \}  {\bm{\mathrm{1}}} {\bm{\mathrm{x}}}^\Tr.
$$

It remains to derive

$$
\partial {\bm{\mathrm{y}}}' =
\frac{\partial L}{\partial {\bm{\mathrm{y}}}'} = \frac{\partial
{\bm{\mathrm{y}}} }{\partial {\bm{\mathrm{y}}}'}  \frac{\partial L}{\partial
{\bm{\mathrm{y}}}}.
$$

From ${\bm{\mathrm{y}}} = \varphi({\bm{\mathrm{A}}} {\bm{\mathrm{y}}}')$, we
have

$$
\frac{\partial {\bm{\mathrm{y}}} }{\partial {\bm{\mathrm{y}}}'} = \diag\{
\varphi'({\bm{\mathrm{z}}} )  \} {\bm{\mathrm{A}}}^\Tr,
$$

from where

$$
\delta {\bm{\mathrm{y}}}' =   \diag\{ \varphi'({\bm{\mathrm{z}}} )  \} {\bm{\mathrm{A}}}^\Tr \delta {\bm{\mathrm{y}}}.
$$

We can therefore summarize the chain rule in our two-layer MLP as
follows: First, we propagate the data *forward* through the network,
computing

$$
\begin{aligned}
{\bm{\mathrm{z}}}' &=& {\bm{\mathrm{B}}}{\bm{\mathrm{x}}}  \\
{\bm{\mathrm{y}}}' &=& \phi( {\bm{\mathrm{z}}}' ) \\
{\bm{\mathrm{z}}} &=& {\bm{\mathrm{A}}}{\bm{\mathrm{y}}}'  \\
{\bm{\mathrm{y}}} &=& \varphi( {\bm{\mathrm{z}}} ).
\end{aligned}
$$

Then,
we propagate the derivatives *backward* through the network:

$$
\begin{aligned}
\delta {\bm{\mathrm{y}}} &=& \nabla L( {\bm{\mathrm{y}}} ) \\
\delta {\bm{\mathrm{A}}} &=&  \mathrm{diag}\{ \delta {\bm{\mathrm{y}}} \}  \, \mathrm{diag}\{ \varphi'({\bm{\mathrm{z}}}) \}   {\bm{\mathrm{1}}} {\bm{\mathrm{y}}}^{\prime \Tr} \\
\delta {\bm{\mathrm{y}}}' &=&   \diag\{ \varphi'({\bm{\mathrm{z}}} )  \} {\bm{\mathrm{A}}}^\Tr \delta {\bm{\mathrm{y}}} \\
\delta {\bm{\mathrm{B}}} &=&  \mathrm{diag}\{ \delta {\bm{\mathrm{y}}}' \}  \, \mathrm{diag}\{ \phi'({\bm{\mathrm{z}}}') \}   {\bm{\mathrm{1}}} {\bm{\mathrm{x}}}^\Tr.
\end{aligned}
$$

The entire procedure, known as error backward propagation or
*backpropagation* for short can be applied recursively for any number of
layers.

#### Forward pass:

Starting with ${\bm{\mathrm{y}}}_0 = {\bm{\mathrm{x}}}$, compute for
$k=1,\dots, L$

-   ${\bm{\mathrm{z}}}\_k = {\bm{\mathrm{W}}}\_k {\bm{\mathrm{y}}}\_{k-1}$

-   ${\bm{\mathrm{y}}}_k = \varphi_k({\bm{\mathrm{z}}}_k)$

and output ${\bm{\mathrm{y}}} = {\bm{\mathrm{y}}}_L$.

#### Backward pass:

Starting with $\delta {y}_L = \nabla L( {\bm{\mathrm{y}}} )$, compute
for $k=L,L-1,\dots, 1$

-   $\delta {\bm{\mathrm{W}}}\_k =  \mathrm{diag}\{ \delta {\bm{\mathrm{y}}}\_k \}
    \, \mathrm{diag}\{ \varphi'\_k ({\bm{\mathrm{z}}}\_k) \}
    {\bm{\mathrm{1}}} {\bm{\mathrm{y}}}\_{k-1}^\Tr$

-   $\delta {\bm{\mathrm{b}}}_k =  \mathrm{diag}\{ \delta {\bm{\mathrm{y}}}_k \}
    \varphi'_k ({\bm{\mathrm{z}}}_k)$

-   $\delta {\bm{\mathrm{y}}}_{k-1} =   \diag\{ \varphi'_k({\bm{\mathrm{z_k}}} ) \}
    {\bm{\mathrm{W}}}_k^\Tr \delta {\bm{\mathrm{y}}}_k$

We remind that $\delta {\bm{\mathrm{W}}}_k$ and
$\delta {\bm{\mathrm{b}}}_k$ are blocks of coordinates of the gradient
of the loss $L$ with respect to the network parameters.

### Exploding and vanishing gradients

Backpropagation allows a recursive calculation of the loss gradient w.r.t. the
parameters of the network without the need to ever construct the Jacobian
matrices of each layer’s output w.r.t. its input. Note, however, that in order
to compute the gradient w.r.t. the first layer, $\delta {\bm{\mathrm{W}}}_1$,
one need to compute the product of $\varphi'_L
({\bm{\mathrm{z}}}_L),\dots,\varphi'_1 ({\bm{\mathrm{z}}}_1)$.  This may lead to
numerical instabilities. For example, in a network with $L=20$ layers, a slopeof
$\varphi' = 2$ in each activation function would be amplified by $10^6$.
Similarly, a slope of $\varphi' = 0.5$ would diminish to $10^{-6}$ – practically
to zero. This problem is known as vanishing and exploding gradients, and it
prevented end-to-end supervised training of deep neural networks from random
initialization.

The introduction of ReLU activations mitigated this problem. In ReLU, the
derivative is $1$ for positive arguments and $0$ for negative ones.  This
implies that depending on the path through the network from the output back to
the inputs, the product of the activation derivatives will always be either $0$
or $1$. The $0$ derivative for negative arguments could still lead to vanishing
gradients, but practice shows that, on the contrary, it helps optimization and
promotes sparse solutions.

ReLU was probably one of the few significant algorithmic changes in the
classical neural networks that enabled deep learning.

## Convolutional neural networks

The layers on MLP described so far are termed *fully connected* in the deep
learning literature, due to the fact that every layer input is connected
(through some weight) to every output. For large input and output dimensions,
such an architecture results in a vast number of degrees of freedom, which
increases the network complexity and requires more data to train.

### Weight sharing and shift invariance

*Weight sharing* is a strategy aiming at reducing the layer complexity by
reusing the same weights at different parts of the input. For the sake of the
following discussion, we assume the input to be discrete and infinitely
supported (i.e., a sequence ${\bm{\mathrm{x}}} = \{ x\_i \},~{i \in \mathbb{Z}}$).
The output is also assumed to be a sequence,
${\bm{\mathrm{y}}} = \{ y\_i \},~{i \in \mathbb{Z}}$.
Let us consider the output of the $i$-th neuron,

$$
y_i = \varphi\left( \sum_{j \in \mathbb{Z}} w_{ij} x_j + b+i \right).
$$


In many cases such as audio signals, images, etc., it is reasonable to assume
that the same operation is valid at different parts of the signal.
Mathematically, this can be expressed by asserting that the action of the neuron
commutes with the action of a translation group.  This leads to demanding

$$
\varphi\left( \sum_{j \in \mathbb{Z}} w_{i-m,j} x_j + b_{i-m} \right) =
\varphi\left( \sum_{j \in \mathbb{Z}} w_{ij} x_{j-m} + b_i \right)
$$

for every input ${\bm{\mathrm{x}}}$. Since the non-linearity is applied
element-wise, the equivalent condition holds on its arguments as well,

$$
\sum_{j \in \mathbb{Z}} w_{i-m,j} x_j  + b_{i-m} =
\sum_{j \in \mathbb{Z}} w_{ij} x_{j-m} + b_i =
\sum_{j' \in \mathbb{Z}} w_{i,j'+m} x_{j'} + b_i.
$$

This implies $b_i = \mathrm{const}$ and $w_{i-m,j} = w_{i,j+m}$; in
other words, if we consider $w_{ij}$ to be the elements of an inifinite
weight matrix, it will have equal elements on each of its diagonals.
Another way to express is is by saying that $w_{ij}$ is a function of
$i-j$.

### Toeplitz operators and convolution

A linear operator exhibiting the above structure is called *Toeplitz*.  The
output of a shift-invariant (Toeplitz) neuron can be written as

$$
y_i = \varphi\left( \sum_{j \in \mathbb{Z}} w_{i-j} x_j + b \right).
$$

Note that the weights ${\bm{\mathrm{w}}}$ can now be considered as a window that
is applied to the input at a certain location to produce an output at the same
location, and then is slided to a different input location to produce the
corresponding output. This operation (the application of the Toeplitz operator)
called *convolution*, denoted as

$$
(w \ast x)_i =
\sum_{j \in \mathbb{Z}} w_{i-j} x_j =
\sum_{j \in \mathbb{Z}} w_j x_{i-j} =
(x \ast w)_i.
$$

In this notation, the action of our layer can be written as

$$
{\bm{\mathrm{y}}} =
\varphi\left( {\bm{\mathrm{w}}} \ast {\bm{\mathrm{x}}} + b \right).
$$

In the signal processing jargon, we can say that the input signal
${\bm{\mathrm{x}}}$ is filtered by a filter with the impulse response
${\bm{\mathrm{w}}}$.

### Convolutional layer

Neural networks making use of shift-invariant linear operations are called
*convolutional neural networks* (CNNs). A convolutional layer accepts an
$m$-dimensional *vector-valued* infinitely supported signal
${\bm{\mathrm{x}}} = ({\bm{\mathrm{x}}}^1,\dots, {\bm{\mathrm{x}}}^m) =
\{ (x\_i^1,\dots, x\_i^m) \}\_{i \in \mathbb{Z}}$;
each input dimension is called a *channel* or *feature map*.
The layer produces an $n$-dimensional infinitely supported signal
${\bm{\mathrm{y}}} = ({\bm{\mathrm{y}}}^1,\dots, {\bm{\mathrm{y}}}^n) =
\{ (y_i^1,\dots, y_i^n) \}_{i \in \mathbb{Z}}$ by applying a bank of filters,

$$
{\bm{\mathrm{y}}}^j =
\varphi\left(  \sum_{i=1}^m {\bm{\mathrm{w}}}^{ij} \ast {\bm{\mathrm{x}}}^{i}  \right),
$$

or, explicitly,

$$
y^j_k = \varphi\left(  \sum_{i=1}^m \sum_{p} w^{ij}_p x^i_{k-p}  \right).
$$

In practice, each filter $w^{ij}$ is supported on some small fixed
domain.

