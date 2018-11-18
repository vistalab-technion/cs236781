---
title: "Lecture 5: Recurrent Neural Networks"
excerpt: Foo
copyright: alex
---

Multi-layered perceptrons require the input to be of a fixed dimension,
and produce an output of a fixed dimension. While CNNs can overcome this
limitation, they still lack the notion of *persistence* that is so
typical to the human thinking process. For example, when trying to
classify what event is happening at every frame in a video, traditional
neural networks lack the mechanism to use the reasoning about previous
events to inform the later ones. A natural way to introduce such a
persistence is by using *feedback* or recurrence. Enter recurrent neural
networks (a.k.a. RNNs).

Basic building blocks of an RNN
===============================

Probably the most useful metaphor to describe a recurrent neural network
is that of a (nonlinear) dynamical system. The network receives a
*sequential* input $\bb{x}_t \in \RR^n$ (we will henceforth denote
sequences using the time parameter $t$ which will be assumed discrete)
and a previous *hidden state* vector $\bb{h}_{t-1} \in \RR^k$. The
network applies some shift-invariant (that is, time-independent)
parametric function $f_{\bb{\Theta}}$ to produce the next hidden state
vector, $$\bb{h}_t = f_{\bb{\Theta}} ( \bb{h}_{t-1}, \bb{x}_t ).$$ The
hidden state is initialized with some $\bb{h}_0$ which can be learned,
but more frequently is set to zero. If the network is to produce some
output, an additional parametric function (e.g., a fully-connected
layer) is applied to the current hidden state $\bb{h}_t$ to produce the
output $\bb{y}_t \in \RR^m$,
$$\bb{y}_t = g_{\bb{\Theta}} ( \bb{h}_{t} ).$$

The simplest RNN that has been proposed in the literature has the form
of a fully-connected layer
$$\bb{h}_t = \varphi(\bb{W}_{hh} \bb{h}_{t-1} + \bb{W}_{xh} \bb{x}_t  + \bb{b}),$$
usually with the hyperbolic tangent activation $\varphi(x) = \tanh(x)$,
to produce the state update, and another fully connected linear layer of
the form $$\bb{y}_t = \bb{W}_{hy} \bb{h}_t.$$ Here, the weight matrices
$\bb{W}_{hh} \in \RR^{k \times k}$, $\bb{W}_{xh} \in \RR^{k \times n}$,
$\bb{W}_{hy} \in \RR^{m \times k}$ and the bias vector
$\bb{b} \in \RR^{k}$ constitute the network parameters that we
collectively denote as $\bb{\Theta}$.

Various settings
----------------

There are several ways to use an RNN that depend on the specific task:

### Many-to-one

the network consumes a sequence of inputs
$\{ \bb{x}_1,\dots, \bb{x}_T\}$ and produces a sequence of hidden states
$\{ \bb{h}_1,\dots, \bb{h}_T\}$ (starting with some $\bb{h}_0$ that is
either learned or left fixed). The last state $\bb{h}_T$ is fed to the
output layer producing a single output $\bb{y}$ for the entire input
sequence. This approach is typically used for classifying varying length
inputs such as in sentiment analysis of text.

### One-to-many

the network consumes a single input vector $\bb{x}$ and an initial state
$\bb{h}_0$ to produce the first hidden state $\bb{h}_1$. It then
produces a sequence of hidden states $\{ \bb{h}_2,\dots, \bb{h}_T\}$
without taking any additional input. Alongside, a sequence
$\{ \bb{y}_1,\dots, \bb{y}_T\}$ of outputs is generated. This
architecture is often used in image annotation, when the input is a
single image (actually, more frequently, its representation in the form
of a feature vector produced by a CNN) and the output is a (variable
length) text sequence describing the image.

### Many-to-many

two networks are concatenated in the form of an encoder-decoder
architecture. The many-to-one encoder network consumes a sequential
input producing a sequence of state vectors. The last state is fed into
a one-to-many decoder network (usually, this state is given as the
initial state of the decoder, which receives no input), and the network
produces an output sequence. This architecture is common in tasks where
the length of the input and that of the output are both variable and,
potentially, distinct. A bold example is machine translation. For
example, translating the English sentence *A cat ate a mouse* (5 words)
into the Italian sentence *Il gatto ha mangiato un topo* (6 words, note
that the word “ate” corresponds to two word “ha mangiato”) or to the
German sentence *Die Katze hat eine Maus gegessen* (6 words, note the
different order of the verb).

### Layered RNN

The RNN takes its strength from the sequential behavior – in fact, one
can unroll the network action into a very *long* (infinite) feedforward
network. However, nothing prevents from adding *depth* to increase the
descriptive capacity of the network in producing the output sequence. To
that end, we can stack $L$ RNNs one on top of the other, each with its
own hidden state $\{ \bb{h}_t^l \}$ and parameters $\bb{\Theta}^l$. The
lowest (input) layer receives the input sequence
$\{ \bb{y}_t^0 = \bb{x}_t \}$ and produces the output sequence
$\{ \bb{y}_t^1 \}$ that is fed as the input (at the same times) to the
subsequent layer. The output of the final layer
$\{ \bb{y}_t^L = \bb{y}_t \}$ serves as the output sequence of the
network.

$$\begin{aligned}
\bb{h}_t^1 &=& f_{\bb{\Theta}^1} ( \bb{h}^1_{t-1}, \bb{x}_t ) \\
\bb{y}_t^1 &=& g_{\bb{\Theta}^1} ( \bb{h}_{t}^1 ) \\
\bb{h}_t^2 &=& f_{\bb{\Theta}^2} ( \bb{h}^2_{t-1}, \bb{y}^1_t ) \\
\bb{y}_t^2 &=& g_{\bb{\Theta}^2} ( \bb{h}_{t}^2 ) \\
\vdots \\
\bb{h}_t^L &=& f_{\bb{\Theta}^L} ( \bb{h}^L_{t-1}, \bb{y}^{L-1}_t ) \\
\bb{y}_t &=& g_{\bb{\Theta}^L} ( \bb{h}_{t}^L ).\end{aligned}$$

A trained RNN can also be used as a generative model – we will discuss
this in the sequel when dealing with generative models.

Training RNNs
=============

Let us now discuss the training a recurrent network. Let us assume that
the RNN is given by the recursive relation $$\begin{aligned}
\bb{h}_t &=& f_{\bb{\Theta}} ( \bb{h}_{t-1}, \bb{x}_t ) \\
\bb{y}_t &=& g_{\bb{\Theta}} ( \bb{h}_{t} )\end{aligned}$$ and the loss
function is evaluated as the sum of individual losses over each time
sample of the output sequence,
$$L(\bb{\Theta}) = \sum_{t > 0} \ell_t( \bb{y}_t )$$ (for example, we
can apply a softmax function to the outputs and evaluate a cross-entropy
loss).

In order to compute the gradient of the loss w.r.t. the network
parameters $\bb{\Theta}$, one needs to perform the forward pass on the
entire input sequence. In theory, the input is of infinite length; in
practice, it has some finite lenght $T$ (possibly very big). The chain
rule yields
$$\delta \bb{\Theta} =  \frac{\partial L(\bb{\Theta}) }{\partial \bb{\Theta}} = \sum_{1 \le t \le T}   \frac{ \partial  \ell_t  }{\partial \bb{\Theta}},$$
with
$$\frac{ \partial  \ell_t }{\partial \bb{\Theta}}= \frac{\partial \ell_t }{\partial   \bb{y}_t}
 \left( 
 \frac{\partial^+ \bb{y}_t }{\partial \bb{\Theta}} + 
 \sum_{1 \le i \le t}   
\frac{ \partial \bb{y}_t }{\partial \bb{h}_t }     
\frac{ \partial \bb{h}_t }{\partial \bb{h}_i }  
 \frac{ \partial^+ \bb{h}_i }{\partial \bb{\Theta}} 
 \right)
,$$ where
$$\frac{ \partial^+ \bb{h}_i }{\partial \bb{\Theta}} = \frac{ \partial  f  }{\partial \bb{\Theta}} ( \bb{h}_{i-1} , \bb{x}_i  )$$
refers to the “immediate” derivative of $\bb{h}_i$ w.r.t. $\bb{\Theta}$,
assuming $\bb{h}_{t-1}$ constant, and
$$\frac{ \partial^+ \bb{y}_t }{\partial \bb{\Theta}} = \frac{ \partial  g  }{\partial \bb{\Theta}} ( \bb{h}_{t}  )$$
refers to the “immediate” derivative of $\bb{y}_t$ w.r.t. $\bb{\Theta}$,
assuming $\bb{h}_{t}$ constant. We can immediately susbstitute the
partial derivative
$$\frac{ \partial \bb{y}_t }{\partial \bb{h}_t }  =  \frac{ \partial  g  }{\partial \bb{h}} ( \bb{h}_{t}  ).$$
The Jacobian of $\bb{h}_t$ w.r.t. $\bb{h}_i$ is evaluated invoking the
chain rule $$\frac{ \partial \bb{h}_t }{\partial \bb{h}_i }  = 
\frac{ \partial \bb{h}_t }{\partial \bb{h}_{t-1} } 
 \frac{ \partial \bb{h}_{t-1} }{\partial \bb{h}_{t-2} }   \cdots 
   \frac{ \partial \bb{h}_{i+1} }{\partial \bb{h}_{i} }
= 
 \frac{ \partial  f  }{\partial \bb{h}}( \bb{h}_{t-1} , \bb{x}_t) \, \frac{ \partial  f  }{\partial \bb{h}}( \bb{h}_{t-2} , \bb{x}_{t-1})
 \cdots
  \frac{ \partial  f  }{\partial \bb{h}}( \bb{h}_{i} , \bb{x}_{i+1}).$$

Note that each term $\frac{ \partial  \ell_t }{\partial \bb{\Theta}}$
has the same form, and the behavior of these terms determine the
behavior of the entire sum. Every gradient component
$\frac{ \partial  \ell_t }{\partial \bb{\Theta}}$ is, in turn, also a
sum whose terms $$\frac{\partial \ell_t }{\partial   \bb{y}_t}
\frac{ \partial \bb{y}_t }{\partial \bb{h}_t }     
\frac{ \partial \bb{h}_t }{\partial \bb{h}_i }  
 \frac{ \partial^+ \bb{h}_i }{\partial \bb{\Theta}}$$ can be interpreted
as *temporal contributions*, measuring how $\bb{\Theta}$ at step $i$
affects the loss at step $t > i$. The factors
$\frac{ \partial \bb{h}_t }{\partial \bb{h}_i }$ have the role of
transporting the error “back in time” from $t$ to $i$. *Long-term
contributions* correspond to $i \ll t$, while *short-term contributions*
to $i \sim t$. In the sequel, we will analyze the dynamics of these
factors to unearth major numerical issues associated with
backpropagation through RNNs.

Backpropagation through time
----------------------------

The metaphor of an RNN unrolled into a (very long) feed-forward network
allows to see immediately that the chain rule we saw before is exactly
the backpropagation rule we have already encountered for MLPs. The only
difference is that now each layer depends on the *same* parameters,
hence the additional sum over $i$ arises in the calculations. Instead of
computing individual gradients w.r.t. the parameters of each layer as we
did in MLPs or CNNs, they are accumulated into a single gradient of the
shared parameters. Such a backward step is known under the name of
*backpropagation through time* or BPTT.

Since RNN input sequences might be very long in practice (consider, for
example, the entire set of wikipedia text), BPTT is rarely used as is.
Instead, the forward pass is performed in chunks of a fixed number of
time samples (still keeping the state created from the beginning of the
input sequence), followed by backpropagation performed for the same
number of steps backwards in time. This training strategy is known as
*truncated* backpropagation through time (TBPTT) and is much more
practical computationally.

Vanishing and exploding gradients
---------------------------------

Substituing the particular parametrization $$\begin{aligned}
\bb{z}_t &=& \bb{W}_{hh} \bb{h}_{t-1} + \bb{W}_{xh} \bb{x}_t + \bb{b} \\
\bb{h}_t &=& \varphi( \bb{z}_t ) \\
\bb{y}_t &=&  \bb{W}_{hy} \bb{h}_t\end{aligned}$$ we can write
$$\frac{ \partial \bb{y}_t }{\partial \bb{h}_t }  = \bb{W}_{hy}^\Tr.$$
and
$$\frac{ \partial \bb{h}_i }{\partial \bb{h}_{i-1} }  =  \bb{W}_{hh}^\Tr \, \mathrm{diag}\{ \varphi'( \bb{z}_i ) \}.$$
The immediate derivative of $\bb{h}_t$ w.r.t. the weight matrix
$\bb{W}_{hh}$ is a rank-3 tensor, so we will write its product with the
gradient of the loss w.r.t. $\bb{h}_t$
$$\frac{ \partial^+ \bb{h}_i }{\partial \bb{W}_{hh}} \delta \bb{h}_i =  \mathrm{diag}\{ \delta \bb{h}_i \} \mathrm{diag}\{ \varphi'( \bb{z}_i ) \} \bb{1}\bb{h}_{i-1}^\Tr$$
where $\delta \bb{h}_t = \frac{\partial \ell_t }{\partial   \bb{h}_t}$.
Using these calculations, we can express each temporal contribution to
the gradient as $$\frac{\partial \ell_t }{\partial   \bb{y}_t}
\frac{ \partial \bb{y}_t }{\partial \bb{h}_t }     
\frac{ \partial \bb{h}_t }{\partial \bb{h}_i }  
 \frac{ \partial^+ \bb{h}_i }{\partial \bb{W}_{hh}}  =  \mathrm{diag}\{\delta \bb{y}_t \} \bb{W}_{hy}^\Tr  \bb{W}_{hh}^\Tr \, \mathrm{diag}\{ \varphi'( \bb{z}_{t} ) \}  \cdots  \bb{W}_{hh}^\Tr \, \mathrm{diag}\{ \varphi'( \bb{z}_{i+1} ) \}  \mathrm{diag}\{ \varphi'( \bb{z}_i ) \} \bb{1}\bb{h}_{i-1}^\Tr.$$
Observe that due to the term
$\frac{ \partial \bb{h}_t }{\partial \bb{h}_i } $, the weight matrix
$\bb{W}_{hh}$ and the diagonal matrix $\mathrm{diag}\{ \varphi' \}$
appear $t-i-1$ times in the product.

Let us now analyze the influence of the above product on the long-term
($i \ll t$) contributions. For the sake of simplicity, we assume that
$\varphi = \mathrm{id}$, leaving us with $(\bb{W}_{hh}^\Tr)^l$,
$l=t-i \gg 1$. We define the spectral radius of the matrix,
$\rho( \bb{W}_{hh} )$, as its maximum absolute eigenvalue. Simple linear
algebra suggests that if $\rho( \bb{W}_{hh} ) < 1$, then
$(\bb{W}_{hh}^\Tr)^l$ vanishes exponentially as $l$ approaches infinity,
and, hence, the long-term contribution to the gradient will vanish. In
the opposite situation when $\rho( \bb{W}_{hh} ) > 1$,
$(\bb{W}_{hh}^\Tr)^l$ will magnify exponentially some directions
corresponding to the eigenvalues bigger than $1$ in the absolute value;
however, directions corresponding to eigenvalues smaller that $1$ will
shrink. Hence, $\rho( \bb{W}_{hh} ) > 1$ is a necessary condition for
the exploding long-term contributions, but not a sufficient one.

If we now assume that $\varphi$ is not identity, yet has a bounded
derivative, $|\varphi'| < \gamma$, the above result straightforwardly
suggests $\rho( \bb{W}_{hh} ) < 1/\gamma$ being a sufficient condition
for vanishing long-term gradients, and $\rho( \bb{W}_{hh} ) > 1/\gamma$
a necessary condition for exploding long-term gradients.

Numerical tricks
----------------

Vanishing and exploding gradients have been plaguing RNNs for many
years, not allowing their efficient training. Several numerical tricks
have been proposed to partially overcome these problems. One of the
straightforward ideas is to control the gradient scale by the following
heuristic: if $\| \bb{g} \| > \tau$, modify it to
$\frac{\tau \bb{g}}{\| \bb{g} \| }$.

The insights into the importance of the spectral radius of the weight
matrix suggests a way to safely initialize it: Suppose $\bb{W}_{hh}$ is
initialized to some random values. We compute the spectral radius
$\rho = \rho(\bb{W_{hh}})$ and scale it by $\frac{c}{\gamma \rho}$ with
$c \approx 1.1$. With such a setting, the gradients are guaranteed not
to vanish but will neither explode too rapidly.

Another heuristic comes from the empirical observation that when the
gradients explode, they do so along some direction, and the curvature of
the loss function (expressed via the corresponding second-order
directional derivative) also explodes. Hence, second-order optimization
methods relying on the Hessian matrix should be capable of scaling the
gradient components according to the curvature and make small steps in
the direction of the exploding gradient/curvature. This explains the
success of second-order methods in RNN training. While a full Newton
step is prohibitively expensive, its truncated version or the
Gauss-Newton method approximating the Hessian via an outer product of
gradients are known to perform very well and be less sensitive to the
exploding gradients issues.

Yet another way to avoid exploding or vanishing gradients is by means of
regularization made such that the back-propagated gradients neither
increase or decrease too much in magnitude. This can be achieved by
adding a regularization term of the form $$R = \sum_{k} \left(  
\frac{\left\|  \delta \bb{h}_{k+1}  \frac{\partial \bb{h}_{k+1}}{ \partial \bb{h}_{k} }  \right\| }{\| \delta \bb{h}_{k+1}  \| } - 1
\right)^2.$$ To make the regularization tractable, its derivatives are
approximated only as the “immediate” derivatives w.r.t. $\bb{W}_{hh}$.

Gated recurrent units
=====================

Another way to address the vanishing gradients problem is by modifying
the structure of the recurrent cell in order to avoid the product with
$\bb{W}_{hh}$ in the backward step. A basic *gated recurrent unit* (GRU)
consists of two gate signals, the *update gate*
$$\bb{z}_t = \sigma( \bb{W}_{hz} \bb{h}_{t-1} + \bb{W}_{xz} \bb{x}_t + \bb{b}_z  )$$
and the *reset gate*
$$\bb{r}_t = \sigma( \bb{W}_{hr} \bb{h}_{t-1} + \bb{W}_{xr} \bb{x}_t + \bb{b}_r  )$$
computed using a linear transfer function followed by the sigmoid
activation with dedicated parameters. The sigmoid scales each gate
signal between $0$ and $1$.

A candidate state update is calculated from the current input and the
previous state as usual, with the only difference that now the
contribution of the previous state is gated by the reset gate signal,
$$\bb{q}_t = \varphi( \bb{r}_t \odot \ \bb{W}_{hh} \bb{h}_{t-1} + \bb{W}_{xh} \bb{x}_t  + \bb{b}_q ).$$
Here $\odot$ denotes Hadamard (element-wise) product and
$\varphi = \tanh$. This gating can be thought of as a soft (and,
consequently, differentiable) version of an if...then condition allowing
to selectively use only some coordinates of the transformed previous
state and thus forget how much of the previous context to carry to the
next time step.

The final state update is computed by blending the candidate new state
and the previous state, with the blending controlled element-wise by the
update gate,
$$\bb{h}_t = \bb{z}_t \odot \bb{h}_{t-1} + (1-\bb{z}_t) \odot \bb{q}_t.$$
The update gate controls how much of the past information to forget.

Invoking the chain rule again, we obtain $$\begin{aligned}
\frac{\partial \bb{h}_t}{\partial \bb{h}_{t-1} } &=& \mathrm{diag}\{ \bb{z}_t  \} +   \bb{W}_{hz}^\Tr \, \mathrm{diag}\{ \sigma'( \bb{W}_{hz} \bb{h}_{t-1} + \bb{W}_{xz} \bb{x}_t + \bb{b}_z ) \} \mathrm{diag}\{ \bb{h}_{t-1} - \bb{q}_t \}  \\
&& + \bb{W}_{hh}^\Tr \, \mathrm{diag}\{ \sigma'( \bb{W}_{hh} \bb{h}_{t-1} + \bb{W}_{xh} \bb{x}_t + \bb{b}_h ) \} \mathrm{diag}\{ 1-\bb{z}_{t-1} \}.\end{aligned}$$
Note that, as before, this term contains terms containing
$\bb{W}_{hh}^\Tr$, however, now a free term depending only on $\bb{z}_t$
is added. Therefore, when the update gate is open $\bb{z}_t \approx 1$,
the gradients flow backward in time uninmpeded. This allows a gated RNN
to learn long-term context without significant numerical issues.

Variants of the gated architecture exist, with the most popular one
being the Long Short Term Memory (LSTM) cell.

Attention 
==========

Informally, an *attention* mechanism equips a NN with the ability to
focus on a subset of its inputs (or intermediate features) by selecting
or emphasizing specific inputs. This resembles, for example, to human
visual attention – the foveal region of the retina where the sampling
resolution is the highest has an angular aperture of only 1 degree. It
is the constant motion of our sight that allows us to actually see. The
pattern of the motion is driven by an attention mechanism that depends
on content of what we see.

A similar mechanism can be built into a neural network. The basic idea
of attention is similar to gating, except that it happens in paraller
rather than being serial as in GRUs. Consider a regular one-to-many RNN
receiving a single input $\bb{x}$ and producing a sequence of hidden
states $\{ \bb{h}_t \}$ via the function
$$\bb{h}_t = f_{\bb{\Theta}}(\bb{h}_{t-1}, \bb{x}_t) = \varphi( \bb{W}_{hh} \bb{h}_{t-1} + \bb{W}_{xh} \bb{x}_t  + \bb{b}_h).$$
For example, such a network could produce a textual annotation of an
image. In an attention network, instead of feeding $\bb{x}$ directly to
the RNN, a *gated* version thereof is computed,
$$\bb{z}_t = \bb{x}_t \odot \bb{g}_t$$ and the network is applied to
$\bb{z}$,
$$\bb{h}_t = \varphi( \bb{W}_{hh} \bb{h}_{t-1} + \bb{W}_{xh} \bb{z}_t  + \bb{b}_h).$$
Note that the gate signal is applied simultaneously to all elements of
the input and can me thought of as a soft (and, consequently,
differentiable) version of masking.

The gate $\bb{g}$ itself is computed from the input and the current
state by means of another parametric function
$$\bb{a}_t = \varphi( \bb{W}_{ha} \bb{h}_{t-1} + \bb{W}_{xa} \bb{z}_t  + \bb{b}_a).$$
This intermediate signal is then soft-maxed (with the temperature
parameter $\alpha$ controlling the sharpness of the softmax),
$$\bb{g}_t = \frac{e^{ \alpha \bb{a}_t }}{\bb{1}^\Tr re^{ \alpha \bb{a}_t }}.$$

Obviously, and RNN can use both gating (e.g., LSTM) and attention
mechanisms – they serve different purposes.

Alternatives to RNNs
====================

While the invention of gated RNNs such as LSTM and some numerical
heuristics mentioned before have significantly improved the ability to
train RNNs and learn long-term dependencies in the thousands of time
samples, recurrent networks still suffer from their inherently
sequential structure, which is bad both for efficient training and
inference. Today, a growing amount of evidence suggests that properly
designed CNNs constitute a formidable alternative to RNNs. Using a
signal processing metaphor, RNNs resemble infinite impulse response
(IIR) filters that are sequential and are based on recursion, while
alternatives are more like finite impulse response filters (FIRs) that
are easily parallelizable. While RNNs can in theory learn arbitrarily
long term dependencies, this never happens in practice, and a
sufficiently deep CNN can achieve similar performance. In what follows,
we briefly overview the time convolutional network (TCN) architecture.

Time convolutional network (TCN)
--------------------------------

TCN can be viewed as a FIR equivalent of an RNN. The main element of TCN
is the *dilated* convolution defined as
$$(\bb{x} \ast_d \bb{w} )_n = \sum_{k} w_k x_{n - dk},$$ where $\bb{x}$
is the input sequence, $\bb{w}$ is the filter impulse response, and
$d \in \bb{N}$ is the dilation factor. Dilation factor $d$ essentially
introduces a fixed step between the filter taps. For $d=1$, dilated
convolution reduces to its regular counterpart. For time signals, the
filters are made causal ($w_k = 0$ for $k<0$). The receptive field of a
dilated convolution with the filter of length $K$ is $Kd$. Using
exponentially increasing dilation factor in a deep network, $d = 2^l$,
with $l$ being the layer number, essentially allows a relatively shallow
network to have a very long history in time.

Causal dilatied convolutions can be combined with other standard CNN
architectural choices such as residual connections, weight decay, batch
normalization, drop out, etc.
