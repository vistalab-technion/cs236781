---
title: "Lecture 6: Unsupervised learning and generative models"
excerpt: Autoencoders, adjoint convolutions and pooling, variational autoencoders, generative adversarial nets
copyright: alex
---

The only training regime we have seen so far for deep neural models was
*supervised*: the network was given a set of example instances and
trained so to minimize a discrepancy between the output produced by the
network and the corresponding desired outputs. The latter outputs could
be either class labels in classification problems, or some continuous
scalars or vectors in regression problems. In what follows, we extend
our machine learning arsenal to the *unsupervised* setting, in which
only example instances and no example outputs are provided. The goal of
unsupervised learning is to discover and model the structure of the
instance space. One of the most useful applications of such models is to
*generate* new examples of instances.

Autoencoders
============

One of the most powerful ideas that humans have invented to model nature
is that of parsimony. The most famous statement of this heurisitc is
attribuited to the Franciscan friar William of Ockham, who suggested
that when presented with two competing hypothetical answers to a
problem, one should select the one that makes the fewest assumptions. In
the realm of data science, Ockham's razor can be translated to the
assertion that many types of data such as signals and images despite
being naturally embedded in a very high dimensional space, possess a
much smaller number of degrees of freedom. For example, the space of all
$255 \times 255$ $8$-bit images theoretically admits about $10^{157826}$
distinct instances, the overwhelming majority of which do not look like
a natural image. It can therefore be posited that natural images can be
described as a low-dimensional manifold embedded into the
$65536$-dimensional space. Techniques for discovering this manifold (or,
in the probabilistic setting, a probability distribution supported on a
low-dimensional manifold) are called *dimensionality reduction*.

Deep learning offers a formidable tool for dimensionality reduction in
the form of *autoencoders*. An autoencoder can be viewed as a
composition of two neural networks, and *encoder*
$\Phi : \mathcal{X} \rightarrow \mathcal{Z}$ and a *decoder*
$\Psi : \mathcal{Z} \rightarrow \mathcal{X}$. Given an instance
$\bb{x} \in \mathcal{X}$, the encoder produces a code vector
$\bb{z} = \Phi(\bb{x})$ called a *latent representation* (the space
$\mathcal{Z}$ of such representations is called the *latent space*). The
role of the decoder is to reproduce an instance from its encoding,
$\bb{x} = \Psi(\bb{z})$.

Ideally, an autoencoder should be such that the concatenaton of the
encoder with the decoder results in an identity transformation,
$\Psi \circ \Phi \approx \mathrm{id}$. As long as the dimension of the
latent space $\mathcal{Z}$ is lower than that of the instance space
$\mathcal{X}$, $\Psi \circ \Phi$ cannot be a trivial identity map and
has to actually reveal the hidden (and, potentially, very non-trivial)
degrees of freedom in the data. The combination of a
dimensionality-reducing encoder followed by a dimensionality-increasing
encoder forces the data to pass through a *bottleneck*. Both networks
are trained on example instances minimizing


$$\min _{ \bb{\alpha}, \bb{\beta} } \sum _{i} D( \bb{x} _i , \Psi _{\bb{\alpha}'}( \Phi _{\bb{\alpha}}( \bb{x} _i ))  ),$$


where $\bb{\alpha}$ and $\bb{\beta}$ are the parameters of the encoder
and the decoder, respectively, and $D$ is some useful distance on
$\mathcal{X}$, e.g., $D(\bb{x},\bb{x}') = \|  \bb{x} - \bb{x}' \| _2^2$.
As we will see in the sequel, the choice of this distance has a crucial
importance.

The minimization of a loss of the form
$\| \bb{x} - \Psi _{\bb{\alpha}'}( \Phi _{\bb{\alpha}}( \bb{x} ))  \|^2$
might seem as a regression problem; in regression as well, we seek some
(often parsimonious) parameteric model of the map between the instance
space and the label space. Here, however, the label space is equal to
the input space, and we restrict the regressor to contain a bottleneck.
By minimizing the above loss, the encoder tries to extract only those
features of the data that allow its most faithful reconstruction, thus
creating a low-dimensional model thereof.

Convolutional autoencoder
=========================

CNN architecture are often used in autoencoders. Typically, an encoder
is a CNN that uses strided convolutions (or pooling) to gradually reduce
the signal dimension toward the bottleneck. One or more fully connected
layers may also be used before the output layer, which produces the
encoding in the latent space. A decoder can be thought of as a
*transposed* version of the encoder, in which the dimensionality
gradually increases toward the output. Though the decoder does not
necessarily need to match the same dimensions (in reversed order) of the
encoder's intermediate layers, such symmetric architectures are very
frequent. In what follows, we remind the working of a convolutional
layer and describe how to formally transpose it.

## Convolutional layer

Recall that a convolutional layer accepts an $m$-dimensional
*vector-valued* (infinitely supported) signal
$\bb{x} = (\bb{x}^1,\dots, \bb{x}^m) = \{ (x _i^1,\dots, x _i^m) \} _{i \in \mathbb{Z}}$,
each input dimension of which is called a *channel* or *feature map*.
The layer produces an $n$-dimensional (infinitely supported) signal
$\bb{y} = (\bb{y}^1,\dots, \bb{y}^n)  = \{ (y _i^1,\dots, y _i^n) \} _{i \in \mathbb{Z}}$
by applying a bank of filters, potentially retaining only every $d$-th
output sample (a fact known as *striding*),


$$\bb{y}^j = \downarrow _{d} \left( \sum _{i=1}^m \bb{w}^{ij} \ast \bb{x}^{i}   \right) ,$$


where $\downarrow _d$ denotes striding (a.k.a. compression or
down-sampling), $(\downarrow _d \bb{x} ) _{k} = \bb{x} _{dk}$. Obviously, a
bias and an element-wise non-linear activation are applied to $\bb{y}$.

Explicitly, the action of the convolutional layer can be written as


$$y^j _k =    \sum _{i=1}^m \sum _{p} w^{ij} _p x^i _{dk-p}.$$

Typically,
each filter $w^{ij}$ is supported on some small fixed domain. Though we
think of the signal $\bb{x}$ as of a function of a one-dimensional
"time" index $n \in \mathbb{Z}$, the same notation is perfectly valid
for images; in the latter case, time indices become higher-dimensional
multi-indices such as $\bb{n} \in \mathbb{Z}^2$.

In the case of finitely-supported time signals that can be represented
as vectors $\bb{x}^1\dots,\bb{x}^m \in \mathbb{R}^M$, the action of a
convolutional layer with $d=1$ can be described as


$$\bb{y}^j =   \bb{W}^{1j} \bb{x}^1 + \cdots + \bb{W}^{mj} \bb{x}^m,$$


where each $\bb{W}^{ij}$ is an $M \times M$ Toeplitz (diagonal-constant)
matrix (a block-Toeplitz matrix in the case of images). $d$-strided
convolution has the same form, but now $\bb{W}^{ij}$ are
$\frac{M}{d} \times M$ obtained by retaining every $d$-th row from the
original Toeplitz matrix. Denoting the striding (sub-sampling) by the
matrix $\bb{S} _d$, we obtain

$$\begin{aligned}
\bb{y}^1 &=&   \bb{S} _d \bb{W}^{11} \bb{x}^1 + \cdots + \bb{S} _d \bb{W}^{m1} \bb{x}^m \nonumber\\
\vdots & & \vdots  \nonumber\\
\bb{y}^n &=&   \bb{S} _d \bb{W}^{1n} \bb{x}^1 + \cdots + \bb{S} _d \bb{W}^{mn} \bb{x}^m.\end{aligned}$$

## Transposed convolutional layer

A transposed convolutional layer (often incorrectly referred to as
"deconvolutional" in the literature) can be thought of as a formal
adjoint operator[^1] of the above linear operator.

We will denote the input by the $m$-dimensional vector-valued signal
$\bb{y} = (\bb{y}^1,\dots, \bb{y}^n) = \{ (y _i^1,\dots, y _i^n) \} _{i \in \mathbb{Z}}$,
and the output by the $n$-dimensional signal
$\bb{x} = (\bb{x}^1,\dots, \bb{x}^m)  = \{ (x _i^1,\dots, x _i^m) \} _{i \in \mathbb{Z}}$.
The linear part of the layer's action is expressed by the formal
tranposition of the action of the convolutional layer,

$$\begin{aligned}
\bb{x}^1 &=&   \overline{\bb{W}}^{11}  \bb{S} _d^\Tr \bb{y}^1 + \cdots +  \overline{\bb{W}}^{1n}  \bb{S} _d^\Tr \bb{y}^n \nonumber\\
\vdots & & \vdots  \nonumber\\
\bb{x}^m &=&    \overline{\bb{W}}^{m1}  \bb{S} _d^\Tr \bb{y}^1 + \cdots +   \overline{\bb{W}}^{mn}  \bb{S} _d^\Tr \bb{y}^n,\end{aligned}$$

where $\overline{\bb{W}}^{ij} = (\bb{W}^{ij})^\Tr$ is the Toeplitz
matrix formed by the mirrored filter
$\overline{w}^{ij} _k = w^{ij} _{-k}$, and $\bb{S} _d^\Tr$ is a
$d$-upsampling (a.k.a. expansion or dilation) operation,

$$(\uparrow _d y) _k = \left\{ \begin{array}{ll} 
y _{k/d} & \mathrm{if}\, k \in d \bb{Z}; \\
0 & \mathrm{else}.
\end{array} \right.$$

Note that since the order of $\bb{S} _d$ and
$\bb{W}^{ij}$ is reversed after transposition, the input signal $\bb{y}$
is first up-sampled and then convolved with the mirrored filters. The
(linear) action of the transposed convolutional layer can be therefore
summarized as


$$\bb{x}^i =  \sum _{j=1}^n \overline{\bb{w}}^{ij} \ast  \left( \uparrow _{d}  \bb{y}^{j}  \right)   .$$


Note that despite superficial similarity, this is *not* dilated
convolution! While transposed strided convolution is
$\overline{\bb{w}} \ast (\uparrow _d \bb{y} )$, dilated convolution is
$(\uparrow _d \bb{w}  ) \ast \bb{y}$.

It is important to note that while in terms of dimensionality the
transposed convolutional layer is an inverse of the convolutional layer,
it is the adjoint and not the inverse of the latter (this is why the
term "deconvolution" is inappropriate here). Since convolutional and
transposed convolutional layers of a convolutional autoencoder are
parametrized independently, the weights of the transposed layer in the
decoder need not match those of its counterpart in the encoder.

## Pooling and unpooling

As an alternative to striding, CNN architectures sometimes use *pooling*
to reduce the output dimensionality and obtain invariance properties. A
$d$-pooling layer takes a sequence $\{x _i\} _{i \in \mathbb{Z}}$ as the
input and produces a new sequence $\{y _i\} _{i \in \mathbb{Z}}$ as the
output replacing non-overlapping windows of size $d$ in its input,
$(x _{di},\dots,x _{(d+1)i-1})$, with a scalar $y _i$. For example,
*average pooling* produces


$$(x _{di},\dots,x _{(d+1)i-1}) \mapsto y _i = \frac{1}{d} \sum _{j=0}^{d-1} x _{di + j}$$


in which each output sample is the average of the samples in the
corresponding input window[^2]. Similarly, *max pooling* is the
window-wise maximum of the input,


$$(x _{di},\dots,x _{(d+1)i-1}) \mapsto  y _i = \max \{ x _{di}, x _{di+1}, \dots, x _{(d+1)i -1} \}.$$


In order to perform backpropagation, max pooling keeps the index of the
selected maximum,


$$k^\ast _i =  \mathrm{arg}\max _{j \in \{ 0,\dots,d-1 \} } x _{di+j}.$$



When the input is vector-valued (i.e., has multiple channels), pooling
is applied channel-wise. When the input is multidimensional (e..g, an
image), the pooling window is created as an appropriate Cartesian
product.

The transposed version of the pooling layer (referred to as *unpooling*
in the literature) can be used in the decoder network. A $d$-unpooling
layer takes an input sequence $\{y _i\} _{i \in \mathbb{Z}}$ and produces
the output sequence $\{x _i\} _{i \in \mathbb{Z}}$, such that each sample
$y _{i}$ in the input corresponds to a non-overlapping window
$(x _{di},\dots,x _{(d+1)i-1})$ in the output. *Average unpooling* can be
seen as the map


$$y _i \mapsto \left( \frac{y _i}{d}, \dots, \frac{y _i}{d} \right).$$

*Max
unpooling*, on the other hand, is the map


$$y _i \mapsto \bb{e} _{k^\ast _i},$$

where $\bb{e} _k$ is the $k$-th
standard basis vector of $\mathbb{R}^d$ containing $1$ at $k$-th
coordinate and zeros elsewhere, and $k _i^\ast$ is the index kept by the
corresponding max pooling counterpart.

Variational autoencoders
========================

While regular autoencoders can produce very intricate data models, it is
still unclear how to generate new instances from the model. The
distribution of the latent variable $\bb{z}$ produced by the encoder can
be very complicated, and sampling from it may result impractical. A very
popular variant of autoencoders combines this type of neural networks
with variational Bayesian methods. In what follows, we will revisit our
problem from this perspective and see how this reformulation leads to
the *variational autoencoder* (VAE).

## Decoder

Let us start from looking at the generative (decoder) part of the model
from a probabilistic perspective. The latent variable $\bb{Z}$ is a
random vector distributed according to some *prior* distribution
$p(\bb{Z})$. The instance $\bb{X}$ is a random vector over $\mathcal{X}$
and is described by the probability distribution $p(\bb{X} | \bb{Z})$
conditioned on $\bb{Z}$. In the Bayesian jargon, this conditional
probability is known as the *likelihood*. Note that in this formulation,
given a specific realization of the latent variable, $\bb{Z}=\bb{z}$,
the output of the encoder is still a stochastic quantity distributed
according to $p(\bb{X} | \bb{z})$. It is customary to set
$p _{\bb{\beta}}(\bb{X} | \bb{z}) = \mathcal{N}( \Psi _{\bb{\beta}}(\bb{z}) , \sigma^2 \bb{I} )$,
where $\Psi _{\bb{\beta}}$ is a deterministic map between the latent
space and the instance space (in practice, our encoder network), and
$\sigma$ is a fixed hyper-parameter. This allows
$p _{\bb{\beta}}(\bb{X} | \bb{Z})$ to be computable and continuous in its
parameters $\bb{\beta}$.

Since any distribution can be mapped into any other distribution by a
sufficiently complicated map, we can fix the prior distribution to be
normal, $p(\bb{Z}) = \mathcal{N}(\bb{0},\bb{I})$. We can now generate
new instances by drawing $\bb{z}$ at random from the multivariate normal
distribution, calculating the mean vector $\Psi _{\bb{\beta}}(\bb{z})$,
and then drawing an $\bb{x}$ instance from
$\mathcal{N}( \Psi _{\bb{\beta}}(\bb{z}) , \sigma^2 \bb{I} )$.

## Encoder

The inference (encoder) part of the model aims at inferring the latent
variable given the observed instance, which in the Bayesian language
amounts to calculating the *posterior*


$$p(\bb{Z}|\bb{X}) = \frac{p(\bb{X}|\bb{Z}) p(\bb{Z}) }{p(\bb{X})}.$$


The denominator expresses the probability distribution on the instance
space, known as the *evidence* in the Bayesian terminology. While it has
a simple expression due to the total probability formula,


$$p(\bb{X}) = \int p(\bb{X}|\bb{z}) p(\bb{z}) d\bb{z},$$

However, the
latter integral is intractable in practice, since its accurate
approximation in a relatively high-dimensional latent space mapping to a
complicated instance distributiuon would require an enormously large
sample.

The main idea of VAEs is to approximate the posterior with a parametric
family of distributions, $q(\bb{Z} | \bb{X})$. A typical choice is,
again, normal
$q(\bb{Z} | \bb{x}) \sim \mathcal{N}( \bb{\mu} _{\bb{\alpha}}(\bb{x}),  \bb{\Sigma} _{\bb{\alpha}}(\bb{x}) )$,
where the mean and covariance parameters of the distribution are
produced by the encoder neural network parametrized by $\bb{\alpha}$. In
practice, $\bb{\Sigma}$ is constrained to be a diagonal matrix, so
$q _{\bb{\alpha}}(\bb{Z} | \bb{x}) \sim \mathcal{N}( \bb{\mu} _{\bb{\alpha}}(\bb{x}),  \mathrm{diag}\{ \bb{\sigma} _{\bb{\alpha}}(\bb{x}) \} )$.

## Evidence lower bound

In order to ensure that the variational posterior
$q _{\bb{\alpha}}(\bb{Z} | \bb{X})$ approximates well the true posterior
$p(\bb{Z} | \bb{X})$, we can measure the Kullback-Leibler divergence[^3]

$$\begin{aligned}
\mathcal{D} _{\mathrm{KL}}\left(q _{\bb{\alpha}}(\bb{Z} | \bb{X})\,\left\|\, p(\bb{Z} | \bb{X})\right.\right) &=& \mathbb{E} _{\bb{z} \sim q _{\bb{\alpha}} }( \log q _{\bb{\alpha}}(\bb{z} | \bb{X}) - \log  p(\bb{z} | \bb{X}) ) \\
&=&  \mathbb{E} _{\bb{z} \sim q _{\bb{\alpha}} }( \log q _{\bb{\alpha}}(\bb{z} | \bb{X}) - \log  p(\bb{X} | \bb{z})    - \log  p(\bb{z})  ) + \log p(\bb{X});\end{aligned}$$

note that $\log p(\bb{X})$ is outside the expectation since it does not
depend on $\bb{z}$. This expression can be rewritten as

$$\log p(\bb{X}) - \mathcal{D} _{\mathrm{KL}}\left(q _{\bb{\alpha}}(\bb{Z} | \bb{X})\,\left\|\, p(\bb{Z} | \bb{X})\right.\right) =  \mathbb{E} _{\bb{z} \sim q _{\bb{\alpha}} }(  \log  p(\bb{X} | \bb{z}) )
-  \mathcal{D} _{\mathrm{KL}}\left(q _{\bb{\alpha}}(\bb{Z} | \bb{X})\,\left\|\, p(\bb{Z} )\right.\right)$$

Note that while we still cannot evaluate the left-hand-side second term
(because of the $p(\bb{X})$ appearing in the KL divergence), we observe
that
$\mathcal{D} _{\mathrm{KL}}\left(q _{\bb{\alpha}}(\bb{Z} | \bb{X})\,\left\|\, p(\bb{Z} | \bb{X})\right.\right)$
is non-negative and small if the model $q _{\bb{\alpha}}$ is rich enough.
We can therefore express the following lower bound on the log evidence

$$\log p(\bb{X}) \ge \mathbb{E} _{\bb{z} \sim q _{\bb{\alpha}} }( \log  p _{\bb{\beta}}(\bb{X} | \bb{z}) )
-  \mathcal{D} _{\mathrm{KL}}\left(q _{\bb{\alpha}}(\bb{Z} | \bb{X})\,\left\|\, p(\bb{Z} )\right.\right).$$


## Loss function

Recall that $P(\bb{x})$ expresses the probability of a given instance
$\bb{x}$ under the entire generative process. Since we aim at maximizing
the probability of each instance, our goal is to minimize the following
loss function:

$$L =-\mathbb{E} _{\bb{x}} \log P(\bb{x}) \le \mathbb{E} _{\bb{x}}  \left( 
\mathbb{E} _{\bb{z} \sim q _{\bb{\alpha}} }( -\log  p _{\bb{\beta}}(\bb{x} | \bb{z}) )
+  \mathcal{D} _{\mathrm{KL}}\left(q _{\bb{\alpha}}(\bb{Z} | \bb{x})\,\left\|\, p(\bb{Z} )\right.\right)
\right).$$

With some abuse of notation, we will refer to the latter
upper bound as our target loss function. Taking the gradient w.r.t. the
parameters $\bb{\alpha},\bb{\beta}$ allows to move the gradient operator
under the expectation

$$\nabla _{\bb{\alpha},\bb{\beta}} L =  \mathbb{E} _{\bb{x}}  \left( 
\mathbb{E} _{\bb{z} \sim q _{\bb{\alpha}} }( \nabla _{\bb{\alpha},\bb{\beta}} (-\log  p _{\bb{\beta}}(\bb{x} | \bb{z}) ))
+ \nabla _{\bb{\alpha},\bb{\beta}}  \mathcal{D} _{\mathrm{KL}}\left(q _{\bb{\alpha}}(\bb{Z} | \bb{x})\,\left\|\, p(\bb{Z} )\right.\right)
\right).$$

Using stochastic gradient, we can sample a single value
$\bb{x}$ of $\bb{X}$ from the training set, sample a single value
$\bb{z}$ of $\bb{Z}$ from the distribution
$q _{\bb{\alpha}}(\bb{Z} | \bb{x})$, and compute the gradient of a point
loss term

$$\begin{aligned}
\ell &=& -\log  p _{\bb{\beta}}(\bb{x} | \bb{z}) + \mathcal{D} _{\mathrm{KL}}\left(q _{\bb{\alpha}}(\bb{z} | \bb{x})\,\left\|\, p(\bb{z} )\right.\right) \\
&=& \frac{1}{2\sigma^2}\| \bb{x}- \Psi _{\bb{\beta}}(\bb{z}) \| _2^2 + \mathcal{D} _{\mathrm{KL}}\left(q _{\bb{\alpha}}(\bb{z} | \bb{x})\,\left\|\, p(\bb{z} )\right.\right).\end{aligned}$$

First, we observe that the second term of the loss has a simple
closed-form expression of the KL divergence between Gaussian
distributions,

$$\begin{aligned}
\mathcal{D} _{\mathrm{KL}}\left(q _{\bb{\alpha}}(\bb{z} | \bb{x})\,\left\|\, p(\bb{z} )\right.\right) &=&  \mathcal{D} _{\mathrm{KL}}\left(\mathcal{N}( \bb{\mu} _{\bb{\alpha}}(\bb{x}),  \bb{\Sigma} _{\bb{\alpha}}(\bb{x}) )\,\left\|\, \mathcal{N}(\bb{0},\bb{I})\right.\right) \\
&=& \frac{1}{2} \left( \mathrm{tr}\,\bb{\Sigma} _{\bb{\alpha}}(\bb{x}) +  \|\bb{\mu} _{\bb{\alpha}}(\bb{x})\|^2 _2 - k - \log\det \bb{\Sigma} _{\bb{\alpha}}(\bb{x})  \right),\end{aligned}$$


where $k$ denotes the dimension of the latent space. This term can be
viewed as a regularization added to the loss.

The first term is trickier. Note that in the original loss function it
depended both on $\bb{\beta}$ (parametrizing $p(\bb{x} | \bb{z})$) as
well as $\bb{\alpha}$ (since $\bb{z}$ was drawn from $q _{\bb{\alpha}}$);
the dependence on $\bb{\alpha}$ somehow evaporated when computing the
gradient. The network works fine as long as the output is averaged over
many samples, producing a correct expected value; however, when
computing the gradient, we need to backpropagate through a layer that
samples $\bb{z}$ from $q _{\bb{\alpha}}$ which is not even continuous and
thus has no gradient. While being able to handle stochastic *inputs*,
stochastic gradient cannot handle stochastic *operations* within the
network. To overcome this problem, the sampling is moved to the input
layer. Given the mean and the covariance,
$\bb{\mu} _{\bb{\alpha}}(\bb{x}),  \bb{\Sigma} _{\bb{\alpha}}(\bb{x})$ of
$q _{\bb{\alpha}}$, we can sample from
$\mathcal{N}( \bb{\mu} _{\bb{\alpha}}(\bb{x}),  \bb{\Sigma} _{\bb{\alpha}}(\bb{x}) )$
by first sampling $\bb{u}$ from $\mathcal{N}(\bb{0},\bb{I})$
(independent of $\bb{\alpha}$) and then computing the deterministic
transformation
$\bb{z} = \bb{\mu} _{\bb{\alpha}}(\bb{x}) + \bb{\Sigma}^{\frac{1}{2}} _{\bb{\alpha}}(\bb{x}) \bb{u}$.
After this reparametrization trick, the loss term becomes the following
tractable expression:


$$\mathbb{E} _{\bb{u} \sim \mathcal{N}(\bb{0},\bb{I})} \, \frac{1}{2\sigma^2}\left\| \bb{x}- \Psi _{\bb{\beta}}\left(  \bb{\mu} _{\bb{\alpha}}(\bb{x})  + \bb{\Sigma}^{\frac{1}{2}} _{\bb{\alpha}}(\bb{x}) \bb{u}   \right) \right\| _2^2.$$



To summarize, the VAE is trained by minimizing the following point loss
function using stochastic gradient descent:


$$\ell = \frac{1}{\sigma^2}\left\| \bb{x}- \Psi _{\bb{\beta}}\left(  \bb{\mu} _{\bb{\alpha}}(\bb{x})  + \bb{\Sigma}^{\frac{1}{2}} _{\bb{\alpha}}(\bb{x}) \bb{u}   \right) \right\| _2^2 +  \mathrm{tr}\,\bb{\Sigma} _{\bb{\alpha}}(\bb{x}) +  \|\bb{\mu} _{\bb{\alpha}}(\bb{x})\|^2 _2 - k - \log\det \bb{\Sigma} _{\bb{\alpha}}(\bb{x})$$


with $\bb{u}$ drawn from the normal distribution and $\bb{x}$ is drawn
from the training data. The first term can be thought of as a data
fitting term like in regression, demanding that the encoder-decoder
combination is nearly an identity map. The second term applies
regularization on the output of the encoder in the latent space. The
hyper-parameter $\sigma^2$ governs the relative strength of the two
terms. The smaller is $\sigma^2$, the less randomness is allowed in the
decoder mapping from $\mathcal{Z}$ to $\mathcal{X}$, and, consequently,
the regression term dominates over the regularization term.

Generative adversarial networks
===============================

One of the consequences of using the $\ell _2$ loss in the training of an
autoencoder is the so-called *regression to the mean* problem, which
explains why generative models trained with this loss tend to produce
blurred results. An powerful alternative consists of allowing the loss
function to train together with the generative model in an *adversarial*
manner. Let $\Psi _{\bb{\beta}} : \mathcal{Z} \rightarrow \mathcal{X}$ be
a generative model (decoder network) taking an input
$\bb{z} \in p(\bb{Z})$ and mapping it to the space of instances. As the
result, the generated instances admit a distribution
$p _{\bb{\beta}}(\bb{X})$, which might be different from the true data
distribution $p(\bb{X})$.

We also define another network
$\Delta _{\bb{\theta}} : \mathcal{X} \rightarrow [0,1]$ taking an
instance and returning the probability that it is coming from the data
rather than from $p _{\bb{\beta}}(\bb{X})$. The *discriminator*
$\Delta _{\bb{\theta}}$ is trained to maximize the correct label assigned
to both the real data coming from $p(\bb{X})$ and generated data coming
from $p _{\bb{\beta}}(\bb{X})$, i.e., it should distinguish as clearly as
possible between instances coming from the real data
($\Delta _{\bb{\theta}} \approx 1$) and the "fake" generated distribution
(ideally, $\Delta _{\bb{\theta}} \approx 0$). The *generator*
$\Psi _{\bb{\beta}}$ is simultaneously trained to minimize
$\log (1-\Delta _{\bb{\theta}}(\Psi _{\bb{\beta}} (\bb{z}) ))$, that is,
to "fool" the discriminator and cause it to misclassify as big a
fraction of generated instances as possible. The training can be
expressed as the following two-player min-max game:

$$\min _{\bb{\beta}} \max _{\bb{\theta}} \, \mathbb{E} _{\bb{x} \sim p(\bb{X}) } \log \Delta _{\bb{\theta}}(\bb{x})  \, + \,
  \mathbb{E} _{\bb{z} \sim p(\bb{Z}) } \log (1-\Delta _{\bb{\theta}}(\Psi _{\bb{\beta}} (\bb{z}) )).$$

The idea of adversarial training can be applied to training AEs and VAEs
as well.

We can interpret the maximum,

$$L({\bb{\beta}}) = \max _{\bb{\theta}} \, \mathbb{E} _{\bb{x} \sim p(\bb{X}) } \log \Delta _{\bb{\theta}}(\bb{x})  \, + \,
  \mathbb{E} _{\bb{z} \sim p(\bb{Z}) } \log (1-\Delta _{\bb{\theta}}(\Psi _{\bb{\beta}} (\bb{z}) )),$$

as the loss function minimized during the training of the generator.
However, note that for every choice of its parameters, we will be
minimizing a different loss, since $\bb{\theta}$ will change as well.


[^1]: Let $\mathcal{X}$ and $\mathcal{Y}$ be some spaces equipped with
    appropriate inner products. Let
    $\mathcal{A} : \mathcal{X} \rightarrow \mathcal{Y}$ and
    $\mathcal{B} : \mathcal{Y} \rightarrow \mathcal{X}$ two operators.
    The operator $\mathcal{B}$ is called the *adjoint* of $\mathcal{A}$,
    denoted as $\mathcal{B} = \mathcal{A}^\ast$, if for every
    $\bb{x} \in \mathcal{X}$ and $\bb{y} \in \mathcal{Y}$,
    $\langle \mathcal{A} \bb{x} , \bb{y} \rangle _{\mathcal{Y}}  = \langle \bb{x}, \mathcal{B} \bb{y}  \rangle _{\mathcal{X}}$.
    Though $\mathcal{A}^\ast$ is a map from the co-domain of
    $\mathcal{A}$ to its domain, it is not an inverse of $\mathcal{A}$,
    which may not even be invertible! The adjoint matches the inverse
    only in the case of unitary operators (a generalized notion of
    rotation).

[^2]: In this notation, we assumed the window to be causal, which is a
    typical choice for time signals. For images, the window is typically
    symmetric about $i$.

[^3]: The KL divergence is an (asymmetric) distance between
    distributions defined as
    $$\mathcal{D} _{\mathrm{KL}}\left((\,\left\|\, Q\right.\right)||P) = \mathbb{E} _{z \sim Q}( \log Q(z) - \log P(z)).$$


