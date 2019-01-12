---
title: "Lecture 11: Learning on Non-Euclidean Domains"
excerpt: "Toeplitz operators, manifolds, graphs, convolution, spectral CNN"
copyright: alex
---

All learning settings we have encountered thus far had a common
(sometimes, tacit) property: they assumed Euclidean geometry of the
data. For example, we could compute standard inner products, subtract
one vector from another, apply matrices to vectors, etc. Data like time
signal and images were further discretized on regular Cartesian grids,
and we could apply operations like convolution by simply sliding the
same window over the signal and computing inner products.

Most of these apparently straightforward notions become less
straightforward when the domain underlying the data is no longer
Euclidean. Such kinds of data arise in a long list of applications. For
example, in social networks, user information can be modeled as signals
on a graph. Sensor networks are also modeled as graphs of distributed
interconnected sensors, whose readings are time-dependent signals on the
graph. In neuroscience, graph models are used to represent anatomical
and functional structures of the brain. In biology, graphs are a common
way to express interactions between genes, proteins, etc. In computer
graphics and vision, three-dimensional geometric objects are often
represented as Riemannian manifolds (surfaces) endowed with attributes
such as color texture.

It is important to distinguish between two very distinct tasks: learning
*on* non-Euclidean domains vs. learning *of* non-Euclidean domains. We
encountered the latter problem when discussing unsupervised learning,
where our goal was to discover (learn) the latent manifold from which
the data are sampled. The former problem of analyzing signals on graphs
and manifolds is what is going to occupy us in this lecture. In what
follows, we wil briefly review the main properties of the basic
ingredients of CNNs on Euclidean domain, specifically, the convolution
operator and pooling. We will then describe how to generalize these
notions to graphs.

Convolution on Euclidean domains
================================

The main ingredient of a CNN is a convolutional layer, describing a
mapping between $m$-dimensional input signals to $n$-dimensional output
signals (we use the term *signal* to generalize the notion of a
sequence, allowing its elements to be indexed by a $d$-dimensional
multi-index; this notion includes images and higher-dimensional signals
besides time series). Formally, a convolutional layer accepts an
$m$-dimensional *vector-valued* (infinitely supported) signal
$\bb{x} = (\bb{x}^1,\dots, \bb{x}^m) = \\{ (x _{\bb{k}}^1,\dots, x _{\bb{k}}^m) \\} _{ {\bb{k}} \in \mathbb{Z}^d}$,
each input dimension of which is called a *channel* or *feature map*.
The layer produces an $n$-dimensional (infinitely supported) signal
$\bb{y} = (\bb{y}^1,\dots, \bb{y}^n)  = \\{ (y _{\bb{k}}^1,\dots, y _{\bb{k}}^n) \\} _{ {\bb{k}} \in \mathbb{Z}^d}$
by applying a bank of filters,


$$\bb{y}^j = \varphi \left( \sum _{i=1}^m \bb{w}^{ij} \ast \bb{x}^{i} + b _j   \right) ,$$


Explicitly, the action of the convolution
$\bb{z}^j = \bb{w}^{ij} \ast \bb{x}^i$ can be written as


$$z^j _{\bb{k}} =    \sum _{i=1}^m \sum _{\bb{p} \in \mathbb{Z}^d } w^{ij} _{\bb{p}} x^i _{\bb{k}-\bb{p}}.$$


Note the $d$-dimensional multi-indices in the sum.

Eigenvectors of Toeplitz operators
----------------------------------

Since the convolution operation is the main ingredient of a CNN, let us
dedicate some attention to listing a few of its properties that will be
instrumental in the generalization of CNNs to non-Euclidean domains. As
we have already seen, any linear shift-invariant[^1] (Toeplitz) operator
$\mathcal{W}$ can be represented as the convolution


$$\mathcal{W}\bb{x} = \bb{x} \ast \bb{w}.$$

The action of any linear
operator on a vector consists of scaling and rotating the vector.
However, there are some privileged directions where no rotation occurs;
such directions are called the eigenvectors of the operator.
Specifically, for Toeplitz operators, given the input signal


$$\bb{\phi}^{\bb{\xi}} _{\bb{n}} = e^{i\, 2\pi \bb{\xi} ^\Tr \bb{n}}$$


parametrized by the vector $\bb{\xi} \in [0,1]^d$, the output of the
operator is

$$(\bb{w} \ast \bb{\phi}^{\xi}) _{\bb{n}} = \sum _{ {\bb{k}}}  w _{\bb{n}} {\phi}^{\bb{\xi}} _{ {\bb{n}}-{\bb{k}}} = 
 \sum _{ {\bb{k}}}  w _{\bb{k}} e^{i\, 2\pi \bb{\xi} ^\Tr (\bb{n} - \bb{k})} =  e^{i\, 2\pi \bb{\xi} ^\Tr \bb{n} } \sum _{ {\bb{k}}}  w _{\bb{k}} e^{-i\, 2\pi \bb{\xi} ^\Tr \bb{k}}.$$

Recalling the standard inner product on $\ell^2$,


$$\langle  \bb{x}, \bb{y} \rangle = \sum _{\bb{k}} x _{\bb{k}} y^\ast _{\bb{k}},$$


we can express


$$\sum _{ {\bb{k}}}  w _{\bb{k}} e^{-i\, 2\pi \bb{\xi} ^\Tr  \bb{k} } =  \sum _{ {\bb{k}}}  w _{\bb{k}}  {\phi}^{\bb{\xi}} _{-k}  =  \sum _{ {\bb{k}}}  w _{\bb{k}}  \left({\phi}^{ {\bb{\xi}}} _{\bb{k}} \right)^\ast = \langle  \bb{w}, \bb{\phi}^{\bb{ {\bb{\xi}}}} \rangle = \hat{\bb{w}}({\bb{\xi}}).$$


In these terms, the output is given by


$$\bb{w} \ast \bb{\phi}^{\bb{\xi}} = \hat{\bb{w}}(\xi) \bb{\phi}^{\bb{\xi}},$$


which means that $\bb{\phi}^{\bb{\xi}}$ is an eigenvector of
$\mathcal{W}$ with the corresponding eigenvalues
$\hat{\bb{w}}(\bb{\xi})$. Note that while the *eigenvalues* depend on
the specific operator (embodied in the sequence $\bb{w}$ called the
*kernel* of the operator), the *eigenvectors* are always the same:
$\\{\bb{\phi}^{\bb{\xi}}\\} _{\bb{\xi} \in [0,1]^d}$.

Fourier transform
-----------------

The function


$$\hat{\bb{w}}(\bb{\xi}) = \langle  \bb{w}, \bb{\phi}^{\xi} \rangle _{\ell^2(\mathbb{Z}^d)} = \sum _{\bb{k}}  w _{\bb{k}} e^{-i\, 2\pi \bb{\xi} ^\Tr \bb{k} }$$


is called the *(forward) Fourier transform* of the sequence $\bb{w}$. It
is customary to define the operator
$\mathcal{F} : \ell^2(\mathbb{Z}^d) \rightarrow L^2([0,1]^d)$ mapping
$\bb{w}$ to $\hat{\bb{w}}$, and refer to the argument $\bb{\xi}$ of the
latter as to *frequency*. The inverse map
$\mathcal{F}^{-1} : L^2([0,1]^d) \rightarrow \ell^2(\mathbb{Z}^d)$,
called the *inverse Fourier transform*, is given by

$$\bb{w} = \mathcal{F}^{-1} \hat{\bb{w} } = \int _{[0,1]^d} \hat{\bb{w}}(\bb{\xi})  \bb{\phi}^{\bb{\xi}} d\bb{\xi} =
 \int _{[0,1]^d} \hat{\bb{w}}(\bb{\xi})  e^{i\, 2\pi \bb{\xi} ^\Tr \bb{n}} d\bb{\xi}.$$

 To prove this, observe that
 
$$\begin{aligned}
 \int _{[0,1]^d} \hat{\bb{w}}(\bb{\xi})  e^{i\, 2\pi \bb{\xi} ^\Tr \bb{n}} d\bb{\xi} &=&  
  \int _{[0,1]^d}   \left( \sum _{\bb{k}}  w _{\bb{k}} e^{-i\, 2\pi \bb{\xi} ^\Tr \bb{k}}  \right)  e^{i\, 2\pi \bb{\xi} ^\Tr\bb{n}}  d\bb{\xi} \\
 &=& \sum _{\bb{k}}  w _{\bb{k}} \left(   \int _{[0,1]^d}  e^{i\, 2\pi \bb{\xi} ^\Tr (\bb{n}-\bb{k})} d\bb{\xi} \right).\end{aligned}$$

Since the complex exponentials
$e^{i\, 2\pi \bb{\xi} ^\Tr (\bb{n}- \bb{k})}$ have an integer number of
periods on the domain $[0,1]^d$, the latter integral is zero unless
$\bb{n}-\bb{k} = \bb{0}$, in which case it is exactly $1$. Hence,


$$\int _{[0,1]^d} \hat{\bb{w}}(\bb{\xi})  e^{i\, 2\pi \bb{\xi} ^\Tr \bb{n}} d\bb{\xi} = \sum _{\bb{k}}  w _{\bb{k}} \delta _{\bb{n}-\bb{k}} =  w _{\bb{n}}.$$



Note that the inverse Fourier transform can also be written as


$$(\mathcal{F}^{-1} \hat{\bb{w}} ) _{\bb{n}}  = \langle  \hat{\bb{w}},  (\bb{\phi}^{\bb{n}})^\ast  \rangle _{L^2(\mathbb{Z}^d)},$$


which emphasizes the above orthonormality property by essentially
stating that $\mathcal{F}^{-1} = \mathcal{F}^\ast$, where
$\mathcal{F}^\ast$ denotes the adjoint operator. Geometrically, this
means that the Fourier transform is *unitary*, a generalized form of
rotation. The fact that rotations preserve distances leads to the
celebrate *Plancherel identity*


$$\langle  \bb{f}, \bb{g} \rangle _{\ell^2(\mathbb{Z}^d)} = \langle  \hat{\bb{f}}, \hat{\bb{g}} \rangle _{L^2([0,1]^d)},$$


and, in the particular case of $\bb{g} = \bb{f}$, even more celebrate
*Parseval's identity*


$$\|  \bb{f} \| _{\ell^2(\mathbb{Z}^d)} = \| \hat{\bb{f}} \| _{L^2([0,1]^d)}.$$



We can think of the Fourier transform as the transformation of a signal
$\bb{x}$ to the (joint) eigenbasis of (all) Toeplitz operators, that is,
describing $\bb{x}$ as a linear combination of the eigenvectors
$\bb{\phi}^{\bb{\xi}}$,


$$\bb{x} = \int _{-\pi}^\pi \hat{\bb{x}}(\bb{\xi})  \bb{\phi}^{\bb{\xi}} d\bb{\xi},$$


with $\hat{\bb{x}}(\bb{\xi})$ serving as the coordinates in the
eigenbasis. In the signal processing parlance, $\bb{x}$ in the standard
basis is called the *impulse response*, while in the Fourier basis it is
referred to as the *frequency response*. The act of converting the
impules response to the frequency response is referred to as *analysis*,
while the inverse is referred to as *synthesis*.

The notion of the Fourier transform allows to apply the operator
$\mathcal{W}$ in its eigenbasis. Indeed, describing the input sequence
as a linear combination of the eigenvectors,


$$\bb{x} = \int _{-\pi}^\pi \hat{\bb{x}}(\xi)  \bb{\phi}^{\bb{\xi}} d\bb{\xi},$$


we obtain

$$\bb{w} \ast \bb{x} = \mathcal{W}  \int _{[0,1]^d} \hat{\bb{x}}(\bb{\xi})  \bb{\phi}^{\bb{\xi}} d\bb{\xi} =  
 \int _{[0,1]^d} \hat{\bb{x}}(\bb{\xi}) \mathcal{W} \bb{\phi}^{\bb{\xi}} d\bb{\xi} =  \int _{[0,1]^d} \hat{\bb{w}}(\bb{\xi}) \hat{\bb{x}}(\bb{\xi}) \bb{\phi}^{\bb{\xi}} d\bb{\xi}.$$

The latter can be written as


$$\bb{w} \ast \bb{x} = \mathcal{F}^{1-} ( \mathcal{F}\bb{w}  \cdot \mathcal{F} \bb{x} )$$


or, schematically,


$$\bb{w} \ast \bb{x}  \mathop{\longleftrightarrow}^{\mathcal{F}} \hat{\bb{w}} \cdot \hat{\bb{x}}.$$


This result is known as the *convolution theorem*, stating that
convolution becomes pointwise product in the Fourier (frequency) domain.
This result is the consequence of the fact that the Fourier transform
diagonalizes Toeplitz operators (convolution). In fact, operating in the
Fourier domain, we can *define* any Toeplitz operator as a diagonal
linear operator (point-wise product), fully defined by the function
$\hat{\bb{w}}(\bb{\xi})$.

Non-Euclidean domains
=====================

In order to generalize the notion of a CNN to the non-Euclidean case,
let us first define two types of non-Euclidean domains: manifolds and
graphs (we will think of the latter as of some sort of discretization of
the former).

Manifolds
---------

A topological space $\mathcal{M}$ is called a $d$-dimensional *manifold*
if every point $p$ in it has a neighborhood topologically equivalent
(homeomorphic) to $\RR^d$. The latter space is referred to as the
*tangent* space at point $p$, denoted as $T _p \mathcal{M}$. The disjoint
union of all tangent spaces is called the *tangent bundle*, denoted as
$T\mathcal{M}$. Since each point $p$ is now associated with a linear
space $T _p \mathcal{M}$, we can endow the latter with an inner product
$\langle \cdot, \cdot \rangle _{T _p \mathcal{M}} : T _p \mathcal{M}\times T _p \mathcal{M}\rightarrow T _p \mathcal{M}$
(which we assume to depend smoothly on $p$, without further defining
preciselt what it means). This inner product is called a *Riemannian
metric* and a manifold endowed with it is called a *Riemannian
manifold*. The metric allows to (locally) measure lengths and angles.

### Fields

A *scalar field* on $\mathcal{M}$ is a function of the form
$f : \mathcal{M}\rightarrow \RR$. A *(tangent) vector field* is a map
$F : \mathcal{M}\rightarrow T \mathcal{M}$ assigning to every point
$p \in \mathcal{M}$ a tangent vector $F(p) \in T _p \mathcal{M}$. Tangent
vectors formalize the notion of infinitesimal displacements that we
routinely use in calculus on Euclidean domains. Next, we define the
Hilbert spaces of scalar and vector fields on $\mathcal{M}$ through the
following standard inner products:

$$\begin{aligned}
\langle  f, g \rangle _{L^2(\mathcal{M}) } &=& \int _{\mathcal{M}} f(p) g(p) dp; \\
\langle  F, G \rangle _{L^2(T \mathcal{M}) } &=& \int _{\mathcal{M}}  \langle F(p) , G(p) \rangle _{T _p \mathcal{M}} dp,\end{aligned}$$

where the integration is performed w.r.t. the $d$-dimensional volume
element $dp$ induced by the metric.

### Differential

The notion of a derivative in calculus describes how the value of a
function changes with an infinitesimal change of its argument. One of
the big differences distinguishing calculus from differential geometry
is a lack of a global vector space structure on the manifold, making
expressions like $f(p+dp)$ meaningless. The conceptual leap that is
required to generalize calculus to manifolds is the need to express all
notions locally in the tangent spaces.

In order to construct calculus on a manifold, we define the
*differential* of $f$ as the operator
$df : T \mathcal{M}\rightarrow \RR$ on tangent vectors. At every
$p \in \mathcal{M}$, the differential is defined as the linear
functional (a.k.a. $1$-form in the differential geometry jargon)


$$df(p) : v \mapsto \langle \nabla f(p), v \rangle _{T _p \mathcal{M}},$$


$v \in T _p \mathcal{M}$. A vector field $F$ generalizes the notion of
small displacements. In fact, we can write


$$df(p)F(p) = \langle \nabla f(p), F(p) \rangle _{T _p \mathcal{M}},$$

as
the extension of the regular notion of directional derivative in
Euclidean spaces,


$$df = f(\bb{p}+d\bb{p}) = \langle \nabla f(\bb{p}), d\bb{p}  \rangle = \frac{\partial f(\bb{p})}{\partial p _1}dp _1 + \cdots + \frac{\partial f(\bb{p})}{\partial p _d} dp _d.$$



### Gradient and divergence

The operator $\nabla f(p) : \mathcal{M}\rightarrow T\mathcal{M}$
appearing in the definition of the differential generalizes the notion
of the *gradient* defining the direction of the steepest increase of
$f$; the main difference is that on a manifold the latter direction is
given by tangent vector. The gradient can be viewed as an operator of
the form $\nabla : L^2(\mathcal{M}) \rightarrow L^2(T\mathcal{M})$
mapping scalar fields to vector fields. Its adjoint is called the
*divergence* operator,
$\mathrm{div} f(p) : L^2(T \mathcal{M}) \rightarrow L^2(\mathcal{M})$
mapping vector fields to scalar fields and satisfying


$$\langle F, \nabla f\rangle _{L^2(T\mathcal{M})} = \langle \nabla^\ast F,  f\rangle _{L^2(\mathcal{M})} = \langle -\mathrm{div}\, F,  f\rangle _{L^2(\mathcal{M})}$$


(note the minus sign!). As vector fields can be thought of as a model of
a flow on the manifold, the divergence operator measures the net flow at
a point.

### Laplacian

The *Laplacian* (a.k.a. the *Laplace-Beltrami operator*)
$\Delta : L^2(\mathcal{M}) \rightarrow L^2(\mathcal{M})$ is defined as


$$\Delta  = \nabla^\ast \nabla = -\mathrm{div} \nabla.$$

The Laplacian
of a scalar field $f$ at point $p$ can be interpreted as the difference
between the average value of the field on an infinitesimal sphere around
$p$ and the value of $f(p)$.

By virtue of the adjoint relation between the gradient and the negative
divergence, the Laplacian is self-adjoint (symmetric), that is, for
every scalar field $f$,


$$\langle \nabla f, \nabla f \rangle _{L^2(T\mathcal{M})} = \langle \nabla^\ast \nabla f,  f \rangle _{L^2(\mathcal{M})} = \langle \Delta f,  f \rangle _{L^2(\mathcal{M})}$$


and


$$\langle \nabla f, \nabla f \rangle _{L^2(T\mathcal{M})} = \langle f,   \nabla^\ast \nabla  f \rangle _{L^2(\mathcal{M})} = \langle  f,  \Delta f \rangle _{L^2(\mathcal{M})}.$$


The expression $\langle \Delta f,  f \rangle _{L^2(\mathcal{M})}$ is
known as the *Dirichlet energy* of the field $f$ and measures the
"smoothness" of the field on $\mathcal{M}$. Physically, it can be
interpreted as the potential energy due to the bending of an elastic
body.

Graphs
------

We will limit our attention to undirected graphs and view them as a
discrete analog of manifolds. We define the *vertex* set
$V = \\{ 1,\dots, n\\}$ (it can be any set containing $n$ objects, which
we canonically map to the above set of natural numbers from $1$ to $n$);
the *edge set* and the *edge* set $E \subseteq V \times V$. An
undirected graph has $(i,j) \in E \Leftrightarrow (j,i) \in E$. We
further define the vertex weights as the function
$a : V \rightarrow (0,\infty)$ and the edge weights as
$w : E \rightarrow \RR _+$ (in fact, $w$ can be defined on the entire
$V \times V$ with $w _{ij} = 0$ meaning $(i,j) \notin E$). We refer to
the tuple $\mathcal{G}= (V,E,a,w)$ as to a *weighted undirected graph*.

### Difference operators

A *vertex* field is a function of the form $f : V \rightarrow \RR$,
while an *edge field* is a function of the form $F : E \rightarrow \RR$.
Vertex and edge fields on a graph are the discrete analogs of scalar and
vector fields on a manifold (under the tacit assumption that
$F _{ij} = -F _{ji}$ for technical reason we are not goind to detail). As
in the case of manifolds, we define the two Hilbert spaces, $\ell^2(V)$
and $\ell^2(E)$ through the corresponding inner products

$$\begin{aligned}
\langle  f, g \rangle _{\ell^2(V) } &=& \sum _{i \in V} a _i f _i g _i ; \\
\langle  F, G \rangle _{\ell^2(E) } &=& \sum _{(i,j) \in E} w _{ij} F _{ij} G _{ij};\end{aligned}$$

note that the weights play the role of discrete volume elements we had
before in the integrals on manifolds.

The graph *gradient* is the operator
$\nabla : \ell^2(V) \rightarrow \ell^2(E)$ defined by


$$(\nabla f) _{ij} = f _i - f _j.$$

Note that the resulting edge field is,
by definition, alternating, that is,
$(\nabla f) _{ij} = -(\nabla f) _{ji}$. Analogously to manifolds, the
adjoint operator, the graph *divergence*
$\mathrm{div} : \ell^2(E) \rightarrow \ell^2(V)$ is defined as


$$(\mathrm{div}\, F) _i = \frac{1}{a _i} \sum _{(i,j) \in E} w _{ij} F _{ij}.$$


It is straightforward to verify that


$$\langle F, \nabla f\rangle _{\ell^2(E)} = \langle \nabla^\ast F,  f\rangle _{\ell^2(V)} = \langle -\mathrm{div}\, F,  f\rangle _{\ell^2(V)}.$$



### Graph Laplacian

Having the gradient and the divergence operators defined, we define the
graph *Laplacian* $\Delta :  \ell^2(V) \rightarrow \ell^2(V)$ as
$\Delta = \nabla^\ast \nabla = - \mathrm{div}\, \nabla$, or, explicitly,


$$(\Delta f) _i = \frac{1}{a _i} \sum _{(i,j) \in E } w _{ij} (f _i - f _j).$$


Observe how this expression manifests the meaning of the Laplacian as
the difference between the value of a field at a vertex and the
(weighed) average of its values in the surrounding.

Since the vertex set is finite, it is convenient to represent the
Laplacian as an $n \times n$ matrix. For that purpose, we denote the
edge weights by the $n \times n$ matrix $\bb{W} = (w _{ij})$, the vertex
weights by the diagonal matrix
$\bb{A} = \mathrm{diag}\\{ a _1,\dots, a _n \\}$, and by
$\bb{D} =  \mathrm{diag}\left\\{ \sum _{j: j\ne i} w _{ij}  \right\\}$ the
*vertex degree* matrix. In this notation, the graph Laplacian is given
by

$$\bb{\Delta} = \bb{A}^{-1} (  \bb{D} - \bb{W} ).$$



Different choices of $\bb{A}$ lead to different definitions of a
Laplacian. For $\bb{A} = \bb{I}$, the *unnormalized graph Laplacian*


$$\bb{\Delta} _{\mathrm{un}} =  \bb{D} - \bb{W}$$

is obtained. The choice
$\bb{A} = \bb{D}^{-1}$ leads to the *random walk Laplacian*


$$\bb{\Delta} _{\mathrm{rw}} =  \bb{I} - \bb{D}^{-1} \, \bb{W}.$$

The
term $\bb{D}^{-1} \, \bb{W}$ in the definition of the above operator can
be interpreted as a transition probability of random walks on the graph,
hence the name. Finally, when the graph is used as a discrete
approximation of the underlying continuous manifold (as is the case of
simplicial complexes a.k.a. *meshes*), its weight matrices $\bb{A}$ and
$\bb{W}$ are obtained from the discretized metric of the manifold.

Fourier transform on non-Euclidean domains
------------------------------------------

Thus far, we have constructed two types of non-Euclidean domains,
manifolds and graphs, that both had a similarly defined Laplacian
operator. Next, we are going to use the Laplacian to define an analog of
Fourier analysis. For convenience, we are going to construct the Fourier
transform on manifolds; the construction for graphs is straighforwardly
similar.

The Laplacian, being a self-adjoint operator, admits an *orthogonal
eigendecomposition*

$$\Delta \phi _i = \lambda _i \phi _i.$$

The
eigenvalues $\lambda _i$ (called the *spectrum* of the Laplacian) can be
furthermore shown to be non-negative, a manifestation of the fact that
the Laplacian is a postive semi-definite operator (by analogy, think of
a matrix defined through $\bb{\Delta} = \bb{\nabla}^\Tr \bb{\nabla}$).
On Euclidean domains, the eigenfunctions of the Laplacian are simply
complex exponentials.

A scalar field $f \in L^2(\mathcal{M})$ can be represented in the
Laplacian eigenbasis as

$$f = \sum _{i \ge 0} \hat{f} _i \phi _i$$

with the
coordinates $\hat{\bb{f}} = \\{ \hat{f} _i \\} _{i \in \mathbb{Z}}$. Because
of orthonormality of the eigenfunctions, the coefficients $\hat{f} _i$
are given by


$$\hat{f} _i = \langle f, \phi _i \rangle _{L^2(\mathcal{M}) }.$$

We will
call the operator
$\mathcal{F} : L^2(\mathcal{M}) \rightarrow \ell^2(\mathbb{Z})$ defined
as


$$\mathcal{F} f = \{  \langle f, \phi _i \rangle _{L^2(\mathcal{M}) } \} _{i \in \mathbb{Z}}$$


as the *Fourier transform* (analysis) on $\mathcal{M}$. The inverse
(synthesis) transform
$\mathcal{F}^{-1} : \ell^2(\mathbb{Z}) \rightarrow L^2(\mathcal{M})$ is
given by


$$\mathcal{F}^{-1} \hat{\bb{f}} = \sum _{i \ge 0} \hat{f} _i \phi _i.$$

As
before, it is easy to show that the above two operations are adjoint
w.r.t. the standard inner products on $L^2(\mathcal{M})$ and
$\ell^2(\mathbb{Z})$.

Convolution on non-Euclidean domains
------------------------------------

Recall that one of the principal properties the Fourier transform
enjoyed on Euclidean domains was the fact that it diagonalized Toeplitz
operators. In fact, we had the property


$$\mathcal{F}(\bb{f} \ast \bb{g}) =  \mathcal{F}\bb{f} \cdot \mathcal{F} \bb{g}.$$


Unfortunately, the absense of a trivially defined translation group on
general non-Euclidean domains does not allow to generalize convolution,
which makes the left-hand-side of the above equation undefined. However,
the right-hand-side, being simply an element-wise product of frequency
responses, is perfectly defined, so we will use it to *define*
convolution on the non-Euclidean domain as


$$f \ast g = \mathcal{F}^{-1} (  \mathcal{F} f \cdot \mathcal{F} g ) = \sum _{i \ge 0}  \langle f, \phi _i \rangle _{L^2(\mathcal{M}) }  \langle g, \phi _i \rangle _{L^2(\mathcal{M}) } \phi _i.$$


The standard shift-invariance (or, more precisely,
translation-equivariance) property of convolution on Euclidean domains
is lost of course. Using a signal processing metaphor, it can be
interpreted as a position-dependent filter, with the impulse response
that can differ significantly at different locations in the domain.

When dealing with discrete domain such as graphs (which we will
henceforth assume for convenience), the Fourier transform and its
inverse have a matrix form. Note that the eigendecomposition of the
Laplacian $\bb{\Delta}$ can be written as
$\bb{\Delta} = \bb{\Phi} \bb{\Lambda} \bb{\Phi}^\Tr$, where $\bb{\Phi}$
has the eigenvectors as its columns and
$\bb{\Lambda} = \mathrm{\diag}\\{ \lambda _1,\dots,\lambda _n\\}$.
Representing vertex fields as $n$-dimensional column vectors, the
analysis (the forward transform) can be written as


$$\hat{\bb{f}} = \mathcal{F} \bb{f} = \bb{\Phi}^\Tr \bb{f} = ( \langle \bb{f}, \bb{\phi} _1  \rangle, \dots, \langle \bb{f}, \bb{\phi} _n  \rangle )^\Tr;$$


likewise, the synthesis operator (the inverse transform) assume the form


$$\bb{f} = \mathcal{F}^{-1} \hat{\bb{f}} = \bb{\Phi} \hat{\bb{f}}  = \bb{\phi} _1 \hat{f} _1 + \dots + \bb{\phi} _n \hat{f} _n.$$


In this notation, the convolution of two fields $\bb{f}$ and $\bb{g}$
can be written as


$$\bb{f} \ast \bb{g} = \bb{\Phi} ( (\bb{\Phi} \bb{f}) \odot (\bb{\Phi} \bb{g}) ),$$


where $\odot$ denotes the Hadamard (element-wise) product.

Convolution on Euclidean domains was an operation commuting with any
translation-equivariant (Toeplitz) operator, including the Laplacian. In
generalizing it to non-Euclidean domains, we only demanded commutativity
with the Laplacian.

Spectral CNN
============

The spectral definition of a convolution-like operation on a
non-Euclidean domain allows to parametrize the action of a filter as


$$\mathcal{W} \bb{f} = \bb{\Phi} \hat{\bb{W}} \bb{\Phi}^\Tr \bb{f},$$


where $\hat{\bb{W}}$ is a diagonal weight matrix containing the filter's
frequency response on the diagonal. In the space domain, it amounts to
applying the operator $\bb{W} = \bb{\Phi} \hat{\bb{W}} \bb{\Phi}^\Tr$ to
$\bb{f}$, by computing the inner products of $\bb{f}$ with every row of
$\bb{W}$ and stacking the resulting numbers into a vertex field.
Different weight matrices $\hat{\bb{W}}$ realize different such
operators.

Note that the definition is basis-dependent: a change in the domain,
and, consequently, in $\bb{\Phi}$ may translate the same $\hat{\bb{W}}$
into a completely different operator. Therefore, this construction must
assume the domain fixed; if we learn the weights $\hat{\bb{W}}$, they
will typically generalize rather poorly even to similarly-looking
domains. Siuch a complication did not exist on Euclidean domains.

Armed with the notion of a generalized convolution on non-Euclidean
domains, we can mimick the construction of a regular CNN. For this
purpose, we construct a *spectral convolutional layer* accepting an
$m$-dimensional vertex field $\bb{x} = (\bb{x}^1,\dots,\bb{x}^m)$ and
outputting an $m'$-dimensional vertex field
$\bb{y}= (\bb{y}^1,\dots,\bb{y}^{m'})$, whose $i$-the dimension is
defined according to


$$\bb{y} _j = \varphi\left( \sum _{i=1}^m \bb{\Phi} \hat{\bb{W}}^{ij} \bb{\Phi}^\Tr \bb{x}^i \right),$$


where $\varphi$ is an element-wise non-linearity such as ReLU, and
$\hat{\bb{W}}^{ij}$, are diagonal matrices parametrizing the filters of
the layer.

Strided convolution
-------------------

Recall that a typical Euclidean CNN architecture used strided
convolutions of the form


$$(\downarrow _{\bb{p}} \left( \bb{w} \ast \bb{x} \right)) _{\bb{k}} =  (\bb{w} \ast \bb{x} ) _{ \bb{p} \odot \bb{k} } = \sum _{ (i _1,\dots,i _d) } w _{i _1,\dots,i _d} \, x _{p _1 k _1 - i _1,\dots,p _d k _d-i _d  },$$


where $\bb{p} = (p _1,\dots,p _d)$ is a $d$-dimensional vector of strides.
This can be thought of projecting the result of the convolution
$\bb{w}\ast\bb{x}$ performed on $\mathbb{Z}^d$ onto the coarser domain
$\downarrow _{\bb{p}} \mathbb{Z}^d$. The subsampling operator
$\downarrow _{\bb{p}}$ can be thought of as a projection of a signal on
$\mathbb{Z}^d$ onto $\downarrow _{\bb{p}} \mathbb{Z}^d$.

The non-Euclidean analog can be constructed along the same lines. Let
$\mathcal{G}$ be the original domain of size $n$ with the Laplacian
$\bb{\Delta} = \bb{\Phi} \bb{\Lambda} \bb{\Phi}^\Tr$, and let
$\tilde{\mathcal{G}}$ be its coarsened (sub-sampled) version containing
$\tilde{n} = \alpha n < n$ vertices. We denote by
$\tilde{\bb{\Delta}} = \tilde{\bb{\Phi}} \tilde{\bb{\Lambda}} \tilde{\bb{\Phi}}^\Tr$
the corresponding Laplacian and its eigendecomposition. To keep the
previous notation, we denote by
$\downarrow _\alpha : \mathcal{G} \rightarrow \tilde{\mathcal{G}}$ the
projection onto the coarse domain, i.e., $\downarrow _\alpha$ maps a
vertex field on $\mathcal{G}$ to a vertex field on
$\tilde{\mathcal{G}}$. In matrix form, $\downarrow _\alpha$ is an
$\tilde{n} \times n$ matrix whose $i$-th row encodes the position of the
$i$-th vertex of the coarse domain $\tilde{\mathcal{G}}$ in the fine
domain $\mathcal{G}$.

The eigenvectors $\bb{\Phi}$ and $\tilde{\bb{\Phi}}$ of the fine and the
coarse Laplacians, $\bb{\Delta}$ and $\tilde{\bb{\Delta}}$, satisfy the
following multi-resolution property:


$$\tilde{\bb{\Phi}} \, \approx \,\, \downarrow _\alpha \bb{\Phi} \bb{P} _{\alpha},$$


where the $n \times \tilde{n}$ matrix


$$\bb{P} _\alpha = \left( \begin{array}{c} \bb{I} _{\alpha n} \\ \bb{0} \end{array} \right)$$


denotes the projection onto the lowest $\tilde{n} = \alpha n$
frequencies. This property essentially means that only the first
$k = \alpha n$ components of the spectrum can be retained. Thus, the
strided convolutional layer assumes the form


$$\bb{y} _j = \varphi\left( \sum _{i=1}^m \tilde{\bb{\Phi}} _k \hat{\bb{W}}^{ij} \bb{\Phi} _k^\Tr \bb{x}^i \right),$$


where $\bb{\Phi} _k = (\bb{\phi} _1,\dots\bb{\phi} _k)$ is the *truncated*
eigenbasis of the fine Laplacian containing the first $k$ eigenvectors,
and the weight matrices $\hat{\bb{W}}^{ij}$ are now
$\tilde{n} \times \tilde{n}$. The layer accepts an $m$-dimensional
vertex field $\bb{x} = (\bb{x}^1,\dots,\bb{x}^m)$ on $\mathcal{G}$ as
the input and produces and $m'$-dimensional vertex field
$\bb{y} = (\bb{y}^1,\dots,\bb{y}^{m'})$ on $\tilde{\mathcal{G}}$ as the
output.

Spatial localization
--------------------

Note that in our construction of a spectral convolutional layer, each
weight matrix has $k=\mathcal{O}(n)$ degrees of freedom, so that each
layer has $\mathcal{O}(nmm')$ degrees of freedom, unlike the regular
CNN, in which the layer was parametrized in the spatial domain by a
fixed-size kernel with the number of parameters independent on the
domain size $n$. In order to keep the number of parameters under control
and avoid overfitting, we would like to impose spatial localization onto
the weights $\hat{\bb{W}}^{ij}$, that is, ensure that the vertex fields
defined by every row of the operator
$\bb{W} = \bb{\Phi} \hat{\bb{W}} \bb{\Phi}^\Tr$ are spatially localized.

On a Euclidean domain, the spatial localization of a signal
$w : \mathbb{Z} \rightarrow \mathbb{R}$ is controlled by the decay of
its moments, defined as


$$\mu _p^2(w) = \sum _{k \in \mathbb{Z}}  k^{2p} w^2 _k = \| k^p \cdot w _k  \| _{\ell^2(\mathbb{Z}) }^2.$$


The faster $\mu _p^2(w)$ vanishes as $p$ increases, the more localized is
$w$. From


$$\frac{\partial}{\partial \xi} \mathcal{F}w  = \frac{\partial}{\partial \xi} \left( \sum _{k \in \mathbb{Z}} w _k e^{-i 2\pi \xi k} \right) =   \sum _{k \in \mathbb{Z}}  -i 2\pi k w _k e^{-i 2\pi \xi k} =  -i 2\pi  \mathcal{F}(k w _k)$$


we obtain the property


$$k^p \cdot w _k  \, \mathop{\longleftrightarrow}^{\mathcal{F}} \, \left( \frac{i}{2\pi} \right)^p \frac{\partial^p \hat{w}}{\partial \xi^p}.$$


Invoking Parseval's identity,


$$\mu _p^2(w) = \| k^p \cdot w _k  \| _{\ell^2(\mathbb{Z}) }^2 =  \left\|   \left( \frac{i}{2\pi} \right)^p \frac{\partial^p \hat{w}}{\partial \xi^p}  \right\| _{L^2([0,1]) }^2 = \frac{1}{(2\pi)^{2p}} \int _{[0,1]} \left|  \frac{\partial^p \hat{w}(\xi) }{\partial \xi^p} \right|^{2}   d\xi.$$


This result implies that fast decay of $\mu _p^2(w)$ implies fast decay
of the derivatives of $\hat{w}$, or, said differently, localization in
the spatial domain is equivalent to smoothness in the frequency domain
(the fact that smoothness is opposite to localization brings forth the
renowned Heisenberg's uncertainty principle). Smoothness of the
frequency response $\hat{w}$ can be asserted by representing it in an
underdetermined smooth basis or, equivalently, specifying it only at a
small set of frequencies and completing the rest via some smooth
interpolation.

This idea can be generalized to non-Euclidean domains. The only
complication is that while in $\RR^d$ we had a trivia notion of
smoothness arising in the spectrum, since the similarity between two
basis functions $\phi^{\bb{\xi}} = e^{i 2 \pi \bb{x}^\Tr \bb{\xi}}$ and
$\phi^{\bb{\xi}'} = e^{i 2 \pi \bb{x}^\Tr \bb{\xi}'}$ could be
quantified as the distance $\|\bb{\xi} - \bb{\xi}' \|$, there is not
such a standard notion in the spectrum of a general non-Euclidean
domain. A formal way to define smoothness is by constructing a dual
graph whose weights $w^\ast _{ij}$ reflect the similarity between the
eigenvectors $\bb{\phi} _i$ and $\bb{\phi} _j$ of the Laplacian of the
original (primal) graph. The question of how to define such a dual graph
the smoothness on which will lead to maximal localization on the primal
graph is still open. However, empirical evidence shows that at least in
some cases, the simple definition of
$w^\ast _{ij} = | \lambda _i - \lambda _j|$ leads to reasonable
localization.

With this notion of smoothness in mind, we fix a set of $q$ smooth basis
functions $\beta _1(\lambda), \dots, \beta _q(\lambda)$ (e.g., cubic
splines) and sample them at
$\lambda \in \\{ \lambda _1,\dots, \lambda _k \\}$. We arrange the samples
into a $k \times q$ matrix $\bb{B}$ with the elements
$b _{rs} = \beta _s(\lambda _r)$. The spectral weight matrices
$\hat{\bb{W}}^{ij}$ can now be defined as


$$\hat{\bb{W}}^{ij} = \mathrm{diag}\{ \bb{B} \bb{\alpha}^{ij}  \},$$


where $\bb{\alpha}^{ij}$ are $q$-dimensional interpolation coefficients.
In order to render the layer complexity independent of the domain size,
one has to choose $q = \mathcal{O}(1)$.

Spatial CNN
===========

One of the main disadvantages of the spectral construction of a
convolutional layer is its high computational complexity. The
multiplication by $\bb{\Phi}$ and $\bb{\Phi}^\Tr$ in the forward and
backward passes require $\mathcal{O}(n^2)$ operations, which quickly
becomes prohibitively expensive for large domains. Unlike Euclidean
domains on which the forward and inverse Fourier transforms can be
carried out using FFT in $\mathcal{O}(n \log n)$ operations, no such
fast algorithms exist for general non-Euclidean domains. In what
follows, we will reformulate the convolutional layer in a way free of
the costly Laplacian eigendecomposition and explicit projection on its
basis.

Let us substitute
$\hat{\bb{W}} = \mathrm{diag}\\{ \bb{B} \bb{\alpha}  \\}$ and examine the
spatial representation of the linear part of the layer:

$$\begin{aligned}
\bb{W} &=& \bb{\Phi} _k \hat{\bb{W}} \bb{\Phi} _k^\Tr  = \bb{\Phi} _k \left( \begin{array}{ccc}
\sum _{i = 1}^q \alpha _i \beta _i(\lambda _1) & & \\
& \ddots & \\
& &  \sum _{i = 1}^q \alpha _i \beta _i(\lambda _n) 
\end{array}   \right) \bb{\Phi} _k^\Tr \end{aligned}$$

(note that we
assumed $\beta _i(\lambda) = 0$ for $\lambda > \lambda _k$). Denoting by


$$b(\lambda) =  \sum _{i = 1}^q \alpha _i \beta _i(\lambda),$$

we have


$$\bb{W} = \bb{\Phi} \, \mathrm{diag}\{ b(\lambda _1), \dots, b(\lambda _n) \} \, \bb{\Phi}^\Tr.$$


(note that we assumed $b(\lambda) = 0$ for $\lambda > \lambda _k$).

Since $b(\lambda)$ is typically a polynomial (of degree $3$ in case of
cubic splines), let us examine how to rewrite it directly in the spatial
domain. Let $\bb{\Delta} = \bb{\Phi} \bb{\Lambda} \bb{\Phi}^\Tr$ be the
eigendecomposition of the Laplacian, and suppose we woud like to compute
$\bb{\Delta}^p$ for some integer power $p$. Then,

$$\bb{\Delta}^p =
\bb{\Phi} \bb{\Lambda} \bb{\Phi}^\Tr  \cdots   \bb{\Phi} \bb{\Lambda} \bb{\Phi}^\Tr =
\bb{\Phi} \bb{\Lambda}^p \bb{\Phi}^\Tr = \bb{\Phi} \, \mathrm{diag}\{ \lambda _1^p, \dots, \lambda _n^p \} \, \bb{\Phi}^\Tr.$$

Using linearity, we can conclude that for any polynomial


$$b(\lambda) = \sum _{i=0}^{r} \alpha _i \lambda^i,$$

one has

$$
b(\bb{\Delta}) =  \sum _{i=0}^{r} \alpha _i \bb{\Delta}^i =
\bb{\Phi} \, \mathrm{diag} \left\{ b(\lambda _1),\dots, b(\lambda _n) \right\} \, \bb{\Phi}^\Tr.
$$


In other words, the expensive right-hand-side can be simply evaluated as
applying the polynomial $b$ directly to the Laplacian. The Laplacian is
typically a sparse $n \times n$ matrix with $\mathcal{O}(1)$ non-zero
entries in every row. In such cases, computing its powers takes
$\mathcal{O}(n)$ operations, and the entire calculation is
$\mathcal{O}(nr)$. Also note that since the Laplacian is a local
operator acting on $1$-rings, its highest power $\bb{\Delta}^{r}$ will
act on $r$-rings, keeping the operator $b(\bb{\Delta})$ spatially
localized.

Using this observation, we can reformulate the convolutional layer
directly in the spatial domain as

$$\bb{y} _j = \varphi\left( \sum _{i=1}^m  \sum _{k=0}^r \alpha _k^{ij} \bb{\Delta}^k  \bb{x}^i \right),$$


[^1]: The term *shift-invariant* is so abundant in the signal processing
    and machine learning literature that we will not even attempt to
    change this unfortunate fact. However, it is worth noting that the
    correct mathematical term would be *shift-equivariant*. In general,
    let $f : \mathbb{U} \rightarrow \mathbb{V}$ be an operator mapping
    from some domain $\mathbb{U}$ to some co-domain $\mathbb{V}$, and
    let $\mathcal{G}$ be a group of transformations that can be applied
    both to the domain and the co-domain. The operator $f$ is said
    *invariant* to the action of $\mathcal{G}$ if $f \circ \tau = f$ for
    every $\tau \in \mathcal{G}$. On the other hand, the operator is
    *equivariant* if $f \circ \tau = \tau \circ f$.
