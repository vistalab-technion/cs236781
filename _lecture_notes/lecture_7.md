---
title: "Lecture 7: Reinforcement learning"
excerpt: foo bar
copyright: alex
---

Until now, we have seen two learning regimes: the *supervised* regime,
in which the learning system attempts to learn a latent map based on
example of its input-output pairs, and the *unsupervised* regime, in
which the learning system attempts to build a model for the data
distribution. In what follows, we will consider another learning
setting, in which a decision-making system is trained to make optimal
decisions.

The basic setting of our problem will be that of an *agent* acting in an
*environment*. At every point of time, the agent observes the state of
the environment and decides on an action that changes that state. For
each such action, the agent is given a reward signal (which can be
negative). The agent's role is to maximize the total received reward.

Markov decison processes
========================

Let us now formalize the above setting. Time $t$ is assumed to be
discrete incremented in steps of $1$. We assume that at every time, the
environment can be found in one of a finite set of states
$s _t \in \mathcal{S}$. The state of the environment will be further
assumed *fully observable* by the agent. At every time, the agent may
take one of a finite set of actions $a _t \in \mathcal{A}$. As the result
of the agent's action, the environment will transition to a new state
$s _{t+1}$ at the next time. The transition rule is, generally,
stochastic and can be characterized by the *transition probability*,
which is the probability
$\mathbb{P}(s _{t+1} | s _{t},a _{t}, s _{t-1},a _{t-1},\dots,s _0,a _0)$ of
the future state conditional on the present and past states and actions.
We assume the random process underlying such transitions to obey the
*Markov property* implying that the conditional probability of the
future state depends only on the present state and action,


$$\mathbb{P}(s _{t+1} | s _{t},a _{t}, s _{t-1},a _{t-1},\dots,s _0,a _0) = \mathbb{P}(s _{t+1} | s _{t},a _{t}).$$


In other words, the effect of an action depends only on the present
state and not the past history. Furthermore, we assume the transition
probability to be *time-invariant* (which does not imply process
stationarity!) In view of these properties, we will denote by
$P _a(s,s') = \mathbb{P}(s _{t+1}=s' | s _t=s, a _t = a)$ the transition
probability from state $s$ to state $s'$ under the action $a$. As the
result of the transition, the agent receives a scalar *immediate reward*
$r _{t+1} = R(s _{t},a _t)$, which is assumed deterministic (or the
expectation of a stochastic reward).

In order to quantify the *return* (or the *total reward*) that an agent
will receive, we are tempted to sum the immediate rewards in time.
However, this will generally yield an infinite sum. A way to overcome
this is by setting a *finite horizon*, summing only for a finite set of
time steps into the future:

$$g _t = \sum _{k = 0}^{n-1} r _{t+1+k},$$

A
smoothed version of a finite horizon reward is known as the *cumulative
discounted reward*

$$g _t = \sum _{k \ge 0} \gamma^k \, r _{t+1+k},$$

where
$\gamma \in [0,1)$ is a *discount factor* giving lower importance to
remote future rewards (vita brevis est). Due to its tractability, this
is a very popular choice for modelling the return.

The tuple $(\mathcal{S},\mathcal{A},P,R,\gamma)$ is known as a *Markov
decision process* (MDP) and can be thought as the set of game rules by
which the agent is obliged to play. Usually, the state set $\mathcal{S}$
will contain a particular *terminal state* (or few such states)
indicating the end of the game (e.g., the agent has died or won the
game). In such cases, the state-action-reward sequence will terminate at
some point, producing a single game *episode*


$$s _0,a _0,r _1, \,\, s _1,a _1,r _2, \,\, \dots, \,\, s _{t-1},a _{t-1},r _{t},\,\, s _t.$$


A sub-sequence representing a single state transition and the
corresponding reward, $s _{t},a _{t},r _{t+1},s _{t+1}$ is usually referred
to as an *experience*.

Policy
------

How does the agent know which action to take? The behavior of the agent
is fully defined by the conditional distribution


$$\pi(a|s) = \mathbb{P}(a _t = a | s _t = s)$$

known as a *policy*. This
formalism captures both stochastic and deterministic policies (in the
latter case, $a _t = f(s _t)$ and the above conditional distribution
becomes a singletone).

Value functions
---------------

Given an MDP and having the agent behavior fixed to some policy $\pi$,
we may predict how beneficial it is for the environment to be in a
certain state, or for the agent to take a certain action in a particular
state. This benefit (=return) is measured by *value functions*, which,
of course, depend on the selected policy.

The *state value function* of an MDP is the expected return of the agent
starting at state $s _t=s$ and following the policy $\pi$ at all
subsequent time steps,


$$v _\pi(s) = \mathbb{E}\left( g _t  \left| s _t = s, \pi \right.\right) = \mathbb{E}\left( \sum _{t \ge 0} \gamma^t \, r _{t+1}  \left| s _0 = s,\pi \right.\right).$$


Note that since our MDP is time-invariant, the exact value of starting
time $t$ is unimportant.

The *action value function* of an MDP is the expected return of the
agent starting at state $s _t=s$, taking action $a _t=a$, and then
following the policy $\pi$ at all subsequent time steps,


$$q _\pi(s,a) = \mathbb{E}\left( g _t  \left| s _t = s, a _t = a \pi \right.\right) = \mathbb{E}\left( \sum _{t \ge 0} \gamma^t \, r _{t+1}  \left| s _0 = s, a _0 = a, \pi \right.\right).$$



Expectation equations
---------------------

Let us have an explicit look at the state value function

$$\begin{aligned}
v _\pi(s) &=&  \mathbb{E}\left( g _0  \left| s _0 = s, \pi \right.\right) \\
&=& \mathbb{E}\left( r _1 + \gamma r _2 + \gamma^2 r _3 + \cdots \left| s _0 = s, \pi \right.\right) \\
&=& \mathbb{E}\left( r _1 + \gamma ( r _2 + \gamma r _3 + \cdots  ) \left| s _0 = s, \pi \right.\right) \\
&=& \mathbb{E}\left( r _1 + \gamma g _{t+1} \left| s _0 = s, \pi \right.\right) \\
&=& \mathbb{E}\left( r _1 + \gamma v _\pi(s _{t+1}) \left| s _0 = s, \pi \right.\right).\end{aligned}$$

Note that the function under the expectation decomposes into two terms:
the immediate reward $r _{t+1}$ and the discounted value of the successor
state reward $v _\pi(s _{t+1})$. Spelling out the expectation, we obtain

$$\begin{aligned}
v _\pi(s) &=& \sum _{a \in \mathcal{A}} \pi(a|s) \left(  R(s,a) + \gamma \sum _{s' \in \mathcal{S}} P(s'|s,a) v _\pi(s')  \right) \\
&=& \sum _{a \in \mathcal{A}}   R(s,a) \pi(a|s) + \gamma \sum _{s' \in \mathcal{S}} \left( \sum _{a \in \mathcal{A}} P(s'|s,a)\pi(a|s) \right) v _\pi(s')\end{aligned}$$

Expressing $v _\pi(s)$ over all states $s \in \mathcal{S}$ as a vector
$\bb{v} _\pi$, the first term in the right-hand-side as a vector
$\bb{r} _\pi$, and the parenthesis in the second term as the matrix
$\bb{P} _\pi$, we obtain the linear system


$$\bb{v} _\pi = \bb{r} _\pi + \gamma \bb{P} _\pi \bb{v} _\pi,$$

for which a
closed-form solution
$\bb{v} _\pi = (\bb{I} -  \gamma \bb{P} _\pi)^{-1} \bb{r} _\pi$ is
available.

In the same way, the action value function can be decomposed into

$$\begin{aligned}
q _\pi(s,a) &=&  \mathbb{E}\left( g _0  \left| s _0 = s, a _0 = 0 \pi \right.\right) \\
&=& \mathbb{E}\left( r _1 + \gamma r _2 + \gamma^2 r _3 + \cdots \left| s _0 = s, a _0 = a,\pi \right.\right) \\
&=& \mathbb{E}\left( r _1 + \gamma Q _\pi(s _{t+1},a _{t+1}) \left| s _0 = s, a _0 = a,\pi \right.\right) \\
&=& R(s,a) + \gamma   \sum _{s' \in \mathcal{S}} P(s'|s,a) \sum _{a' \in \mathcal{A}} \pi(a'|s') q _\pi(s',a').\end{aligned}$$

The state and the action value functions are related to each other via


$$v _\pi(s) = \sum _{a \in \mathcal{A}} \pi(a|s) q _\pi(s,a).$$



Optimal control
===============

Both value functions predict future reward. Starting at some initial
state $s _0 = s$ (and, perhaps, some initial action $a _0 = a$) and
running the game forward in time following a policy $\pi$, the MDP will
realize a certain trajectory
$\tau = \{ (s _{t},a _{t},r _{t+1}) \} _{t \ge 0}$ (since it is a stochastic
process, every game will realize a different trajectory). Each such
trajectory has a certain probability of being realized, and can be
associated with the return


$$g : \tau \mapsto \sum _{t \ge 0} \gamma^t \, r _{t+1}.$$

Both value
functions average the latter quantity over all possible trajectories
starting at a state $s$ in the case of $v _\pi(s)$, or a state-action
pair $(s,a)$ in the case of $q _\pi(s,a)$. Our desire to maximize the
return can be translated into an optimal control problem, which can be
informally stated as selecting such a policy making high-return
trajectory more probable.

Bellman equation
----------------

Let us define the so-called Bellman operator $T$ mapping a state value
function $u : \mathcal{S} \rightarrow \RR$ to a new state value function


$$(Tu)(s) =  \max _{a \in \mathcal{A}} \, R(s,a) + \gamma \sum _{s' \in \mathcal{S}} P(s'|s,a) u(s').$$


Given a value function $u$, we construct a determinisitic policy

$$\pi^\ast _u (a|s)= \left\{ \begin{array}{cl} 1 & : \,  a = \mathrm{arg}\max _{\alpha \in \mathcal{A}} \, R(s,\alpha) + \gamma \sum _{s' \in \mathcal{S}} P(s'|s,\alpha) u(s')
 \\
0 & : \, \mathrm{else}
\end{array}
\right.$$

It is straightforward to show that the state value function
associated with this policy is exactly given by the application of the
Bellman operator to $u$, $v _{\pi^\ast _u} = Tu$. It can also be shown
quite straightforwardly that $Tu \ge u$ in the sense that
$(Tu)(s) \ge u(s)$ for every $s \in \mathcal{S}$. In other words,
replacing a policy $\pi$ with a new policy $\pi' = \pi^\ast _{v _\pi}$
improves the policy in the sense that
$\pi^\ast _{v _\pi'}= Tv _\pi \ge v _\pi$.

This monotonicity property together with the fact that the return is
upper-bounded (by $\max _{s,a} R(s,a)/(\gamma-1)$) implies that $T^n v$
produces a convergent sequence, $T^n v \uparrow v^\ast$, with the limit
point being the fixed point of $T$,


$$v^\ast(s) = (Tv^\ast)(s) = \max _{a \in \mathcal{A}} \, R(s,a) + \gamma \sum _{s' \in \mathcal{S}} P(s'|s,a) v^\ast(s').$$


An optimal policy (not necessarily unique!) producing the above optimal
state value function is given by

$$\pi^\ast(a|s) = \pi^\ast _{v^\ast} = \left\{ \begin{array}{cl} 1 & : \,  a = \mathrm{arg}\max _{\alpha \in \mathcal{A}} \, R(s,\alpha) + \gamma \sum _{s' \in \mathcal{S}} P(s'|s,\alpha) v^\ast(s')
 \\
0 & : \, \mathrm{else}.
\end{array}
\right.$$

The latter resut is known as the *Belman equation* or
*Belman's optimality principle*. Informally, it states that an optimal
policy has the property that whatever is the initial state and initial
decision, the remaining decisions must constitute an optimal policy with
regard to the state resulting from the first decision.

In a very similar manner, the Bellman equation can be written in terms
of the action value function,


$$q^\ast(s,a) = R(s,a) + \gamma \sum _{s' \in \mathcal{S}} P(s'|s,a) \max _{a' \in \mathcal{A}} q^\ast(s',a'),$$


with the associated optimal policy

$$\pi^\ast(a|s) = \left\{ \begin{array}{cl} 1 & : \,  a = \mathrm{arg}\max _{a \in \mathcal{A}} \, q^\ast(s,a)
 \\
0 & : \, \mathrm{else},
\end{array}
\right.$$

which is identical to the one associated with the optimal
state value function. The two optimal value functions are related via


$$v^\ast(s) = \max _{a \in \mathcal{A}} q^\ast(s,a).$$



Dynamic programming
-------------------

Our previous discussion suggests a very simple recipe for finding the
optimal value function (and the corresponding optimal policy): a
fixed-point iteration of the Bellman operator. We start with an
arbitrary value function $q _0$ and produce a sequence of value functions


$$q _{n+1}(s,a) = R(s,a) + \gamma \sum _{s' \in \mathcal{S}} P(s'|s,a) \max _{a' \in \mathcal{A}} q _n(s',a')$$


for $n\ge 0$. This sequence converges to $q^\ast$ as $n$ approaches
infinity. This technique is typically known under the name of *dynamic
programming*, *backward induction* or *value iteration*.

However, despite its apparent simplicity, dynamic programming is
completely infeasible for problems with a moderately big state space,
not mentioning real-world problems with huge dimensionalities of
$\mathcal{S}$. Approximate solutions using learning systems are known
under the name of *reinforcement learning*. In what follows, we will
examine several such approaches. Reinforcement learning balances reward
accumulation and system identification (model learning) in case of
unknown dynamics ($R$ and $P$ of the MDP). The on-line nature of
reinforcement learning makes it possible to approximate optimal policies
in ways that put more effort into learning to make good decisions for
frequently encountered states, at the expense of less effort for
infrequently encountered states.

Value-based learning
====================

The idea of value-based learning is to learn a parametric function
$q _{\bb{\theta}}(s,a)$ (realized by a neural network with the parameters
$\bb{\theta}$) approximating the optimal value function $q^\ast$. Since
in most implementations the action value function is used, the method is
also known under the name of *$q$-learning* (when a deep neural network
is used as the approximator, it is also known as deep $q$ network or
DQN). In practice, the network is realized as a vector-valued function
$\bb{q} _{\bb{\theta}}(s)$ receiving the state $s$ and producing the
values of the function for every $a \in \mathcal{A}$. For example, an
agent playing Pacman receives as the state the set of pixels displayed
on the screen, and produces the value of the approximate $q$-function
for all the four control actions (up,down,left,right).

Recall that our goal is to find such a vector of parameters
$\bb{\theta}$ satisfying the Bellman equation


$$q _{\bb{\theta}}(s _t,a _t) = r _{t+1} + \gamma \sum _{s \in \mathcal{S}} P(s|s _t,a _t) \max _{a \in \mathcal{A}} q _{\bb{\theta}}(s,a).$$


We will relax the above equality in the least squares sense and define
the loss function


$$L(\bb{\theta}) = \mathbb{E} _{s,a} \left( y -  q _{\bb{\theta}}(s,a) \right)^2$$


where the expectation is in practice an empirical average on a
mini-batch of experiences of the form $(s _t,a _t,r _{t+1},s _{t+1})$; for
every such experience,

$$y = \left\{ \begin{array}{ll} 
r _{t+1} + \gamma  \max _{a \in \mathcal{A}} q _{\bb{\theta}^-}(s _{t+1},a) & : \, s _{t+1} \,\, \mathrm{not\,terminal} \\
r _{t+1} & : \, s _{t+1} \,\, \mathrm{terminal}. 
\end{array}\right.$$

Here $\bb{\theta}^-$ denotes the previous vector of
parameters to emphasize that $y$ is constant w.r.t the optimization
variable of the loss $L(\bb{\theta})$. Note that we do not average the
second term over $s \in \mathcal{S}$, since the average weights
$P(s|s _t,a _t)$ are typically unknown (the agent is discovering the rules
of the game, and the MDP is latent at least initially). Since many such
$y$'s are averaged over the mini-batches, the weighting by the
transition probabilities arises naturally.

Experience replay
-----------------

The most natura way of constructing mini-batches for $q$-learning is by
taking sequences of consecutive samples, updating the network in
between. However, this is a very bad idea for several reasons. First,
the samples are correlated, which makes the learning inefficient.
Second, since the current parameters determine the next training
samples, the mini-batches are likely to be biased towards specific
states and actions. Such unhealthy feedback loops are avoided by using
the *experience replay* methodology. A replay cache of experiences of
the form $(s _t,a _t,r _{t+1},s _{t+1})$ is constantly updated as the game
is played. Mini-batches are drawn at random from the cache.

In a typical learning scenario, entire episodes are played
consecutively. With the environment currently present at state $s _t$ at
time $t$ in the episode, the greedily optimal action


$$a _t = \mathrm{arg}\max _{a \in \mathcal{A}}  q _{\bb{\theta}}(s _t,a)$$


is selected and is executed agains the emulated environment, which
returns the next state $s _{t+1}$ and the reward $r _{t+1}$. The tuple
$(s _t,a _t,r _{t+1},s _{t+1})$ is inserted into the cache.

In order to allow the agent to balance the exploration of new states and
actions vs. the exploitation of the learned policy, with some small
probability $\epsilon \in (0,1)$, the greedy optimal action is replaced
with a uniformly random action on $a _t \sim U(\mathcal{A})$.

Policy-based learning
=====================

While value-based learning is much more tractable than dynamic
programming (and also allows to implicitly discover the underlying MDP),
the approximate $q$-function might still be very complicated in real
settings. Often, the policy itself is a much simpler function.
*Policy-based* learning methods learn a policy $\pi _{\bb{\theta}}(a|s)$
from some parametric family of functions (for deterministic policies,
the network has the form $a = \pi _{\bb{\theta}}(s)$, receiving a state
and producing an action $a$).

A natural score function to associate with a policy $\pi _{\bb{\theta}}$
is the expected return


$$J( \bb{\theta} )  =  \mathbb{E} \left(  g(\tau) | \pi _{\bb{\theta}}  \right) =  \mathbb{E} \left(  \sum _{t \ge 0} \gamma^t \, r _{t+1} | \pi _{\bb{\theta}}  \right),$$


where the expectation is taken over all trajectories
$\tau = \{ (s _{t},a _{t},r _{t+1}) \} _{t \ge 0}$ realizable under the
policy $\pi _{\bb{\theta}}$ with the probability distribution


$$P(\tau | \bb{\theta}) = \mathbb{P}(  \{ (s _{t},a _{t},r _{t+1}) \} _{t \ge 0} | \bb{\theta}) = \mathbb{P}(s _0) \prod _{t \ge 0} \pi _{\bb{\theta}}(a _t|s _t) P(s _{t+1} | s _t,a _t).$$


In these terms, we can re-write the objective as


$$J( \bb{\theta} )  = \mathbb{E} _{\tau \sim P(\tau | \bb{\theta}) }  \,  g(\tau)  = \int P(\tau | \bb{\theta}) g(\tau) d\tau.$$


Taking the gradient w.r.t. the network parameters results in

$$\begin{aligned}
\nabla _{\bb{\theta} } J( \bb{\theta} )  &=& \int \nabla _{\bb{\theta} } P(\tau | \bb{\theta}) g(\tau) d\tau
 = \int P(\tau | \bb{\theta})  \frac{\nabla _{\bb{\theta} } P(\tau | \bb{\theta})}{P(\tau | \bb{\theta}) } g(\tau) d\tau\\
&  = &\int P(\tau | \bb{\theta})  \nabla _{\bb{\theta} } \log P(\tau | \bb{\theta})  g(\tau) d\tau = 
\mathbb{E} _{\tau \sim P(\tau | \bb{\theta}) }  \left( \nabla _{\bb{\theta} } \log P(\tau | \bb{\theta})  g(\tau) \right).\end{aligned}
$$

The latter trick allows to write the seemingly intractable gradient of
the expectation as an expectation of the gradient of the log conditional
density $P(\tau | \bb{\theta})$. Let us now evaluate the latter gradient
explicitly. By observing that in the expression

$$\begin{aligned}
\log P(\tau | \bb{\theta})  &=& \log \mathbb{P}(s _0) + \sum _{t \ge 0}\pi _{\bb{\theta}}(a _t|s _t) + \sum _{t \ge 0} P(s _{t+1} | s _t,a _t)\end{aligned}
$$

only the second term depends on $\bb{\theta}$, we can write

$$\begin{aligned}
 \nabla _{\bb{\theta} } \log P(\tau | \bb{\theta})  &=& \sum _{t \ge 0}  \nabla _{\bb{\theta} }\log \pi _{\bb{\theta}}(a _t|s _t).
 \end{aligned}$$

The gradient of the score function reduces to
 
$$\begin{aligned}
\nabla _{\bb{\theta} } J( \bb{\theta} )  &=& 
\mathbb{E} _{\tau  }  \left( g(\tau)  \sum _{t \ge 0}  \nabla _{\bb{\theta} }\log \pi _{\bb{\theta}}(a _t|s _t) \right).\end{aligned}$$

When working with stochastic gradient, it further simplifies to

$$\begin{aligned}
\nabla _{\bb{\theta} } J( \bb{\theta} )  & \approx &   \sum _{t \ge 0}  g(\tau)  \nabla _{\bb{\theta} }\log \pi _{\bb{\theta}}(a _t|s _t).\end{aligned}$$

Making gradient ascent steps with the gradient of this form implies that
for high-return trajectories, the probability of all incurred actions
shall be increase, while for low-return trajectories, they should be
decreased.

A slightly less drastic approach would be to increase the probabilities
of of an action encountered only by the cumulative discounted future
reward from that state on,

$$\begin{aligned}
\nabla _{\bb{\theta} } J( \bb{\theta} )  &\approx&   \sum _{t \ge 0}  \left( \sum _{t' \ge t}  \gamma^{t'-t} r _{t'}  \right)  \nabla _{\bb{\theta} }\log \pi _{\bb{\theta}}(a _t|s _t),\end{aligned}$$

thus localizing the effect of an action in time. A problem that still
persists is that the absolute value of the reward is of little meaning
in the decision whether to increase or decrease the probability of a
certain action; what matters more is whether the action increases the
reward already expected in that state. Formally, this can be embodied by
subtracting a *baseline* $b(s _t)$

$$\begin{aligned}
\nabla _{\bb{\theta} } J( \bb{\theta} )  &\approx&   \sum _{t \ge 0}  \left( \sum _{t' \ge t}  \gamma^{t'-t} r _{t'}  - b(s _t) \right)  \nabla _{\bb{\theta} }\log \pi _{\bb{\theta}}(a _t|s _t),\end{aligned}$$

which can be, for example, a moving average of the rewards previously
observed from state $s _t$.

Actor-critic architecture
-------------------------

The desire to weight the gradient of $\log \pi _{\bb{\theta}}(a _t|s _t)$
by the difference between the expected future return if action $a _t$ is
taken and that expected by all actions taken from state $s _t$ suggests
that the weighing should be performed by the difference between the
action value function and the state value function,

$$\begin{aligned}
\nabla _{\bb{\theta} } J( \bb{\theta} )  &\approx&   \sum _{t \ge 0}  \left( q _{\pi _{\bb{\theta}}}(s _t,a _t) -  v _{\pi _{\bb{\theta}}}(s _t) \right)  \nabla _{\bb{\theta} }\log \pi _{\bb{\theta}}(a _t|s _t),\end{aligned}$$

Since we do not known the value functions, we can estimate them (or,
actually, their difference known as the *advantage function*
$a _{\pi _{\bb{\theta}}}(s _t,a _t)  = q _{\pi _{\bb{\theta}}}(s _t,a _t) -  v _{\pi _{\bb{\theta}}}(s _t)$
) using a neural network as we did in value-based learning. To that end,
we define another neural network $a _{\bb{\phi}}(s,a)$ parametrized by
${\bb{\phi}}$ aiming at estimating the advantage function
$a _{\pi _{\bb{\theta}}}(s _t,a _t)$ and train it simultaenously with the
policy $\pi _{\bb{\theta}}$. This approach combining value- and
policy-based learning is known as *actor-critic* architecture, since the
actor decides which action to take (the policy $\pi _{\bb{\theta}}$) and
the critic tells it how beneficial the action was (the value function
$a _{\bb{\phi}}$). Based on the latter, the actor knows how to adjust its
policy.

