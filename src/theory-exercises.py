# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: default
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Computational Probabilistic Inference Theory Exercises
#
# ## Setup

# %%
# setup

# %% [markdown]
# ## Exercise 1
#
# Life in high-dimensional spaces has some surprising features compared to our low-dimensional intuitions.  Consider the following calculations:
# * Within the $d$-dimensional unit ball, the sub-ball of radius $0.99$ captures a $(0.99)^d$-fraction of the volume, a fraction which decreases *exponentially* in $d$.  Thus not much volume is to be found near anyone point; rather, volume accumulates more due to the sheer multiplicity of directions in which it can be packed out around the point.
# * Consider probability distributions $p$ on the $d$-dimensional unit cubes $[0,1]^d$, and we subdivide each axis of the latter into $N$ equal sub-intervals to get $N^d$ sub-cubes.  We can informally distinguish between two cases.  On the one hand, $p$ could be *spread out*, in that $p$ does not assign disproportionately much mass to any one sub-cube.  In this case, $p$ cannot assign much more than $1/N^d$ mass to each cube---an amount that is exponentially small in $d$.  On the other hand, $p$ could exhibit *concentration*, where some sub-cubes are assigned rather more than $1/N^d$ mass.  In this case, the proportion of the sub-cubes that can recieve non-negligible mass must be an exponentially small in $d$, leading to a needle-in-haystack situation: "typical sets" are hard to locate, describe, or generate samples over.
#
# Related phenomena arise for multivariate normal distributions $p$ in $d$ variables, as $d$ grows.  For simplicity we take $p = \mathrm{Norm}(0_d,\mathrm{I}_d)$ to be origin-centered and have identity covariance, so that $p = \mathrm{Norm}(0,1)^d$ is equivalent to the indepenent product of $d$ univariate normals of mean zero and unit covariance.  This distribution has density $p(x) = (2\pi)^{-d/2}\exp(-r^2/2)$ where $r = |x|$.  The *$\chi$ distribution* $q$ is, by definition, the pushforward of $p$ under the norm map; it embodies the distribution of norms $r = |x|$ for vectors $x \sim p$.  A little multivariable calculus shows that $q$ has density
# $$
# q(r) = \frac2{2^{d/2}\Gamma(d/2)}r^{d-1}\exp(-r^2/2).
# $$
#
# 1. Where is the density function $q(r)$ increasing (resp. decreasing)?  What is the mode, i.e. the norm $r = r_0$ that maximizes the density?  Note the asymptotic behavior of this mode as $d \to +\infty$.
# 2. Using a monotonicity property of $q(r)$ from part (1), show how to give a simple upper bound on the probability $q(0 \leq r \leq r_1)$ when $r_1$ satisfies $0 \leq r_1 \leq r_0$.  Then, taking $r_1 = 1$, explain the asymptotic behavior of the probability $q(0 \leq r \leq r_1)$ as $d \to +\infty$.
#
# Here is a discrete variant.  For $0 < \theta < 1$, let $\mathrm{Bern}(\theta)$ be the Bernoulli distribution on $\{0,1\}$ of weight $\theta$.  We consider $p = \mathrm{Bern}(p)^d$, with density $p(x) = \theta^{\sum_i x_i}(1-\theta)^{d-\sum_i x_i}$.  The *binomial distribution* $\mathrm{Binom}(\theta, d)$ is, by definition, the pushforward of $p$ under the sum map; it embodies the distribution over numbers $r = \sum_i x_i$ of heads for sequences $x \sim p$.  A count of the possibilities shows that $q$ has the density
# $$
# q(r) = \binom dr \theta^r(1-\theta)^{d-r}.
# $$
# Fix $\theta = 0.9$ in the following.
#
# 3. Where is the density function $q(r)$ increasing (resp. decreasing)?  Hint: determine when $q(r+1)/q(r)$ is greater or less than $1$.  What is its mode $r_1$, to within an integer?  Note the asymptotic behavior of $r_1/d$ as $d \to +\infty$.
# 4. Explain the asymptotic behavior of the probability $q(r = d)$ as $d \to +\infty$.
#
# In both of these cases, first drawing a sample $x$ from one distribution $p$, then using it to produce samples $r$ from a derived distribution $q$ under pushforward, must be done with care.  Explain the following phenomena in terms of the earlier discussion of high-dimensional geometry.
#
# 5. Starting from a value $x$ of high density can lead to values $r$ of low probability.
# 6. The likely values of $r$ come from values of $x$ with much lower density than the mode, and are not even near it.

# %% [markdown]
# [Enter your answer here]

# %% [markdown]
# ## Exercise 2
#
# Let $p_1$ be the distribution over the unit interval $[0,1]$ with density function $p(x) = 6x(1-x)$.  Suppose that in our applications we are concerned with quantities $r = r(x^2)$ that only make use of $y := x^2$, so to simplify we introduce the pushforward distribution $q$ of $p$ under the square map, embodying such $y$ where $x \sim p$.  A little calculus shows that $q$ has the density $q(y) = 3(1-\sqrt y)$ on $[0,1]$.
#
# 1. The distribution $p$ has mode at $x = \frac12$ in the center of the domain, whereas $q$ has mode at $y = 0$ on the end.  How can these behaviors be so different while representing the same situation?  When speaking of likely values of $r$, should we take $r(x^2)$ with $x$ near $\frac12$, or should we take $r(y)$ with $y$ near $0$, or should we do something else?
# 2. Our parameter transformation $y = x^2$ has derivative $\frac{\mathrm{d}y}{\mathrm{d}x}$ with a zero at $x = 0$.  What is the relationship between this fact and the very different shapes of $p(x)$ and $q(y)$?

# %% [markdown]
# [Enter your answer here]

# %% [markdown]
# ## Exercise 3
#
# Probabilistic reasoning is often structured so that we consider a distribution $p$ over *latent* variables $x$, perhaps some posterior distribution given some information, as well as some quantity $r$ that depends on $x$ (perhaps stochastically).  Our knowledge of $p$ gives rise to values for $x$, which we *deductively* parlay into conclusions about $r$.
#
# The approach of *parameter optimization* would have us *find the best $x$* and then use it in reasoning about $r$.  The preceding two exercises demonstrate how doing so is ill-posed.
# * According to Exercise 2, the modal $x$ is parameterization-dependent.  Natural nonlinear changes of variables give rise to inequivalent optimization problems.
# * According to Exercise 1, the mass of the distribution is not even concentrated near the mode.
# * According to both, optimization is not compositional.  The optimization conclusions are not stable under deductive reasoning.
#
# We with the approach of *sampling processes*, which would have us reason about $r$ by composing it, as a process, with the process for generating samples distributed according to $p$.
# * I need a problem to solve here.
# * Should show:
#   * It gives answers.
#   * It gives different answers than optimization.
#   * It gives better answers.

# %% [markdown]
# ## Exercise 4
#
# What properly weighted samplers are.
#
# Show how rejection sampling promotes them to their targets, provided an explicit bound on the weight.
#
# Show generically how joint density gives unnormalized posterior PWS.
#
# Implement this in a little discrete example.

# %% [markdown]
# ## Exercise 5
#
# Hierarchical Bayes and model selection
#
# See [slides](https://www.doc.ic.ac.uk/~mpd37/teaching/2014/ml_tutorials/2014-01-29-slides_zoubin1.pdf) 23, 24 then 22, 25.
#
# [Occam's razor](https://v1.probmods.org/occam's-razor.html)
#
# Any model implicitly focuses on some information ($x$), while marginalizing out all remaining information as latent ($\theta$), that is, working with
# $$
# p(x) = \int_\Theta p(x|\theta)\,\mathrm{d}p(\theta).
# $$
# Increasing our model complexity to account for more information amounts to working with a particular value of $\theta$ instead of averaging it out, thus working with
# $$
# p(x|\theta)\,p(\theta) = p(x,\theta),
# $$
# or, the joint distribution.  We express the ratio of these densities as the product of two factors,
# $$
# \frac{p(x,\theta)}{p(x)} =  \frac{p(x|\theta)}{p(x)} \cdot p(\theta).
# $$
# First is the ratio $p(x|\theta)/p(x)$, which expresses how conditioning on $\theta$ affects our focus on the information $x$.  Second is $p(\theta)$, which expresses the associated cost: we are then required to allocate mass across the values of $\theta$, and homing in on the values of interest contracts the resulting probability.  This second factor, especially when applied many times while accounting for more and more information, is how the curse of dimensionality enters the picture.
#
# Model selection
# * coin flipping?  curve fitting?
# * Showing off "why rationality can protect against (the most basic form of) overfitting, assuming computationally-bounded hierarchical models"

# %% [markdown]
# ## Exercise 6
#
# Asymptotic Consistency
#
# Estimate any continuous density on the real line with a mixture of Gaussians, then can make good approximation of broad range of datasets.  See [slide 26 here](https://docs.google.com/presentation/d/1LdO6SPAFyC99Gb2QHa8-ikLLYluOPOB9MTavuOLTY9I/edit#slide=id.g10d6a8dad9_0_593) for picture.
#
# (Connect to slide 19 of prior slides.)
#
# GenJAXMix demo to illustrate it.
#
# > is partly why the bitter lesson is wrong: can have models that just memorize the data, and then the explanation is shallow
