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
# This exercise contains simple explorations of the so-called *curse of dimensionality*.

# %% [markdown]
# ## Exercise 2
#
# This exercise illustrates some difficulties with *parameter optimization* approaches, in contrast with *sampling*.
#
# Basic points:
# * They give different answers, and the answers from sampling are better.
# * Mass is not concentrated near the mode (cf. Ex 1).
# * Optimization can be parameterization-dependent in subtle ways.
# * Optimization is not compositional.

# %% [markdown]
# ## Exercise 3
#
# What properly weighted samplers are.

# %% [markdown]
# ## Exercise 4
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
# ## Exercise 5
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
