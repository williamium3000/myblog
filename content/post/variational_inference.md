---
title: Variational Inference
date: 2023-12-30
share: true
tags:
  - statistics
  - vae
description: variational inference and ELBO.
author: Yijiang Li
categories:
  - statistics
width: -120
---
# Variational inference  
  
## ELBO  
Assume a set of data $X = \{x_1, \cdots, x_m \}$ generated from some known prior $p(z)$ with latent variable (unknown) $Z=\{z_1, \cdots, z_m \}$, we want to know the posteior $p (z|x)$, that is, how to generate z with x.  
From Bayes theorem, we have  
  
$$  
p(z|x) = \frac{p(x|z) p(z)}{p(x)}\\  
=\frac{p(x|z) p(z)}{\int_z p(x|z)p(z)dz}  
$$  
  
However, the problem is that integral on z is usually impossible, especially when z is high dimensional. So, to approximate the posteior $p (z|x)$, we assume a distribution $q_\theta(z)$ parameterized by $\theta$ and want to approximate the true posteior $p (z|x)$ with $q_\theta(z)$ by measuring the KL Divergence between $p (z|x)$ and $q_\theta(z)$.  
  
$$  
q(z, \theta*) = \operatorname*{argmin}_q D_{KL}(q_\theta(z) | p (z|x))  
$$  
  
We rewrite KL in expection form:  
  
$$  
D_{KL}(q_\theta(z) | p (z|x)) = \int q_\theta(z) \log \frac{q_\theta(z)}{p (z|x)} dz \\  
= E_q[\log \frac{q_\theta(z)}{p (z|x)}] \\  
= E_q[\log (q_\theta(z)] - E_q[\log p (z|x)]\\  
= E_q[\log (q_\theta(z)] - E_q[\log (p (z,x))] + E_q[\log p(x)]\\  
=-E_q[\log \frac{p (z,x)}{q_\theta(z)}]+ E_q[\log p(x)]  
$$  
  
We first remove the expectation over $q$ on the second term since it dose not contain $q$ and rewrite the equation as followed:  
  
$$\log p(x) = E_q[\log \frac{p (z,x)}{q_\theta(z)}] + D_{KL}(q | p)\\  
= \int_z q_\theta(z) \log \frac{p (z,x)}{q_\theta(z)} dz + D_{KL}(q | p)$$  
  
Given a fixed $p(x) $, the $\log p(x)$ is fixed and KL Divergence is non-negative (i.e. $D_{KL}(q | p)\ge 0$), thus by maximizing the first term, which we call **evidence lower bound (ELBO)**, we minimize the discrepancy bewteen $q$ and $p$, i.e. $D_{KL}(q | p)$  
  
$$ELBO = \int_z q_\theta(z) \log \frac{p (z,x)}{q_\theta(z)} dz$$  
  
##  Mean field variational inference  
We introduce one assumption (mean field) to optimize the $ELBO$: the variational family factorizes:  
  
$$  
q_\theta(z) = q_\theta(z_1, z_2, \cdots, z_m) = \prod \limits_{i=1}^m q(z_i)  
$$  
  
By factorizing the $q$, we manage to solve for each $z_i$ with the rest as constant (since each component of $z$ is independent with each other). We take $z_j$ as an example and show how each component of $z$ can be solved. We first tackle each of the two terms of $ELBO$ by denoting them as  (1) and (2):  
  
$$ELBO = \int_z q_\theta(z) \log p (z,x)dz - \int_z q_\theta(z) \log q_\theta(z) dz$$  
  
By taking the $ q(z)=\prod \limits_{i=1}^m q(z_i)$ into the first term and only care about $z_j$, we have:  
  
$$  
(1)=\int_z \prod \limits_{i=1}^m q(z_i) \log p (z,x) dz_1 dz_2\cdots dz_m\\  
=\int_{z_j} q(z_j) (\int_{z\neq z_j} \log p (z,x) (\prod \limits_{i=1, i\neq j}^m q(z_i)) dz_1 \cdots dz_{j-1} dz_{j+1}\cdots dz_m) dz_j\\  
=\int_{z_j} q(z_j) (\int_{z\neq z_j} \log p (z,x) (\prod \limits_{i=1, i\neq j}^m q(z_i)dz_i)) dz_j\\  
=\int_{z_j} q(z_j) E_{\prod \limits_{i=1, i\neq j}^m q(z_i)}[\log p (z,x)] dz_j  
$$  
  
We do the same with the second term:  
  
$$  
(2)=\int_z \prod \limits_{i=1}^m q(z_i) \log (\prod \limits_{i=1}^m q(z_i)) dz\\  
=\int_z \prod \limits_{i=1}^m q(z_i) (\sum_{i=1}^m\log q(z_i))  
dz\\  
$$  
  
We take the $z_j$ out among the sum over m:  
  
$$  
\int_z \prod \limits_{i=1}^m q(z_i) \log q(z_j)dz\\  
= \int_{z_1, \cdots, z_m} \prod \limits_{i=1}^m q(z_i) \log q(z_j)dz_1 \cdots dz_m\\  
= \int_{z_j} q(z_j)\log q(z_j) dz_j \int_{z_i}q(z_1)dz_1 \cdots \int_{z_{j-1}}q(z_{j-1})dz_{j-1} \int_{z_{j+1}}q(z_{j+1})dz_{j+1}\cdots \int_{z_m}q(z_m)dz_m\\  
=\int_{z_j} q(z_j)\log q(z_j) dz_j  
$$  
  
Thus, (2) can be deduced to:  
  
$$  
(2)=\sum_{i=1}^m \int_{z_i} q(z_i)\log q(z_i) dz_i  
$$  
  
Since we take the rest of $z_1, \cdots, z_{j-1}, z_{j+1}, \cdots, z_m$ as constant:  
  
$$  
(2)= \int_{z_j} q(z_j)\log q(z_j) dz_j + C  
$$  
  
$ELBO(z_j)$ subtracts (2) from (1):  
  
$$  
ELBO(z_j) = (1) - (2)\\  
= \int_{z_j} q(z_j) E_{\prod \limits_{i=1, i\neq j}^m q(z_i)}[\log p (z,x)] dz_j - \int_{z_j} q(z_j)\log q(z_j) dz_j - C\\  
= \int_{z_j}q(z_j)\log \frac{EXP(E_{\prod \limits_{i=1, i\neq j}^m q(z_i)}[\log p (z,x)])}{q(z_j)}\\  
= -D_{KL}(q(z_j)||EXP(E_{\prod \limits_{i=1, i\neq j}^m q(z_i)}[\log p (z,x)]))  
$$  
  
We denote the exponential term as $\hat p (x, z_j)$ and rewrite the ELBO as:  
  
$$  
ELBO(z_j) = -D_{KL}(q(z_j)||\hat p (x, z_j))\le 0  
$$  
  
When $q(z_j)=\hat p (x, z_j)$, $ELBO(z_j)$ is maximized. Thus,  
  
$$  
q(z_j) = EXP(E_{\prod \limits_{i=1, i\neq j}^m q(z_i)}[\log p (z,x)])  
$$