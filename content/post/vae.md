---
title: Variational Autoencoder
date: 2023-12-30
share: true
tags:
  - generative
  - vae
description: Intro to variational autoencoder and derivation of ELBO.
author: Yijiang Li
categories:
  - generative
width: -120
---
  
  
# Variational Autoencoder  
Variational Autoencoder (VAE) is a type of generative model that define an explicit density function for $p(x)$. The variational part of VAE denotes that it uses Bayes' Rule to derive an explicit $p(x)$ and by introducing a variational, VAE provides an evidence lower bound to maximize.  
  
## Intuition  
  
### Mathematical perspective  
The goal of generative models is to maximize the log liklihood of $\log p_\theta(x)$.  
  
$$  
\theta^* = \operatorname*{argmax}_\theta E_{x\sim p_{\theta^*}(x)}[\log p_\theta (x)]$$  
  
The problem becomes how to model $\log p_\theta(x)$. VAE assumes that there exists a latent variable $z$ following some prior distribution $p(x)$ that can decompose the data likelihood as the marginal distribution of the conditional likelihood w.r.t this latent variable.  
  
$$  
p_\theta (x) = \int_z p_\theta (x|z) p(z) dz  
$$  
  
Following this assumption, the data generation process now becomes:  
  
1. Sample $z$ from $p(z)$. Prior $p(z)$ is a simple distribution we assign for easy compuation, e.g. Gaussian.  
2. Generate $x$ with $z$ and $p_\theta(x|z)$. $z$ is sampled from $p(x)$ and $p_\theta(x|z)$ is represented with the neural network.  
  
Despite we have both $p(z)$ (pre-defined Gaussian) and $p_\theta(x|z)$ (neural network), the intergral over $z$ is intractable as we cannot compute $p_\theta(x|z)$ for every $z$ in the intergral. This is because, despite $p(z)$ is usually low dimensional, $p_\theta(x|z)$ is often times complex, making it impossible to integrate with an analytical solution. One way is to approximate $p_\theta(x|z)$ is Monte Carlo estimation, where we sample over $p(z)$ to estimate $p_\theta(x|z)$:  
  
$$  
p_\theta (x) = \int_z p_\theta (x|z) p(z) dz = E_{z\sim p(z)}[p_\theta (x|z)]\approx \frac{1}{m} \sum_{i=1}^m p(x|z_i)\\  
\log p_\theta (x) \approx \log (\frac{1}{n} \sum_{i=1}^n p(x|z_i))  
$$  
  
However, since space of $z$ is too large, Monte Carlo is inefficient as most samples contributes to zero probability. Random sampling over $z$ also brings a high variance to the estimation of $p_\theta(x)$. Another approach to approximate $p_\theta(x)$ is to leverage Bayes' Rule:  
  
$$  
p_\theta(z|x) = \frac{p_\theta(x|z) p(z)}{p_\theta(x)}  
$$  
  
Once we have $p_\theta(z|x)$, we can approximate by averaging $p_\theta(z|x)$ over training set $X$.  
  
$$  
p_\theta (x) = \int_z p_\theta (z|x) p(x) dx = E_{x\sim p_{\theta^*}(x)}[p_\theta (z|x)] \approx \frac{1}{m}\sum_{i=1}^m p_\theta(z|x_i)  
$$  
  
However, the denominator $p_\theta(x)$ is also intractable (core problem).  
To solve this intractability, VAE introduce a variation $q_\phi(z|x)$ to approximate $p_\theta (z|x)$. $q_\phi(z|x)$ is also represented with a neural network and jointly optimize both $p_\theta(x|z)$ and $q_\phi(z|x)$.  
  
  
  
### Auto-encoder perspective  
  
## Optimization  
Given the true distribution of data $p_{\theta^*}(x)$ with parameter $\theta^*$, the goal of VAE, as in any generative model, is to find the optimal $\theta^*$, which maximize the log likelihood of $p_\theta (x)$ with $x$ sampled from $p_{\theta^*}(x)$:  
  
$$  
\theta^* = \operatorname*{argmax}_\theta E_{x\sim p_{\theta^*}(x)}[\log p_\theta (x)]\\  
$$  
  
VAE takes the form of an autoencoder, but instead of mapping the input into a fixed vector, we want to map it into a latent distribution $p(z)$. However, since both intergral and posterior is intractable, VAE introduce $q_\phi(z|x)$ to approximate $p_\theta(z|x)$. Note that once we have $p_\theta(x|z)$ and $p(z)$, $p_\theta(z|x)$ is fixed given by Bayes' Rule while $q_\phi(z|x)$ is independent of $p_\theta(x|z)$ with a different parameter $\phi$. What we want to solve now becomes to find a $\theta$ and $\phi$ to maximize the likelihood of $p_\theta(x)=\int_{z\sim p(z)} p_\theta(x, z) dz = \int_{z\sim p(z)} p_\theta(x|z) p(z) dz$.  
  
We first derive the evidence lower bound (ELBO) for $\log p_\theta (x)$ which we optimize to maximize the log likelihood. We show three popular derivation as Derivation 1, Derivation 2 and Derivation 3. Notice that final objective is the expectation of $\log p_\theta (x)$ with respect to true distribution $x \sim p_{\theta^*}(x)$. Empirically, this is approximated with average over all training samples in $X$, which we show in Section "Loss derived from ELBO".  
### ELBO: Derivation 1  
Derivation 1 derectly derives the ELBO using anneal sampling and Jensen's inequality. We resmaple $p_\theta(x)$ with $q_\phi(z|x)$ (equals to multiply by constant 1).  
  
$$  
\begin{aligned}  
p_\theta(x) &= \int_{z\sim p(z)} p_\theta(x|z) p(z) dz\\  
&=\int_{z\sim p(z)} q_\phi(z|x) \frac{p_\theta(x|z) p(z)}{q_\phi(z|x)} dz \quad \textcolor{blue}{\text{(importance sampling)}} \\  
&=E_{z\sim q_\phi(z|x)}[\frac{p_\theta(x|z) p(z)}{q_\phi(z|x)}] \quad \textcolor{blue}{\text{(re-write in expectation)}}  
\end{aligned}$$  
  
The log liklihood of $p_\theta(x)$ now becomes:  
  
$$  
\log p_\theta(x) = \log (E_{z\sim q_\phi(z|x)}[\frac{p_\theta(x|z) p(z)}{q_\phi(z|x)}])  
$$  
  
The expectation (or intergral) under log is unhandy. By using Jensen's inequality, we can put the log under the expectation (or intergral):  
  
$$  
\begin{aligned}  
\log p_\theta(x) &= \log (E_{z\sim q_\phi(z|x)}[\frac{p_\theta(x|z) p(z)}{q_\phi(z|x)}]) \\  
&\ge E_{z\sim q_\phi(z|x)}[\log \frac{p_\theta(x|z) p(z)}{q_\phi(z|x)})] \quad \textcolor{blue}{\text{(Jensen's Inequality)}}\\  
&= \int_z q_\phi(z|x) \log \frac{p_\theta(x|z) p(z)}{q_\phi(z|x)} dz \quad \textcolor{blue}{\text{(re-write in intergral)}}\\  
&= ELBO  
\end{aligned}  
$$  
  
### ELBO: Derivation 2  
We refer to stanford [CS231N](http://cs231n.stanford.edu/slides/2022/lecture_13_jiajun.pdf) for this derivation.  
  
$\log p_\theta(x)$ can be re-written as expectation with respect to $q_\phi(z|x)$ because $p_\theta(x)$ does not contain $z$:  
  
$$  
\begin{aligned}  
\log p_\theta(x) &= \int_z q_\phi(z|x) \log p_\theta(x) dz \quad \textcolor{blue}{\text{($p_\theta(x)$ does not depend on $z$)}}\\  
&= \int_z q_\phi(z|x) \log \frac{p_\theta(x|z) p(z)}{p_\theta(z|x)} dz \quad \textcolor{blue}{\text{(Bayes' Rule)}}\\  
&=\int_z q_\phi(z|x) \log (\frac{p_\theta(x|z) p(z)}{q_\phi(z|x)} \frac{q_\phi(z|x)}{p_\theta(z|x)}) dz \quad \textcolor{blue}{\text{(multiply \ by \ 1)}}\\  
&=\int_z q_\phi(z|x) \log \frac{p_\theta(x|z) p(z)}{q_\phi(z|x)} dz + \int_z q_\phi(z|x) \log \frac{q_\phi(z|x)}{p_\theta(z|x)} dz      
\end{aligned}  
$$  
  
The first term, as termed in variational inference, is the $ELBO$ while the second term is the KL Divergence between $q_\phi(z|x)$ and $p_\theta(z|x)$.  
  
$$  
\log p_\theta(x) = ELBO + D_{KL}(q_\phi(z|x) || p_\theta(z|x))  
$$  
  
Since KL Divergence is non-negative. Thus, we have the lower boound of $\log p_\theta(x)$  
$$\log p_\theta(x) \ge ELBO = \int_z q_\phi(z|x) \log \frac{p_\theta(x|z) p(z)}{q_\phi(z|x)} dz$$  
  
Another way to look at the objective is to re-write the equations:  
  
$$  
\log p_\theta(x) - D_{KL}(q_\phi(z|x) || p_\theta(z|x)) = ELBO  
$$  
  
By maximizing the RHS, we maximize the log probability of $p_\theta(x)$ and meanwhile minimize the discrapance of $q_\phi(z|x)$ and $p_\theta(z|x)$. The KL can be viewed as a regularization that constrains $q_\phi(z|x)$.  
  
### ELBO: Derivation 3  
We refer to [Lil's log](https://lilianweng.github.io/posts/2018-08-12-vae/) for the third derivation. This derivation comes from the intuition that we want to minimize the discrapency between $q_\phi(z|x)$ and $p_\theta(z|x)$, which is also used in [variational_inference]({{< relref "variational_inference.md" >}}).  
  
$$  
\begin{aligned}  
D_{KL}&(q_\phi(z|x) || p_\theta(z|x)) = \int_z q_\phi(z|x) \log \frac{q_\phi(z|x)}{p_\theta(z|x)} dz \\  
&= \int_z q_\phi(z|x) \log \frac{q_\phi(z|x)p_\theta(x)}{p_\theta(x|z)p(z)} dz \quad \textcolor{blue}{\text{(Bayes' Rule)}}\\  
&= \int_z q_\phi(z|x) \log p_\theta(x) dz + \int_z q_\phi(z|x) \log \frac{q_\phi(z|x)}{p_\theta(x|z)p(z)} dz\\  
&= \log p_\theta(x) + \int_z q_\phi(z|x) \log \frac{q_\phi(z|x)}{p_\theta(x|z)p(z)} dz\quad \textcolor{blue}{\text{($p_\theta(x)$ does not depend on $z$)}}\\  
&= \log p_\theta(x) + \int_z q_\phi(z|x) \log \frac{q_\phi(z|x)}{p(z)} dz - \int_z q_\phi(z|x) \log p_\theta(z|x) dz \\  
&=\log p_\theta(x) + D_{KL}(q_\phi(z|x)||p(z)) - E_{z\sim q_\phi(z|x)}[\log p_\theta(z|x)]\\  
\end{aligned}  
$$  
  
Once rearrange the left and right hand side of the equation, we have:  
  
$$  
\log p_\theta(x) - D_{KL}(q_\phi(z|x) || p_\theta(z|x)) = \\  
E_{z\sim q_\phi(z|x)}[\log p_\theta(z|x)] - D_{KL}(q_\phi(z|x)||p(z))  
$$  
  
The LHS of the equation is exactly what we want to maximize when learning the true distributions: we want to maximize the (log-)likelihood of generating real data (that is $\log p_\theta(x)$) and also minimize the difference between the real and estimated posterior distributions (the term $D_{KL}$ works like a regularizer). The RHS is $ELBO$. Since the KL Divergence $D_{KL}(q_\phi(z|x) || p_\theta(z|x))$ is non-negative:  
  
$$  
\log p_\theta(x) \ge E_{z\sim q_\phi(z|x)}[\log p_\theta(z|x)] - D_{KL}(q_\phi(z|x)||p(z))  
$$  
  
  
### Loss derived from ELBO  
Now, we take a deeper look at $ELBO$ in VAE:  
  
$$  
\int_z q_\phi(z|x) \log \frac{p_\theta(x|z) p(z)}{q_\phi(z|x)} dz\\  
= \int_z q_\phi(z|x) \log p_\theta(x|z) dz + \int_z q_\phi(z|x) \log \frac{p(z)}{q_\phi(z|x)} dz\\  
= E_{z\sim q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x)||p(z))  
$$  
  
The first term (reconstruction loss) is to maximize the log probability of $p(x|z)$ given $z\sim q_\phi(z|x)$ while the second term (regularization loss) is to reguarlize the $q_\phi(z|x)$ making it close to prior $p(z)$. The first term is typically a MSELoss in the form of $||x - g(z')||_2$ where $z' = q_\phi(z|x)(x)$ and $g$ is the generator that represents $p_\theta(x|z)$. For the second term we usually assume that $z$ follows a standard normal distribution, i.e. $p(z) = N(0, 1)$, and we also optimize $q_\phi(z|x)$ to be a normal distribution, which means that $q_\phi(z|x)$ is represented by a function $f$ that predicts the mean and variance of $q_\phi(z|x)$. Under this assumption, we can write down the pdf of both $p(z)$ and $q_\phi(z|x)$ (in 2D case):  
  
$$  
p(z) = \frac{1}{\sqrt{2\pi\sigma_p^2}}exp(-\frac{(z - \mu_p)^2}{2\sigma_p^2})=\frac{1}{\sqrt{2\pi}}e^{-\frac{z^2}{2}}\\  
q_\phi(z|x) = \frac{1}{\sqrt{2\pi\sigma_\phi^2}}exp(-\frac{(z - \mu_\phi)^2}{2\sigma_\phi^2})  
$$  
  
Thus, the second term can be derived as:  
  
$$  
D_{KL}(q_\phi(z|x) || N(0, 1)) = D_{KL}(N(\mu_\phi, \sigma_\phi) || N(0, 1))\\  
=-\int_z \frac{1}{\sqrt{2\pi\sigma_\phi^2}}exp(-\frac{(z - \mu_\phi)^2}{2\sigma_\phi^2}) \log \frac{\frac{1}{\sqrt{2\pi\sigma_\phi^2}}exp(-\frac{(z - \mu_\phi)^2}{2\sigma_\phi^2})}{\frac{1}{\sqrt{2\pi}}e^{-\frac{z^2}{2}}} dz\\  
= - \int_z \frac{1}{\sqrt{2\pi\sigma_\phi}} exp(-\frac{z - \mu_\phi}{2\sigma_\phi^2}) (-\log \sigma_\phi + \frac{z^2}{2} - \frac{(z - \mu_\phi)^2}{2\sigma_\phi^2})dz\\  
= - E_{q_\phi(z|x)}[-\log \sigma_\phi + \frac{z^2}{2} - \frac{(z - \mu_\phi)^2}{2\sigma_\phi^2}]\\  
= \log \sigma_\phi + \frac{1}{2\sigma_\phi^2}E_{q_\phi(z|x)}[(z - \mu_\phi)^2] - \frac{1}{2}E_{q_\phi(z|x)}[z^2]\\  
=\log \sigma_\phi + \frac{1}{2\sigma_\phi^2}\sigma_\phi^2 - \frac{1}{2}E_{q_\phi(z|x)}[z^2]\\  
=\log \sigma_\phi + \frac{1}{2} - \frac{1}{2}E_{q_\phi(z|x)}[(z - \mu_\phi + \mu_\phi)^2]\\  
=\log \sigma_\phi + \frac{1}{2} - \frac{1}{2}E_{q_\phi(z|x)}[((z - \mu_\phi)^2 + (z - \mu_\phi)\mu_\phi + \mu_\phi^2]\\  
=\log \sigma_\phi + \frac{1}{2} - \frac{1}{2}\sigma_\phi^2 - \frac{1}{2} E_{q_\phi(z|x)}[(z - \mu_\phi)\mu_\phi + \mu_\phi^2]\\  
=\log \sigma_\phi + \frac{1}{2} - \frac{1}{2}\sigma_\phi^2 - \frac{1}{2}\mu_\phi^2 \\  
=\frac{1}{2}(1 + \log \sigma_\phi^2 - \sigma_\phi^2 - \mu_\phi^2)  
$$  
  
Now, we have the final form of $ELBO$ for a single sample $x_i \in X=\{x_1, x_2, \cdots, x_m\}$:  
  
$$  
L_i = ||x_i - g(q_\phi(z|x)(x_i))||_2 + \frac{1}{2}(1 + \log \sigma_{i,\phi}^2 - \sigma_{i,\phi}^2 - \mu_{i,\phi}^2)  
$$  
  
  
We put it back to the expectation and maximize the expectation of $ELBO$ given $x\sim p_{\theta^*}(x)$:  
  
$$  
E_{x\sim p_{\theta^*}(x)}[\log p_\theta (x)] \ge E_{x\sim p_{\theta^*}(x)}[\int_z q_\phi(z|x)\frac{p_\theta(x|z) p(z)}{q_\phi(z|x)} dz]\\  
\theta^*, \phi^* \approx \operatorname*{argmax}_\theta E_{x\sim p_{\theta^*}(x)}[\int_z q_\phi(z|x)\frac{p_\theta(x|z) p(z)}{q_\phi(z|x)} dz]  
$$  
  
We use $\approx$ because sometimes optimal $\theta$ and $\phi$ cannot be obtained due to a loose lower bound. Nevertheless, we manage avoid the intractablity and sucessfully maximize the log likelihood $E_{x\sim p_{\theta^*}(x)}[\log p_\theta (x)]$.   
  
Empirically, we average the ELBO of every training sample $x_i$ in $X$:  
  
$$  
E_{x\sim p_{\theta^*}(x)}[\int_z q_\phi(z|x)\frac{p_\theta(x|z) p(z)}{q_\phi(z|x)} dz] \approx \frac{1}{m} \sum_{i=1}^m L_i\\  
=\frac{1}{m} \sum_{i=1}^m \{||x_i - g(q_\phi(z|x)(x_i))||_2 + \frac{1}{2}(1 + \log \sigma_{i,\phi}^2 - \sigma_{i,\phi}^2 - \mu_{i,\phi}^2)\}\\  
\theta^*, \phi^* \approx \operatorname*{argmax}_\theta \frac{1}{m} \sum_{i=1}^m \{||x_i - g(q_\phi(z|x)(x_i))||_2 \\+\frac{1}{2}(1 + \log \sigma_{i,\phi}^2 - \sigma_{i,\phi}^2 - \mu_{i,\phi}^2)\}  
$$  
  
