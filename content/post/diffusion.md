---
title: Diffusion Models
date: 2023-12-30
share: true
tags:
  - diffusion
  - generative
description: Derivation of diffusion models.
author: Yijiang Li
categories:
  - diffusion
width: -120
---
# Diffusion Model  
  
Diffusion models are inspired by non-equilibrium thermodynamics. They define a Markov chain of diffusion steps to gradually add random noise to the data and then learn to reverse the diffusion process to construct desired data samples from the noise. Unlike VAE or flow models, diffusion models are learned with a fixed $p(z \mid x)$ and the latent variable has high dimensionality (same as the original data).  
  
## Forward Process  
  
Forward process adds noise to the sampled data graduately and finally turns it into noise equivalent to one sampled from an isotropic Gaussian distribution. Formally, given data point $x_0 \sim q(x)$, the forward process produce a sequence of noisy samples $x_1, \cdots, x_T$ in the form of markov chain. $x_t$ is generated by sampling from a Gaussian with mean of $\sqrt{1-\beta_t} x_{t-1}$ and variance $\sigma^2=\beta_t I$. $\beta_t$ is the step size controlling how fast it will turn into a zero-mean Gaussian noise.  
  
$$  
\begin{equation}  
  q(x_t \mid x_{t-1}) = N(\sqrt{1-\beta_t} x_{t-1}, \beta_t I)  
\end{equation}  
$$  
  
The joint distribution of all the noisy samples given $x_0$ is $q ( x_1, \cdots, q_T \mid x_0 )$ (denoted as $q(x_{1:T} \mid x_0)$):  
  
$$  
\begin{equation}  
q(x_{1:T} \mid x_0)=\prod_{t=1}^T q(x_t \mid x_{t-1})  
\end{equation}  
$$  
  
The direct and simple solution to obtain the $t^{th}$ noisy sample $x_t$ is to iteratively sample according to $q(x_t\mid x_{t-1})$. A nice property of Gaussian is that this process can be written in a closed-form. That's to say, we can sample $x_t$ at any arbitrary time step $t$ using reparameterization trick. Let $\alpha_t = 1 - \beta_t$, we have  
  
$$  
\begin{equation}  
x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1 - \alpha_t}\epsilon_t  
\end{equation}  
$$  
  
Here, $\epsilon_t \sim N(0,1)$ is sampled from a standard Gaussian.  
We can further re-write $x_{t-1}$ in the same form:  
  
$$  
\begin{align}  
x_t &= \sqrt{\alpha_t} x_{t-1} + \sqrt{1 - \alpha_t}\epsilon_t\\  
&= \sqrt{\alpha_t} (\sqrt{\alpha_{t-1}} x_{t-2} + \sqrt{1 - \alpha_{t-1}}\epsilon_{t-1}) + \sqrt{1 - \alpha_t}\epsilon_t\\  
&= \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \sqrt{\alpha_t(1 - \alpha_{t-1})}\epsilon_{t-1} + \sqrt{1 - \alpha_t}\epsilon_t  
\end{align}  
$$  
  
The second and the third term is the sum of two Gaussian with mean $\mu=0$ and variance $\sigma_t^2 = 1-\alpha_t$ and $\sigma_{t-1}^2=\alpha_t(1 - \alpha_{t-1})$. The sum of two independent Gaussian $N(\mu_1, \sigma^2_1)$, $N(\mu_2, \sigma^2_2)$ is also an Gaussian $N(\mu_1+\mu_2, \sigma^2_1 + \sigma^2_2)$.   
  
$$  
\begin{equation}  
\sigma^2_1 + \sigma^2_2=\alpha_t(1 - \alpha_{t-1})+1-\alpha_t=1-\alpha_t \alpha_{t-1}  
\end{equation}  
$$  
  
 Thus, the sum of the second and the third term can be viewed as a new noise sampled from $N(0, 1-\alpha_t \alpha_{t-1})$. Using the reparameterization trick, we write it as $\sqrt{1 - \alpha_t \alpha_{t-1}}\hat \epsilon_{t-2}$. We can further expand the $x_t$ to $x_0$:  
  
$$  
\begin{align}  
x_t &= \sqrt{\alpha_t} x_{t-1} + \sqrt{1 - \alpha_t}\epsilon_t\\  
&= \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \sqrt{1 - \alpha_t \alpha_{t-1}}\hat \epsilon_{t-2}\\  
&\cdots \notag \\  
&= \sqrt{\alpha_t \alpha_{t-1} \cdots \alpha_1} x_0 + \sqrt{1 - \alpha_t \alpha_{t-1} \cdots \alpha_1}\epsilon  
\end{align}  
$$  
  
Here all $\epsilon_t$, $\hat\epsilon_{t-2}$ and $\epsilon$ are noise sampled form standard Gaussian $N(0,1)$. Thus, we can sample $x_t$ at any arbitrary time step t in a closed form:  
  
$$  
\begin{equation}  
  q(x_t\mid x_0) = N(\sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t)I)  
\end{equation}  
$$  
  
where $\alpha_t = 1-\beta_t$ and $\bar\alpha_t=\alpha_t \alpha_{t-1} \cdots \alpha_1=\prod_{i=1}^t \alpha_i$.  
  
## Reverse Process  
  
If we can reverse the process of each diffusion step $q(x_{t+1}\mid x_t)$ with $q(x_t\mid x_{t+1})$. Bayes' Rule can be used to get $q(x_{t+1}\mid x_t)$, but $q(x_t)$ is intractable since $x_t$ is usually in high dimensional space (same dimension as input image).  
  
$$  
\begin{equation}  
q(x_t\mid x_{t+1}) = \frac{q(x_{t+1}\mid x_t) q(x_t)}{q(x_{t+1})}  
\end{equation}  
$$  
  
Diffusion model thus use a $p(x_t\mid x_{t+1})$ to approximate $q(x_t\mid x_{t+1})$. $p(x_T)$ equals to $q(x_T)$ being the final noise distribution, which is a standard Gaussian. The joint disrtibution $p(x_0, x_1, \cdots, x_T) = p(x_{0:T})$ can be formulated by probability of the reverse trajectory:  
  
$$  
\begin{align}  
p(x_T) &= N(0,1) \\  
p(x_{0:T}) &= p(x_T) \prod_{t=1}^Tp(x_{t-1}\mid x_t)  
\end{align}  
$$  
  
The goal of optimization is to maximize the (log-) probability of $p(x_0)$  
  
$$  
\begin{align}  
p(x_0) &= \int p(x_{0:T}) dx_1\cdots dx_T\\  
&= \int p(x_T) \prod_{t=1}^Tp(x_{t-1}\mid x_t) dx_1\cdots dx_T\\  
&= \int p(x_T) \prod_{t=1}^Tp(x_{t-1}\mid x_t) \frac{q(x_{1:T}\mid x_0)}{q(x_{1:T}\mid x_0)}dx_1\cdots dx_T \quad \textcolor{blue}{\text{(importance sampling)}}\\  
&= \int p(x_T)  q(x_{1:T}\mid x_0) \frac{\prod_{t=1}^T p(x_{t-1}\mid x_t)}{q(x_{1:T}\mid x_0)}dx_1\cdots dx_T \quad \textcolor{blue}{\text{(importance sampling)}}\\  
&= \int p(x_T) q(x_{1:T}\mid x_0) \frac{\prod_{t=1}^Tp(x_{t-1}\mid x_t)}{\prod_{t=1}^T q(x_t\mid x_{t-1})}dx_1\cdots dx_T\\  
&= \int p(x_T) q(x_{1:T}\mid x_0) \prod_{t=1}^T\frac{p(x_{t-1}\mid x_t)}{q(x_t\mid x_{t-1})}dx_1\cdots dx_T  
\end{align}  
$$  
  
Training amounts to maximizing the expectation of log likelihood $E_{x_0 \sim q(x_0)}[\log p(x_0)]$. From the above, we can see there is intergral under logarithm in $p(x_0)$, which is problematic to optimize. We use Jensen’s inequality to put the logarithm under intergral:  
  
$$  
\begin{align}  
E_{x_0 \sim q(x_0)}[\log p(x_0)] &= E_{x_0 \sim q(x_0)}[  
    \notag  \\  
    \log(\int p(x_T) & q(x_{1:T}\mid x_0) \frac{\prod_{t=1}^Tp(x_{t-1}\mid x_t)}{\prod_{t=1}^T q(x_t\mid x_{t-1})}dx_1\cdots dx_T)]\\  
&=\int q(x_0)dx_0 \notag  \\  
\log(\int p(x_T) & q(x_{1:T}\mid x_0) \frac{\prod_{t=1}^Tp(x_{t-1}\mid x_t)}{\prod_{t=1}^T q(x_t\mid x_{t-1})}dx_1\cdots dx_T) \quad \textcolor{blue}{\text{(re-write in intergral form)}}\\  
&\ge \int q(x_0) q(x_{1:T}\mid x_0) dx_0 \notag \\  
\log(p(x_T) & \frac{\prod_{t=1}^Tp(x_{t-1}\mid x_t)}{\prod_{t=1}^T q(x_t\mid x_{t-1})})dx_1\cdots dx_T \quad \textcolor{blue}{\text{(Jensen’s inequality)}}\\  
&=\int q(x_{0:T})dx_{0:T}  
\log(p(x_T) \prod_{t=1}^T \frac{p(x_{t-1}\mid x_t)}{q(x_t\mid x_{t-1})}) = K  
\end{align}  
$$  
  
We split the product under logarithm into sum of logarithm and peel off the contribution from $p(x_T)$:  
  
$$  
\begin{align}  
  K &= \int q(x_{0:T})dx_{0:T}  
\log p(x_T) + \int q(x_{0:T})dx_{0:T}  
\log \prod_{t=1}^T \frac{p(x_{t-1}\mid x_t)}{q(x_t\mid x_{t-1})} \\  
&=\int q(x_T) \log p(x_T)dx_T + \sum_{t=1}^T \int q(x_{0:T})dx_{0:T}  
\log \frac{p(x_{t-1}\mid x_t)}{q(x_t\mid x_{t-1})}  
\end{align}  
$$  
  
The first term is essentially entropy of $x_T$ since $p(x_T)=q(x_T)\approx N(0,1)$.  
  
$$  
\begin{equation}  
  K = -H(x_T) + \sum_{t=1}^T \int q(x_{0:T})dx_{0:T}  
\log \frac{p(x_{t-1}\mid x_t)}{q(x_t\mid x_{t-1})}  
\end{equation}  
$$  
  
Then we split the term of $t=1$ out for edge effect. We show later why this is neccessary.  
  
$$  
\begin{align}  
K = -H(x_T) + \sum_{t=2}^T & \int q(x_{0:T})dx_{0:T}  
\log \frac{p(x_{t-1}\mid x_t)}{q(x_t\mid x_{t-1})} + \notag \\  
&\int q(x_{0:T})dx_{0:T}  
\log \frac{p(x_0\mid x_1)}{q(x_1\mid x_0)}\\  
= -H(x_T) + \sum_{t=2}^T & \int q(x_{0:T})dx_{0:T}  
\log \frac{p(x_{t-1}\mid x_t)}{q(x_t\mid x_{t-1}, x_0)} + \notag \\  
&\int q(x_{0:T})dx_{0:T}  
\log \frac{p(x_0\mid x_1)}{q(x_1\mid x_0)} \quad \textcolor{blue}{\text{(q is a Markov process)}}\\  
= -H(x_T) + \sum_{t=2}^T & \int q(x_{0:T})dx_{0:T}  
\log \frac{p(x_{t-1}\mid x_t)q(x_{t-1}\mid x_0)}{q(x_{t-1}\mid x_t, x_0) q(x_t\mid x_0)} + \notag \\  
&\int q(x_{0:T})dx_{0:T}  
\log \frac{p(x_0\mid x_1)}{q(x_1\mid x_0)} \quad \textcolor{blue}{\text{(Bayes' Rule on conditional probability)}}  
\end{align}  
$$  
  
Here, we show why it is neccessary to split the term with $t=1$. If the summation starts from $t=1$, then we will have to perform Bayes' Rule on the first term which ended up with something like the following with a loop ($q(x_1\mid x_0)$ exists on both sides).  
  
$$  
\begin{equation}  
q(x_1\mid x_0) = \frac{q(x_0\mid x_0)}{q(x_0\mid x_1, x_0) q(x_1\mid x_0)}  
\end{equation}  
$$  
  
We take the second term out and split the product under logarithm into sum of logarithm. We find out that the second and third term is essentially conditional entropies and many of them are canceling each other out.  
  
$$  
\begin{align}  
&\sum_{t=2}^T \int q(x_{0:T})dx_{0:T}  
\log \frac{p(x_{t-1}\mid x_t)q(x_{t-1}\mid x_0)}{q(x_{t-1}\mid x_t, x_0) q(x_t\mid x_0)}\\  
&=\sum_{t=2}^T \int q(x_{0:T})dx_{0:T}  
\log \frac{p(x_{t-1}\mid x_t)}{q(x_{t-1}\mid x_t, x_0) } + \sum_{t=2}^T \int q(x_{0:T})dx_{0:T}  
\log q(x_{t-1}\mid x_0) \notag \\  
&\qquad \qquad \qquad \qquad \qquad \qquad \qquad - \sum_{t=2}^T \int q(x_{0:T})dx_{0:T}  
\log q(x_t\mid x_0) \\  
&=\sum_{t=2}^T \int q(x_{0:T})dx_{0:T}  
\log \frac{p(x_{t-1}\mid x_t)}{q(x_{t-1}\mid x_t, x_0) }  + \int q(x_{0:T})dx_{0:T} \log \frac{q(x_1\mid x_0)}{q(x_T\mid x_0)}\\  
\end{align}  
$$  
  
Now, we show how the first term can be converted into a KL divergence:  
  
$$  
\begin{align}  
&\sum_{t=2}^T \int q(x_{0:T})dx_{0:T}  
\log \frac{p(x_{t-1}\mid x_t)}{q(x_{t-1}\mid x_t, x_0)}\\  
&=\sum_{t=2}^T \int q(x_0, x_t, x_{t-1})dx_0 dx_t dx_{t-1}  
\log \frac{p(x_{t-1}\mid x_t)}{q(x_{t-1}\mid x_t, x_0)} \\  
&=\sum_{t=2}^T \int q(x_{t-1}\mid x_0, x_t) q(x_0, x_t) dx_0 dx_t dx_{t-1}  
\log \frac{p(x_{t-1}\mid x_t)}{q(x_{t-1}\mid x_t, x_0)} \\  
&=\sum_{t=2}^T \int q(x_0, x_t) dx_0 dx_t\int q(x_{t-1}\mid x_0, x_t)  
\log \frac{p(x_{t-1}\mid x_t)}{q(x_{t-1}\mid x_t, x_0)} dx_{t-1}\\  
&=-\sum_{t=2}^T \int q(x_0, x_t) dx_0 dx_t D_{KL}(q(x_{t-1}\mid x_0, x_t)\mid \mid p(x_{t-1}\mid x_t))  
\end{align}  
$$  
  
Now we have all the components to compute the optimization objective where we replace $-H(x_T)$ with intergral form ($-H(x_T)=\int q(x_{0:T})dx_{0:T}  
\log p(x_T)$):  
  
$$  
\begin{align}  
  K &= \int q(x_{0:T})dx_{0:T}  
\log p(x_T) \notag \\  
&- \sum_{t=2}^T \int q(x_0, x_t) dx_0 dx_t D_{KL}(q(x_{t-1}\mid x_0, x_t)\mid \mid p(x_{t-1}\mid x_t)) \notag \\  
&+ \int q(x_{0:T})dx_{0:T} \log \frac{q(x_1\mid x_0)}{q(x_T\mid x_0)} + \int q(x_{0:T})dx_{0:T} \log \frac{p(x_0\mid x_1)}{q(x_1\mid x_0)}  
\end{align}  
$$  
  
Then we first merge the last two term canceling the $q(x_1\mid x_0)$ and then merge the denominator $q(x_T\mid x_0)$ with the first term:  
  
$$  
\begin{align}  
K &= \int q(x_{0:T})dx_{0:T}  
\log \frac{p(x_T)}{q(x_T\mid x_0)} \notag \\  
&- \sum_{t=2}^T \int q(x_0, x_t) dx_0 dx_t D_{KL}(q(x_{t-1}\mid x_0, x_t)\mid \mid p(x_{t-1}\mid x_t)) \notag \\  
&+ \int q(x_{0:T})dx_{0:T} \log p(x_0\mid x_1)  
\end{align}  
$$  
  
Then we re-write the first term into KL Divergence:  
  
$$  
\begin{align}  
K &= - \int q(x_0)dx_0  D_{KL}(q(x_T\mid x_0)\mid \mid p(x_T)) \notag \\  
&-\sum_{t=2}^T \int q(x_0, x_t) dx_0 dx_t D_{KL}(q(x_{t-1}\mid x_0, x_t)\mid \mid p(x_{t-1}\mid x_t)) + \int q(x_{0:T})dx_{0:T} \log p(x_0\mid x_1)  
\end{align}  
$$  
  
Now take a look at each term. The first term  
  
$$  
\begin{equation}  
\int q(x_0)dx_0 D_{KL}(q(x_T\mid x_0)\mid \mid p(x_T))    
\end{equation}  
$$  
  
is essentially a small constant since $q(x_T\mid x_0)$ converges to $p(x_T)$ which is a standard Gaussian.  
The third term is tractable since $p(x_0\mid x_1) = N(\mu_\theta(x_1, 1), \beta_1)$ is in the form of Gaussian distribution:  
  
$$  
\begin{align}  
p(x_0\mid  x_1) = \prod_{i=1}^D \int^{\delta^+(x_0^{i})}_{\delta^-(x_0^{i})} N(\mu_\theta^i(x_1, 1), \beta_1) \notag \\  
\delta^+(x) = \left\{  
\begin{aligned}  
& = x+\frac{1}{255} \quad \text{if x < 1}\\  
& = \infty \quad \text{if x = 1}\\  
\end{aligned}  
\right.  
\delta^-(x) = \left\{  
\begin{aligned}  
& = x-\frac{1}{255} \quad \text{if x > -1}\\  
& = -\infty \quad \text{if x = -1}\\  
\end{aligned}  
\right.  
\end{align}  
$$  
  
A easier and more computational simple way is to get rid of the noise at $q(x_1\mid  x_0)$.  
The second term contributes to the main loss of the training.  
  
$$  
\begin{equation}  
-\sum_{t=2}^T \int q(x_0, x_t) dx_0 dx_t D_{KL}(q(x_{t-1}\mid x_0, x_t)\mid \mid p(x_{t-1}\mid x_t))  
\end{equation}  
$$  
  
Both $q(x_{t-1}\mid x_0, x_t)=N(\tilde \mu_t(x_t, x_0), \tilde \beta_t I)$ and $p(x_{t-1}\mid x_t)=N(\mu_\theta(x_t, t), \Sigma_\theta(x_t, t) I)$ are Gaussian. We recall at the beginning that $q(x_{t-1}\mid x_t)$ is intractable, but if it's conditional on $x_0$, then it's tractable and we can give a closed form solution via Bayes' rule:  
  
$$  
\begin{equation}  
q(x_{t-1}\mid x_{t}, x_0) = \frac{q(x_{t}\mid x_{t-1}, x_0) q(x_{t-1}\mid x_0)}{q(x_{t}\mid x_0)}  
\end{equation}  
$$  
  
All $q(x_{t}\mid x_{t-1}, x_0)$, $q(x_{t-1}\mid x_0)$ and $q(x_{t}\mid x_0)$ has analytical forms (desity function of Gaussian distribution). $q(x_{t+1}\mid x_t, x_0) = q(x_{t+1}\mid x_t)$ since $q$ is a markov process.  
  
$$  
\begin{align}  
q(x_{t}\mid x_{t-1}, x_0) &= \frac{1}{\sqrt{2\pi \beta_t}} \exp \big(-\frac{(x_t - \sqrt{1-\beta_t}x_{t-1})^2}{2\beta_t} \big)\\  
q(x_t\mid x_0)&=\frac{1}{\sqrt{2\pi (1-\bar \alpha_t)}}\exp \big(-\frac{(x_t - \sqrt{\bar \alpha_t}x_0)^2}{2(1-\bar \alpha_t)}\big)\\  
q(x_{t-1}\mid x_0)&=\frac{1}{\sqrt{2\pi (1-\bar \alpha_{t-1})}}\exp \big(-\frac{(x_{t-1} - \sqrt{\bar \alpha_{t-1}}x_0)^2}{2(1-\bar \alpha_{t-1})} \big)  
\end{align}  
$$  
  
Then, we have $q(x_{t-1}\mid x_{t}, x_0)$. Essentially, we want to get $\tilde \mu_t(x_t, x_0)$ and $\tilde \beta_t$, which is the mean and variance of Gaussian distribution $q(x_{t-1} \vert x_t, x_0)$. $q(x_{t-1} \vert x_t, x_0)$ shares the similar density function as any Gaussian:  
  
$$  
\begin{equation}  
q(x_{t-1}\mid x_{t}, x_0) = \frac{1}{\sqrt{2\pi \tilde \beta_t}} \exp \big( -\frac{(x_{t-1} - \tilde \mu_t(x_t, x_0))^2}{2 \tilde \beta_t^2}  \big)  
\end{equation}  
$$  
  
By taking the $q(x_{t}\mid x_{t-1}, x_0)$, $q(x_{t-1}\mid x_0)$ and $q(x_{t}\mid x_0)$ to the Bayes' rule, we re-write it as following:  
  
$$  
\begin{align}  
q(x_{t-1} \vert x_t, x_0)  
&= \frac{q(x_t \vert x_{t-1}, x_0) q(x_{t-1} \vert x_0) }{ q(x_t \vert x_0) } \\  
&\propto \exp \Big(-\frac{1}{2} \big(\frac{(x_t - \sqrt{\alpha_t} x_{t-1})^2}{\beta_t} + \frac{(x_{t-1} - \sqrt{\bar{\alpha}_{t-1}} x_0)^2}{1-\bar{\alpha}_{t-1}} - \frac{(x_t - \sqrt{\bar{\alpha}_t} x_0)^2}{1-\bar{\alpha}_t} \big) \Big) \\  
&= \exp \Big(-\frac{1}{2} \big(\frac{x_t^2 - 2\sqrt{\alpha_t} x_t \color{blue}{x_{t-1}} \color{black}{+ \alpha_t} \color{red}{x_{t-1}^2} }{\beta_t} + \frac{ \color{red}{x_{t-1}^2} \color{black}{- 2 \sqrt{\bar{\alpha}_{t-1}} x_0} \color{blue}{x_{t-1}} \color{black}{+ \bar{\alpha}_{t-1} x_0^2}  }{1-\bar{\alpha}_{t-1}} - \frac{(x_t - \sqrt{\bar{\alpha}_t} x_0)^2}{1-\bar{\alpha}_t} \big) \Big) \\  
&= \exp\Big( -\frac{1}{2} \big( \color{red}{(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}})} x_{t-1}^2 - \color{blue}{(\frac{2\sqrt{\alpha_t}}{\beta_t} x_t + \frac{2\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}} x_0)} x_{t-1} \color{black}{ + C(x_t, x_0) \big) \Big)}  
\end{align}  
$$  
  
We can see,  
  
$$  
\begin{align}  
\tilde{\beta}_t  
&= 1/(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}})  
= 1/(\frac{\alpha_t - \bar{\alpha}_t + \beta_t}{\beta_t(1 - \bar{\alpha}_{t-1})})  
= \color{green}{\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t} \\  
\tilde{\mu}_t (x_t, x_0)  
&= (\frac{\sqrt{\alpha_t}}{\beta_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1} }}{1 - \bar{\alpha}_{t-1}} x_0)/(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}}) \\  
&= (\frac{\sqrt{\alpha_t}}{\beta_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1} }}{1 - \bar{\alpha}_{t-1}} x_0) \color{green}{\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t} \\  
&= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} x_0\\  
\end{align}  
$$  
  
We can replace $x_0$ with $x_t = \sqrt{\bar \alpha_t} x_0 + \sqrt{1 - \bar \alpha_t}\epsilon$,  
  
$$  
\begin{align}  
x_0 &= \frac{x_t - \sqrt{1 - \bar \alpha_t}\epsilon}{\sqrt{\bar \alpha_t}}\\  
\tilde{\mu}_t (x_t, x_0)&=\frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} \frac{x_t - \sqrt{1 - \bar \alpha_t}\epsilon}{\sqrt{\bar \alpha_t}}\\  
&=\frac{1}{\sqrt{\bar \alpha_t}}(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar \alpha_t}}\epsilon)  
\end{align}  
$$  
  
Now, we derive the final loss of the second term and since we use a fixed $\beta_t$ meaning that $\Sigma_\theta(x_t, t)$ is the same as $\beta_t$.  
  
$$  
\begin{equation}  
L = E_{x_0 \sim q(x_0), \epsilon \sim N(0, 1)}\big[\frac{1}{2\mid \mid \Sigma_\theta(x_t, t)\mid \mid ^2_2} \mid \mid \mu_\theta(x_t, t) - \tilde{\mu}_t (x_t, x_0) \mid \mid ^2\big]    
\end{equation}  
$$  
  
Since we have $x_t$ during training, we re-write $\mu_\theta(x_t, t)$ in the same form as $\tilde{\mu}_t (x_t, x_0)$ and the objective now becomes predicting the noise $\epsilon$:  
  
$$  
\begin{equation}  
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\bar \alpha_t}}(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar \alpha_t}}\epsilon_\theta(x_t, t))  
\end{equation}  
$$  
  
$$  
\begin{align}  
L &= E_{x_0 \sim q(x_0), \epsilon \sim N(0, 1)} \big[ \frac{1}{2\mid \mid \Sigma_\theta(x_t, t)\mid \mid ^2_2} \mid \mid \frac{1}{\sqrt{\bar \alpha_t}}(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar \alpha_t}}\epsilon_\theta(x_t, t)) - \frac{1}{\sqrt{\bar \alpha_t}}(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar \alpha_t}}\epsilon)\mid \mid ^2 \big]\\  
&=E_{x_0 \sim q(x_0), \epsilon \sim N(0, 1)} \big[ \frac{(1-\alpha_t)^2}{2\alpha_t(1-\bar\alpha_t)\mid \mid \Sigma_\theta(x_t, t)\mid \mid ^2_2} \mid \mid \epsilon_\theta(x_t, t) - \epsilon\mid \mid ^2 \big]\\  
&=E_{x_0 \sim q(x_0), \epsilon \sim N(0, 1)} \big[ \frac{(1-\alpha_t)^2}{2\alpha_t(1-\bar\alpha_t)\mid \mid \Sigma_\theta(x_t, t)\mid \mid ^2_2} \mid \mid \epsilon_\theta(\sqrt{\bar \alpha_t} x_0 + \sqrt{1 - \bar \alpha_t}\epsilon, t) - \epsilon\mid \mid ^2 \big]  
\end{align}  
$$  
  
DDPM gets rid of the constant as it is empirically better.  
  
$$  
\begin{align}  
L &= E_{x_0 \sim q(x_0), \epsilon \sim N(0, 1)} \mid \mid \epsilon_\theta(x_t, t) - \epsilon\mid \mid ^2 \big]\\  
&=E_{x_0 \sim q(x_0), \epsilon \sim N(0, 1)} \mid \mid \epsilon_\theta(\sqrt{\bar \alpha_t} x_0 + \sqrt{1 - \bar \alpha_t}\epsilon, t) - \epsilon\mid \mid ^2 \big]  
\end{align}  
$$