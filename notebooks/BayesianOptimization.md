---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.10.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Bayesian Optimization

All hail distill: https://distill.pub/2020/bayesian-optimization/

Other resources:
* https://www.cse.wustl.edu/~garnett/cse515t/spring_2015/files/lecture_notes/12.pdf
* https://people.orie.cornell.edu/pfrazier/Presentations/2018.11.INFORMS.tutorial.pdf


# Applied Bayesian Optimization

In bayesian optimization we're interested in solving optimization problems of the form:
$$x_* = arg_{x\in X}\max f(x)$$
where $f$ is a black box.

This black box often has the following properties.
* No gradients
* Expensive to compute
* Observations are corrupted by noise.

Optimization involves designing a sequential strategy which maps collected data to the next query point. For example 'AB testing', 'Hyperparameter tuning'.

## Model based black box optimization"

**Black box optimization**
1. Initial sample
2. Initiate our model
3. Get the acquisition fuinction $\alpha(x)$
4. Optimize it, $x_{next} = arg \max\alpha(x)$
5. Sample new data; update model
6. Repeat, 3->5
7. Make final recommendation

For example bandit problems, reinforcement learning.

In this type of optimization problems one often has to choose between exploration vs exploitation. One can also think about regret minimization. 

So we need a model? We want a model $M$ that can make predictions and maintain a measure of uncertainty. Use a **Gaussian Process**!

Now up untill this point we haven't really discussed the bayesian part about this optimization problem right? So...

## Inference in Gaussian Processes

We can now assume
* A set of noisy observations
$$y\sim N(f(x_n), \sigma^2)$$
* Collected in to vector $y$
* We can form the joint $(f(x), y)$ for some new point $x$

We also need a acquisition function. $\alpha(x)$

Options are:
* Probability of Improvement
* Expected Improvement
* Thompson Sampling
* 


Use skopt. https://scikit-optimize.github.io/stable/modules/generated/skopt.gp_minimize.html

Good start for sampling:
Latin hypercube, Sobol, Halton and Hammersly


## Skopt

```python
import numpy as np
np.random.seed(237)
import matplotlib.pyplot as plt
from skopt.plots import plot_gaussian_process
```

```python
noise_level = 0.1

def f(x, noise_level=noise_level):
    return np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2))\
            + np.random.randn() * noise_level
```

```python
# Plot f(x) + contours
x = np.linspace(-2, 2, 400).reshape(-1, 1)
fx = [f(x_i, noise_level=0.0) for x_i in x]
plt.plot(x, fx, "r--", label="True (unknown)")
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate(([fx_i - 1.9600 * noise_level for fx_i in fx],
                         [fx_i + 1.9600 * noise_level for fx_i in fx[::-1]])),
         alpha=.2, fc="r", ec="None")
plt.legend()
plt.grid()
plt.show()
```

```python
from skopt import gp_minimize

res = gp_minimize(f,                  # the function to minimize
                  [(-2.0, 2.0)],      # the bounds on each dimension of x
                  acq_func="EI",      # the acquisition function
                  n_calls=15,         # the number of evaluations of f
                  n_random_starts=5,  # the number of random initialization points
                  noise=0.1**2,       # the noise level (optional)
                  random_state=1234)   # the random seed
```

```python
"x^*=%.4f, f(x^*)=%.4f" % (res.x[0], res.fun)
```

```python
from skopt.plots import plot_convergence
plot_convergence(res);
```

```python
plt.rcParams["figure.figsize"] = (10, 16)


def f_wo_noise(x):
    return f(x, noise_level=0)
```

```python
for n_iter in range(5):
    # Plot true function.
    plt.subplot(5, 2, 2*n_iter+1)

    if n_iter == 0:
        show_legend = True
    else:
        show_legend = False

    ax = plot_gaussian_process(res, n_calls=n_iter,
                               objective=f_wo_noise,
                               noise_level=noise_level,
                               show_legend=show_legend, show_title=False,
                               show_next_point=False, show_acq_func=False)
    ax.set_ylabel("")
    ax.set_xlabel("")
    # Plot EI(x)
    plt.subplot(5, 2, 2*n_iter+2)
    ax = plot_gaussian_process(res, n_calls=n_iter,
                               show_legend=show_legend, show_title=False,
                               show_mu=False, show_acq_func=True,
                               show_observations=False,
                               show_next_point=True)
    ax.set_ylabel("")
    ax.set_xlabel("")

plt.show()
```

```python
plt.rcParams["figure.figsize"] = (6, 4)

# Plot f(x) + contours
_ = plot_gaussian_process(res, objective=f_wo_noise,
                          noise_level=noise_level)

plt.show()
```

## Strains in the 2d case

```python
sample_id = [1, 2, 3, 4, 5]
strains = ["strainA", "strainB"]

def run_experiment(ratio_strain1, ratio_strain2):
    
```

```python

```
