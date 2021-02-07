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

```python
import numpy as np
import matplotlib.pyplot as plt
```

<!-- #region -->
# Gaussian Process

In this notebook we're going to explore how a Gaussian Process works, what it is and how we can implement it in python. Let's start with a definition.

***Definition 1:***<br>
For any set $S$ a gaussian process on $S$ is a set of random variables $\{ Z_{t}, t \in S\}$ such that $\forall n \in N$, $t_{1},...,t_{n} \in S$ the vector $(Z_{t_1}, Z_{t_2}, ... , Z_{t_n})$ is multivariate Gaussian distributed.


***Example 1:***<br>
Let $S=\{1,2,...,d\}$, and $Z_t = (Z_1, Z_2, ..., Z_d)\in \mathbb{R}^d$, where $Z_i$ are Gaussian distributed random variables. Then by the affine property of the gaussian $Z_t$ is multivariate gaussian. (trivial example)

***Example 2:***<br>
Random lines. Let $S = \mathbb{R}$. $Z_t = tW$, where $W \sim N(0,1)$. Then $\forall n\in N$, $t_1,...,t_n \in S$, $(Z_{t_1},...,Z_{t_n})$ is multivariate gaussian.
$$(Z_t) = (Z_{t_1},...,Z_{t_n}) = (t_1W, t_2W,...,t_nW) = (t_1, t_2,..., t_n)\cdot W = AW,$$
Where $AW$ is multivariate gaussian, because of the affine property (matrix $A$ with multivariate [1x1] gaussian $W$).
<!-- #endregion -->

```python
# Example 2
Ws = np.random.normal(size=5)
print("W's: ", Ws)

for W in Ws:
    x = np.arange(-2, 2, 0.2)
    y = lambda x: W*x
    plt.plot(x, y(x));
    plt.xlabel("t")
    plt.ylabel("y")
    plt.title("Five random draws of a Gaussian Process defined in Example 2.")
```

Now this is a great example of a Gaussian Process, but it doesn't really describe one of the key elements of a Gaussian Process: 'The Kernel'. Or also the Covariance function. So let's define a couple of kernels.

```python
# Choose a kernel
def switch_kernel(arg):
    switcher = {
        1: lambda x,y: x*y,
        2: lambda x,y: min(x,y),
        3: lambda x,y: np.exp(-100*(x-y)*(x-y))
    }
    return switcher.get(arg, "Invalid argument")

def sample(kernel):
    # Choose a kernel
    k = switch_kernel(kernel)
    
    # Choose points to sample from
    x = np.arange(0, 1, 0.01)
    n = len(x)

    # Construct the covariance matrix
    C = np.zeros(shape=(n,n))
    for i in range(n):
        for j in range(n):
            C[i,j] = k(x[i], x[j])

    # Sample from a gaussian process at these random points
    mu = np.random.normal()
    A,S,B = np.linalg.svd(C)
    z = A @ np.sqrt(S)* mu

    plt.plot(x, z);
    plt.xlim(0, 1);
    plt.ylim(-2, 2);
```

```python
sample(1);
```

Standard Brownian Motion, Wiener Process

```python
sample(2);
```

```python
sample(3);
```

```python

```

```python

```

```python

```

```python

```

A gaussian process needs a kernel, which is also the covariance function

First we need a covariance function!

$$K(x \mid \alpha, \rho, \sigma)_{i, j}=\alpha^{2} \exp \left(-\frac{1}{2 \rho^{2}} \sum_{d=1}^{D}\left(x_{i, d}-x_{j, d}\right)^{2}\right)+\delta_{i, j} \sigma^{2}$$

```python
def kronecker_delta(i, j):
    if i == j:
        return 1
    else:
        return 0
    
kronecker_delta(1,1)
```

Easy! Let $D=3$, be the number of dimensions for our input. Then $\left(x_{i, d}-x_{j, d}\right)^{2}$ is just the L2 norm between the input variables.

```python
def K(x, alpha, rho, sigma):
    alpha**2 * np.exp(- (1/(2*rho**2)*np.sum) )
```

```python
np.subtract.outer([6,1], [1,12])**2
```

```python
import numpy as np
 
def exponential_cov(x, y, params):
    return params[0] * np.exp( -0.5 * params[1] * np.subtract.outer(x, y)**2)
```

```python
def conditional(x_new, x, y, params):
    B = exponential_cov(x_new, x, params)
    C = exponential_cov(x, x, params)
    A = exponential_cov(x_new, x_new, params)

    mu = np.linalg.inv(C).dot(B.T).T.dot(y)
    sigma = A - B.dot(np.linalg.inv(C).dot(B.T))

    return(mu.squeeze(), sigma.squeeze())
```

```python
import matplotlib.pylab as plt
 
θ = [1, 10]
σ_0 = exponential_cov(0, 0, θ)
xpts = np.arange(-3, 3, step=0.01)
plt.errorbar(xpts, np.zeros(len(xpts)), yerr=σ_0, capsize=0)
```

```python
x = [1.]
y = [np.random.normal(scale=σ_0)]
y
```

```python
σ_1 = exponential_cov(x, x, θ)
 
def predict(x, data, kernel, params, sigma, t):
    k = [kernel(x, y, params) for y in data]
    Sinv = np.linalg.inv(sigma)
    y_pred = np.dot(k, Sinv).dot(t)
    sigma_new = kernel(x, x, params) - np.dot(k, Sinv).dot(k)
    return y_pred, sigma_new
 
x_pred = np.linspace(-3, 3, 1000)
predictions = [predict(i, x, exponential_cov, θ, σ_1, y) for i in x_pred]
```

```python
y_pred, sigmas = np.transpose(predictions)
plt.errorbar(x_pred, y_pred, yerr=sigmas, capsize=0)
plt.plot(x, y, "ro")
```

```python

```
