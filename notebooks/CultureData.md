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
import pandas as pd
```

```python
# Choose a kernel
def switch_kernel(arg):
    switcher = {
        1: lambda x,y: x*y,
        2: lambda x,y: min(x,y),
        3: lambda x,y: np.exp(-100*(x-y)*(x-y)),
        4: lambda x,y: np.exp(-1*np.sin(5*np.pi*(x-y))**2)
    }
    return switcher.get(arg, "Invalid argument")

def sample(kernel, x_min, x_max, mu=None):
    # Choose a kernel
    k = switch_kernel(kernel)
    
    # Choose points to sample from
    x = np.arange(x_min, x_max, 0.01)
    n = len(x)

    # Construct the covariance matrix
    C = np.zeros(shape=(n,n))
    for i in range(n):
        for j in range(n):
            C[i,j] = k(x[i], x[j])

    # Sample from a gaussian process at these random points
    mu = mu or np.random.normal()
    A,S,B = np.linalg.svd(C)
    z = A @ np.sqrt(S)* mu

    plt.plot(x, z);
#     plt.xlim(0, 1);
#     plt.ylim(-2, 2);
    plt.title(f"Gaussian process with mu: {mu}")
```

```python
def generate_random_gaussian_data(n_points, sigma, epsilon, x_min, x_max, kernel_function):
    """
    Generate random data point around a random gaussian process.
    
    Params
    ------
    n_points: int, Amount of random data points to generate
    sigma: standard deviation from the mean of the gaussian
    epsilon: deviation from the x coordinates of the gaussian
    x_min: float, lower bound of the x coordinates of the data points
    x_max, float, upper bound of the x coordinates of the data points
    kernel_function: func, kernel function (covar function) of the gaussian to use to generate the GP's
    
    Returns
    -------
    mu: float, The mean of the gaussian.
    x: list, x coordinates of the random data points
    y: list, y coordinates of the random data points
    """
    
    # Choose points to sample from
    x = np.arange(x_min, x_max, 0.01)
    n = len(x)

    # Construct the covariance matrix
    C = np.zeros(shape=(n,n))
    for i in range(n):
        for j in range(n):
            C[i,j] = kernel_function(x[i], x[j])

    # Sample from a gaussian process at these random points
    mu = np.random.normal()
    A,S,B = np.linalg.svd(C)
    z = A @ np.sqrt(S)* mu
    
    random_ix = np.random.choice(np.arange(len(x)), size=n_points)
    
    random_x = x[random_ix] + np.random.normal(0, epsilon, size=n_points)
    random_y = z[random_ix] + np.random.normal(0, sigma, size=n_points)
    
    return (mu, random_x, random_y)
```

```python
mu_A, biomass_A, viscosity_A = generate_random_gaussian_data(50, 0.01, 0.05, 0, 1, switch_kernel(3))

# plt.scatter(x,y)
sample(3, 0, 1, mu=mu_A)
plt.scatter(biomass_A, viscosity_A);
plt.xlabel("biomass, strainA")
plt.ylabel("viscosity");
```

```python
mu_B, biomass_B, viscosity_B = generate_random_gaussian_data(50, 0.01, 0.05, 0, 1, switch_kernel(3))

# plt.scatter(x,y)
sample(3, 0, 1, mu=mu_B)
plt.scatter(biomass_B, viscosity_B);
plt.xlabel("biomass, strainB")
plt.ylabel("viscosity");
```

```python
def v(X, Y):
    return viscosity_A[X] + viscosity_B[Y]
```

```python
x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 40)

Z = v(X, Y)

plt.contour(X, Y, Z, 20, cmap = 'RdGy')
# plt.contourf(X, Y, Z, 50, cmap = 'RdGy')
```

```python

```
