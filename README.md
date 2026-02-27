# Multi-exponential fitting utilizing VARPRO algorithm

## 1. Abstract
This repository provides a lightweight Python implementation for fitting signals with a sum of exponentials using the Variable Projection (VARPRO) approach. The target model is a multi-exponential curve with an optional constant offset, i.e., a linear combination of exponential basis functions where the decay rates (or time constants) are nonlinear parameters, while amplitudes and offset are linear parameters. VARPRO exploits this separable structure: for any fixed set of decay rates, the linear coefficients are solved by a (weighted) linear least-squares problem, and only the nonlinear parameters are optimized iteratively. In this implementation, the weighted linear subproblem is solved via SVD for numerical robustness, and SciPy’s `least_squares` is used for the outer nonlinear optimization. The result is a practical fitter that is often more stable than “all-parameters-at-once” nonlinear least squares, especially when amplitudes are strongly correlated or when the number of exponentials is moderate.

## 2. How to use
### Requirements
- Python 3.x
- `numpy`
- `scipy`

### Quick start

```python
import numpy as np
from varpro4multiexps import VarPro4MultiExponetial

# Example data (synthetic)
t = np.linspace(0.0, 0.1, 10001)
y = 0.10*np.exp(-1250*t) + 0.25*np.exp(-1000*t) + 0.45*np.exp(-400*t) + 0.03
y = y + np.random.normal(0.0, 1e-6, size=t.size)

# Fit: initial_guess is a list of time constants (must be positive)
solver = VarPro4MultiExponetial(t, y)
solver.verbose = 1  # (optional) SciPy verbosity
exps = solver.fit(initial_guess=[100, 1000, 10000])

print("Coefficients:", exps.Coefficients)         # amplitudes
print("TimeConstants:", exps.TimeConstants)       # tau (positive)
print("Offset:", exps.Offset)                     # constant term
# Evaluate fitted curve:
yhat = exps(t)
```

### Weights and solver options

You can pass `weight=` to emphasize/de-emphasize samples. Optimization settings (bounds, tolerances, loss, etc.) are managed via `SolverOption`, optionally loadable from an XML file (`solver_option_path`).

## 3. Theory

Consider measurements $y(t_i)$ and a separable model
\[
y(t_i) \approx \sum_{k=1}^{K} c_k \exp(\lambda_k t_i) + c_0,
\]
where $\lambda_k<0$ are decay coefficients (often parameterized by time constants $\tau_k=-1/\lambda_k>0$), and $\{c_k\}$ plus offset $c_0$ are linear parameters. Define the design matrix $\Phi(\lambda)$ whose columns are $\exp(\lambda_k t)$ plus a column of ones for the offset. With weights $w_i$, the problem is
\[
\min_{\lambda,\,c}\ \|W(\Phi(\lambda)c - y)\|_2^2.
\]
VARPRO eliminates $c$ analytically: for fixed $\lambda$, compute $c(\lambda)=\arg\min_c \|W(\Phi(\lambda)c-y)\|_2$ by linear least squares (here via SVD and a small singular-value cutoff). Then the outer problem becomes $\min_{\lambda}\ \|r(\lambda)\|_2^2$. Efficient optimization requires the Jacobian of the reduced residual; this code forms it from projector-based expressions typical of VARPRO, and maps variables using $\lambda=-1/\tau$ so that time constants stay positive via bounds.

## 4. Literature

- G. H. Golub and V. Pereyra, “The differentiation of pseudoinverses and nonlinear least squares problems whose variables separate,” *SIAM Journal on Numerical Analysis*, 1973.
- L. Kaufman, “A variable projection method for solving separable nonlinear least squares problems,” *Computational Optimization and Applications* / related early VARPRO developments, 1970s.
- D. P. O’Leary and B. W. Rust, “Variable Projection for Nonlinear Least Squares Problems,” *Computational Optimization and Applications*, 2013 (robust formulations, constraints, Jacobians).
- (Optional background) Reviews on separable nonlinear least squares / variable projection applications across exponential fitting and related inverse problems.
