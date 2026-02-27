# Multi-exponential-fitting-utilizing-VARPRO-algorithm

## 1. Abstract

This repository provides a compact Python utility for fitting **multi-exponential decay / relaxation models** using the **Variable Projection (VARPRO)** approach. The target model is a sum of exponentials plus (optionally) a constant offset, where the **amplitudes and offset are linear parameters** and the **time constants (or decay rates) are nonlinear parameters**. VARPRO reduces the nonlinear optimization dimension by eliminating the linear parameters through a (weighted) least-squares solve at each iteration. As a result, the remaining optimization is performed only over the nonlinear parameters, which often improves numerical stability and convergence compared with a fully nonlinear formulation—especially for ill-conditioned multi-exponential problems. The implementation is designed to be practical: it builds the exponential basis from the current nonlinear parameters, computes the best linear coefficients, and then calls a standard nonlinear least-squares solver (SciPy) with appropriate residuals and a Jacobian. This repository is intended for research and engineering workflows where quick, reproducible fitting of multi-exponential signals is needed.

---

## 2. How to use

### Requirements
- Python 3.x
- `numpy`
- `scipy` (for nonlinear least-squares)

### Installation (simple)
Clone this repository and make sure the Python files are on your `PYTHONPATH` (or run from the repository root).

### Minimal example
```python
import numpy as np
from varpro4multiexps import VarPro4MultiExponetial

# synthetic signal: 3 exponentials + constant offset
t = np.linspace(0, 0.1, 10001)
y = 0.10*np.exp(-1250*t) + 0.25*np.exp(-1000*t) + 0.45*np.exp(-400*t) + 0.03

# construct fitter
obj = VarPro4MultiExponetial(t, y)

# optional: configure solver behavior
obj.verbose = 1                      # print solver progress
obj.SolverOption.method = "trf"      # e.g., "trf", "dogbox", "lm"

# initial guess: time constants (positive), one per exponential term
result = obj.fit([100.0, 1000.0, 10000.0])

print("Coefficients:", result.Coefficients)
print("Time constants:", result.TimeConstants)
print("Offset:", result.Offset)
