# Risk Budgeting using Stochastic Gradient Descent
The codes in this repository can be used to compute risk budgeting portfolios for various risk measures from volatility to expected shortfall. It only requires a sample set of asset returns (historical or simulated) to be used in the stochastic optimization process. Implementation is based on [*Stochastic Algorithms for Advanced Risk Budgeting*](https://arxiv.org/abs/2211.07212) by Cetingoz, Fermanian and Guéant. 

This repository allows to compute the risk budgeting portfolio for the following risk measures:
- Volatility
- Median Absolute Deviation 
- Median Absolute Deviation with expected return
- Expected Shortfall 
- Expected Shortfall without expected return
- Power Spectral Measure
- Power Spectral Measure without expected return
- Variantile

## Installation

```bash
pip install git+https://github.com/rengo-python/stochastic_rb.git
```

## Usage

```python
from risk_budgeting import RiskBudgeting

# define the problem
rb = RiskBudgeting(risk_measure='volatility', budgets='ERC')

# solve the defined problem for a given array of asset returns X
rb.solve(X)

# get computed weights
rb.x
```
