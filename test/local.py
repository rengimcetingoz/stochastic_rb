import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('/Users/floriancarpentey/Documents/stochastic_rb/test/daily_returns_data.csv', index_col=0)

sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__)),"../"))

from risk_budgeting import RiskBudgeting

np.random.seed(0)

X = data.values

# we define our risk budgeting problem
rb = RiskBudgeting(params = {
    "risk_measure" : 'volatility',
    "budgets" : {
        "name" : "CUSTOM",
        "value": np.array([0.5, 0.3, 0.2])}})

# rb = RiskBudgeting(params = {
#     "risk_measure" : 'volatility',
#     "budgets" : {
#         "name" : 'ERC'}})

# rb.solve(X)
# print(rb.x)

rb.solve(X, store=True)
print(plt.plot(rb.ys))