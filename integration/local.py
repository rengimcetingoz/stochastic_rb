import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('./daily_returns_data.csv', index_col=0)

sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__)),"../"))

from risk_budgeting import RiskBudgeting

np.random.seed(0)

X = data.values

# we define our risk budgeting problem
# rb = RiskBudgeting(rb_params = {
#     "risk_measure" : 'volatility',
#     "budgets" : {
#         "name" : "CUSTOM",
#         "value": np.array([0.5, 0.3, 0.2])}})

rb = RiskBudgeting(rb_params = {
    "risk_measure" : 'volatility',
    "budgets" : {
        "name" : 'ERC'}})

# rb.solve(X)
# print(rb.x)

rb.solve(params_solver = {"X" : X}, store=False)
print('ERC volatility:', rb.solution)

rb = RiskBudgeting(rb_params = {
    "risk_measure" : 'volatility',
    "budgets" : {
        "name" : "CUSTOM",
        "value": np.array([0.5, 0.3, 0.2])}})

rb.solve(params_solver = {"X" : X}, store=False)
print('CUSTOM volatility:', rb.solution)


rb = RiskBudgeting(rb_params = {
    "risk_measure" : 'median_absolute_deviation',
    "budgets" : {
        "name" : "ERC",
        }})

rb.solve(params_solver = {"X" : X}, store=False)
print('ERC MAD:', rb.solution)

rb = RiskBudgeting(rb_params = {
    "risk_measure" : 'expected_shortfall',
    "budgets" : {
        "name" : "ERC",
        }, 
    "alpha" : .95})

rb.solve(params_solver = {"X" : X}, store=False)
print('ERC Expected Shortfall:', rb.solution)


rb = RiskBudgeting(rb_params = {
    "risk_measure" : 'power_spectral_risk_measure',
    "budgets" : {
        "name" : "ERC",
        },
    "gamma" : 20.0})

rb.solve(params_solver = {"X" : X}, store=False)
print('ERC Power Spectral Risk Measure:', rb.solution)