import numpy as np
from scipy.stats import norm

from risk_budgeting.business.model.riskbudgeting import RiskBudgetingParams
from risk_budgeting.business.model.solve import SolveParams, Discretize

def power_spectral_risk_measure_method(
    rb_params: RiskBudgetingParams.__annotations__,
    solve_params: SolveParams.__annotations__,
    store: bool,
) -> SolveParams:

    # Discretization
    if solve_params.discretize is None:
        Discretize.step = 50
        Discretize.bounds = (.5, .99)

    u = np.linspace(Discretize.bounds[0], Discretize.bounds[1], Discretize.step)
    w = (rb_params.gamma * u ** (rb_params.gamma - 1))  # power law
    delta_w = np.diff(w)
    u = u[1:]

    # Initialize t
    if solve_params.t_init is None:
        t = -np.dot(solve_params.y, np.mean(solve_params.X, axis=0)) + np.dot(np.dot(solve_params.y, 
        np.cov(solve_params.X, rowvar=False)), solve_params.y) * norm.ppf(u)
    else:
        t = solve_params.t_init

    y_ = [solve_params.y]
    t_ = [t]

    for s in range(solve_params.epochs): 
        np.random.shuffle(solve_params.X)
        for i in range(0, solve_params.n, solve_params.minibatch_size):

            # Mini-batch
            x = solve_params.X[i:i + solve_params.minibatch_size]

            # Step size schedule
            eta_t = solve_params.eta_0_t / (1 + solve_params.k) ** solve_params.c
            eta_y = solve_params.eta_0_y / (1 + solve_params.k) ** solve_params.c

            # Gradient
            indicator = (-np.dot(solve_params.y, x.T)[:, None] - t >= 0)
            grad_t = np.mean(rb_params.beta * delta_w * (1 - u) - delta_w * indicator, axis=0)
            grad_y = np.mean(rb_params.beta *
                                -np.dot(delta_w, indicator.T).reshape(
                                    (x.shape[0], 1)) * x - rb_params.budgets.value / solve_params.y + rb_params.delta * rb_params.expectation * x,
                                axis=0)
            
            # Descent
            t = t - eta_t * grad_t
            solve_params.y = solve_params.y - eta_y * grad_y
            solve_params.y = np.where(solve_params.y <= 0, solve_params.proj_y, solve_params.y)

            if solve_params.k + 1 > solve_params.sum_k_first:
                solve_params.y_sum += solve_params.y

            if store:
                y_.append(solve_params.y)
                t_.append(t)

            solve_params.k += 1

    return solve_params, t_, y_
        