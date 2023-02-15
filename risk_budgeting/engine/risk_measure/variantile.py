import numpy as np

from risk_budgeting.business.model.riskbudgeting import RiskBudgetingParams
from risk_budgeting.business.model.solve import SolveParams

def variantile_method(
    rb_params: RiskBudgetingParams.__annotations__,
    solve_params: SolveParams.__annotations__,
    store: bool,
) -> SolveParams:

    # Initialize t
    if solve_params.t_init is None:
        t = np.dot(
            np.dot(solve_params.y, np.cov(solve_params.X, rowvar=False)), solve_params.y
        )
    else:
        t = solve_params.t_init
    
    # Create charts
    y_ = [solve_params.y]
    t_ = [t]

    for s in range(solve_params.epochs):
        np.random.shuffle(solve_params.X)
        for i in range(0, solve_params.n, solve_params.minibatch_size):

            # Mini-batch
            x = solve_params.X[i : i + solve_params.minibatch_size]

            # Step size schedule
            eta_t = solve_params.eta_0_t / (1 + solve_params.k) ** solve_params.c
            eta_y = solve_params.eta_0_y / (1 + solve_params.k) ** solve_params.c

            # Gradient
            loss = -np.dot(solve_params.y, x.T)
            indicator_pos = (loss - t >= 0).reshape((x.shape[0], 1))
            indicator_neg = (loss - t < 0).reshape((x.shape[0], 1))
            grad_t = np.mean(
                rb_params.beta * -2 * rb_params.alpha * (loss - t) * indicator_pos + -2 * (1 - rb_params.alpha) * (
                        loss - t) * indicator_neg)
            grad_y = np.mean(rb_params.beta *
                            -2 * rb_params.alpha * (loss - t).reshape((x.shape[0], 1)) * x * indicator_pos + -2 * (
                                    1 - rb_params.alpha) * (loss - t).reshape(
                (x.shape[0], 1)) * x * indicator_neg - rb_params.budgets.value / solve_params.y, axis=0)

            # Descent
            t = t - eta_t * grad_t
            solve_params.y = solve_params.y - eta_y * grad_y
            solve_params.y = np.where(
                solve_params.y <= 0, solve_params.proj_y, solve_params.y
            )

            if solve_params.k + 1 > solve_params.sum_k_first:
                solve_params.y_sum += solve_params.y

            if store:
                y_.append(solve_params.y)
                t_.append(t)

            solve_params.k += 1
            
    return solve_params, t_, y_

        # elif self.rb_params.risk_measure == 'variantile':
        #     # Initialize t
        #     if t_init is None:
        #         t = -np.dot(np.dot(y, np.cov(X, rowvar=False)), y)
        #     else:
        #         t = t_init
        #     t_ = [t]
        #     for s in range(epochs):
        #         np.random.shuffle(X)
        #         for i in range(0, n, minibatch_size):
        #             # Mini-batch
        #             x = X[i:i + minibatch_size]
        #             # Step size schedule
        #             eta_t = eta_0_t / (1 + k) ** c
        #             eta_y = eta_0_y / (1 + k) ** c
        #             # Gradient
        #             loss = -np.dot(y, x.T)
        #             indicator_pos = (loss - t >= 0).reshape((x.shape[0], 1))
        #             indicator_neg = (loss - t < 0).reshape((x.shape[0], 1))
        #             grad_t = np.mean(
        #                 self.rb_params.beta * -2 * self.rb_params.alpha * (loss - t) * indicator_pos + -2 * (1 - self.rb_params.alpha) * (
        #                         loss - t) * indicator_neg)
        #             grad_y = np.mean(self.rb_params.beta *
        #                              -2 * self.rb_params.alpha * (loss - t).reshape((x.shape[0], 1)) * x * indicator_pos + -2 * (
        #                                      1 - self.rb_params.alpha) * (loss - t).reshape(
        #                 (x.shape[0], 1)) * x * indicator_neg - self.rb_params.budgets / y, axis=0)

        #             # Descent
        #             t = t - eta_t * grad_t
        #             y = y - eta_y * grad_y
        #             y = np.where(y <= 0, proj_y, y)

        #             if k + 1 > sum_k_first:
        #                 y_sum += y

        #             if store:
        #                 y_.append(y)
        #                 t_.append(t)

        #             k += 1