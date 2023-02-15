import numpy as np
from typeguard import typechecked

from risk_budgeting.business.model.riskbudgeting import RiskBudgetingParams
from risk_budgeting.business.model.solve import SolveParams
from risk_budgeting.engine import risk_measure
from risk_budgeting.utils.exceptions import (
    BetaSizeNotCorrect,
    BudgetsValueSizeNotCorrect,
)


@typechecked
class RiskBudgeting:
    """

    Representation of a Risk Budgeting problem.
    This class allows to find the Risk Budgeting portfolio for different risk measures under given additional
    specifications.

    Parameters
    ----------

    risk_measure : {'volatility' (default),
                    'median_absolute_deviation',
                    'expected_shortfall',
                    'power_spectral_risk_measure',
                    'variantile'}
        String describing the type of risk measure to use.

    budgets : {'ERC' (default), numpy.ndarray}
        String or array describing the risk budgets. 'ERC' stands for Equal Risk Contribution. In other cases, risk budgets
        should be given as an array with relevant dimension.

    expectation : bool, default to False.

    beta : float, defaults to 1.00
        Weight of the risk measure component when 'expectation' is True. Not used when 'expectation' is False.

    delta : float, defaults to 1.00
        Weight of the expected return component when 'expectation' is True. Not used when 'expectation' is False.

    alpha : float
        Confidence level when 'risk_measure' is 'expected_shortfall'. Weight of the first component when 'risk_measure'
        is 'variantile'. Not used in other cases.

    gamma : float
        Coefficient of the power utility function needed when 'risk_measure' is 'power_spectral_risk_measure' Not used
        in other cases.

    Attributes
    ----------
    x : numpy.ndarray
        The weights of the computed Risk Budgeting portfolio.

    ys: numpy.ndarray, default to None.
        If 'store' parameter in solve() function is True, store y vectors along the optimization path.

    ts: numpy.ndarray, default to None.
        If 'store' parameter in solve() function is True, store t values along the optimization path.

    """

    ys = None
    ts = None
    success = None
    x = None

    def __init__(self, rb_params: RiskBudgetingParams.__annotations__):
        self.rb_params = RiskBudgetingParams(**rb_params)

    def __settings(self) -> None:
        # Store along the optimization path
        self.solve_params.k = 0

        self.solve_params.n, self.solve_params.d = self.solve_params.X.shape

        # Set budgets if ERC
        if self.rb_params.budgets.name == "ERC":
            self.rb_params.budgets.value = (
                np.ones(self.solve_params.d) / self.solve_params.d
            )

        if (
            False in self.rb_params.budgets.value > 0
            or True in self.rb_params.budgets.value >= 0
        ):
            raise BudgetsValueSizeNotCorrect(self.rb_params.budgets.value)

        # Choose number of epochs based on sample size
        if self.solve_params.epochs is None:
            self.solve_params.epochs = int(2e06 / self.solve_params.n)

        # Initialize y
        if self.solve_params.y_init is None:
            self.solve_params.y = self.rb_params.budgets.value / np.std(
                self.solve_params.X, axis=0
            )
        else:
            self.solve_params.y = self.solve_params.y_init

        if self.solve_params.proj_y is None:
            self.solve_params.proj_y = self.solve_params.y

        # Set step size coefficients for y and t
        if self.solve_params.eta_0_y is None:
            self.solve_params.eta_0_y = 360 / self.solve_params.d
        if self.solve_params.eta_0_t is None:
            self.solve_params.eta_0_t = 0.5

        if self.rb_params.beta <= 0:
            raise BetaSizeNotCorrect(self.rb_params.beta)

        # Needed for Polyak-Ruppert averaging
        self.solve_params.y_sum = np.zeros(self.solve_params.d)
        self.solve_params.sum_k_first = int(
            (1 - self.solve_params.polyak_ruppert)
            * (
                self.solve_params.epochs
                * self.solve_params.n
                / self.solve_params.minibatch_size
            )
        )

    def solve(
        self, params_solver: SolveParams.__annotations__, store: bool = False, **kwargs
    ) -> np.ndarray:

        """

        Solves the defined risk budgeting problem using a given sample of asset returns via
        stochastic gradient descent and returns the risk budgeting portfolio.

        Parameters
        ----------

        X : numpy.ndarray shape=(n,d)
            Sample of asset returns

        epochs : int, optional. Defaults to int(2e06/n).
            Number of epochs.

        minibatch_size : int, optional. Defaults to 128.
            Mini-batch size.

        y_init (numpy.ndarray, optional): numpy.ndarray shape=(d,). Defaults to a vector which is a
            solution to risk budgeting problem for volatility under the assumption that the correlation matrix is
            all-ones matrix.
            Initial value for each element of the vector of asset weights.

        t_init : float, optional. Defaults to a minimizer of a similar problem with analytical solution.
            Initial value for t.

        eta_0_t : float, optional. Defaults to 0.5.
            Step size coefficient for variable t.

        eta_0_y : float, optional. Defaults to 50/d.
            Step size coefficient for vector y.

        c : float, optional. Defaults to 0.65.
            Step size power.

        polyak_ruppert : float, optional. Defaults to 0.2.
             Polyak-Ruppert averaging for last % iterates.

        discretize : dict, optional. Defaults to {'step': 50, 'bounds': (.5, .99)}
            Parameters to discretize the integral for spectral risk measures.

        proj_y : float, optional. Defaults to y_init.
            Value for projection of asset weights into the feasible space.

        store : bool, optional. Defaults to False.
            store y and t along the optimization path.

        """
        # TODO : Update, maybe remove all parameters from method and rename all params, not very explicite
        self.solve_params = SolveParams(**params_solver)
        self.__settings()

        if self.rb_params.risk_measure == "volatility":
            solve_params_update, t_, y_ = risk_measure.volatility_method(
                self.rb_params, self.solve_params, store=store
            )

        elif self.rb_params.risk_measure == 'median_absolute_deviation':
            solve_params_update, t_, y_ = risk_measure.median_absolute_deviation_method(
                self.rb_params, self.solve_params, store=store
                )

        elif self.rb_params.risk_measure == 'expected_shortfall':
            solve_params_update, t_, y_ = risk_measure.expected_shortfall_method(
                self.rb_params, self.solve_params, store=store
                )

        # elif self.rb_params.risk_measure == 'expected_shortfall':
        #     # Initialize t
        #     if t_init is None:
        #         t = -np.dot(y, np.mean(X, axis=0)) + np.dot(np.dot(y, np.cov(X, rowvar=False)), y) * norm.ppf(
        #             self.rb_params.alpha)
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
        #             indicator = (-np.dot(y, x.T) - t >= 0).reshape((x.shape[0], 1))
        #             grad_t = np.mean(self.rb_params.beta * 1 - (1 / (1 - self.rb_params.alpha)) * indicator)
        #             grad_y = np.mean(self.rb_params.beta *
        #                              (-x / (
        #                                      1 - self.rb_params.alpha)) * indicator - self.rb_params.budgets / y + self.rb_params.delta * self.rb_params.expectation * x,
        #                              axis=0)

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

        # elif self.rb_params.risk_measure == 'power_spectral_risk_measure':
        #     # Initialize t
        #     if discretize is None:
        #         discretize = {'step': 50, 'bounds': (.5, .99)}
        #     u = np.linspace(discretize['bounds'][0], discretize['bounds'][1], discretize['step'])
        #     w = (self.rb_params.gamma * u ** (self.rb_params.gamma - 1))  # power law
        #     delta_w = np.diff(w)
        #     u = u[1:]
        #     if t_init is None:
        #         t = -np.dot(y, np.mean(X, axis=0)) + np.dot(np.dot(y, np.cov(X, rowvar=False)), y) * norm.ppf(u)
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
        #             indicator = (-np.dot(y, x.T)[:, None] - t >= 0)
        #             grad_t = np.mean(self.rb_params.beta * delta_w * (1 - u) - delta_w * indicator, axis=0)
        #             grad_y = np.mean(self.rb_params.beta *
        #                              -np.dot(delta_w, indicator.T).reshape(
        #                                  (x.shape[0], 1)) * x - self.rb_params.budgets / y + self.rb_params.delta * self.rb_params.expectation * x,
        #                              axis=0)

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

        # else:
        #     raise ValueError('The given risk measure is not applicable.')

        y_sgd = solve_params_update.y_sum / int(
            solve_params_update.polyak_ruppert
            * (
                solve_params_update.epochs
                * solve_params_update.n
                / solve_params_update.minibatch_size
            )
        )
        self.solution = y_sgd / y_sgd.sum()

        # TODO : Create specific Method // Don't mix solver params and matplotlib
        if store:
            self.ys = y_
            self.ts = t_
        return self.solution
