import numpy as np
from scipy.stats import norm


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

    def __init__(self,
                 risk_measure='volatility',
                 budgets='ERC',
                 expectation=False,
                 beta=1.00,
                 delta=1.00,
                 alpha=None,
                 gamma=None
                 ):

        self.risk_measure = risk_measure
        self.budgets = budgets
        self.expectation = expectation
        self.beta = beta
        self.delta = delta
        self.alpha = alpha
        self.gamma = gamma
        self.ys = None
        self.ts = None
        self.success = None
        self.x = None

    def solve(self, X, epochs=None, minibatch_size=128, y_init=None, t_init=None, eta_0_y=None, eta_0_t=None, c=0.65,
              polyak_ruppert=0.2, discretize=None, proj_y=None, store=False):

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

        n, d = X.shape

        # Set budgets if ERC
        if type(self.budgets) == str:
            if self.budgets == 'ERC':
                self.budgets = np.ones(d) / d

        if False in self.budgets > 0 or True in self.budgets >= 0:
            raise ValueError('The budgets should be in the range (0,1).')

        # Choose number of epochs based on sample size
        if epochs is None:
            epochs = int(2e06 / n)

        # Initialize y
        if y_init is None:
            y = self.budgets / np.std(X, axis=0)
        else:
            y = y_init

        if proj_y is None:
            proj_y = y

        # Set step size coefficients for y and t
        if eta_0_y is None:
            eta_0_y = 360 / d
        if eta_0_t is None:
            eta_0_t = .5

        if self.beta <= 0:
            raise ValueError('beta should greater than 0.')

        # Needed for Polyak-Ruppert averaging
        y_sum = np.zeros(d)
        sum_k_first = int((1 - polyak_ruppert) * (epochs * n / minibatch_size))

        # Store along the optimization path
        y_ = [y]
        k = 0

        if self.risk_measure == 'volatility':
            # Initialize t
            if t_init is None:
                t = np.dot(np.dot(y, np.cov(X, rowvar=False)), y)
            else:
                t = t_init
            t_ = [t]
            for s in range(epochs):
                np.random.shuffle(X)
                for i in range(0, n, minibatch_size):

                    # Mini-batch
                    x = X[i:i + minibatch_size]

                    # Step size schedule
                    eta_t = eta_0_t / (1 + k) ** c
                    eta_y = eta_0_y / (1 + k) ** c

                    # Gradient
                    r = np.dot(y, x.T)
                    grad_t = np.mean(self.beta * -2 * (r - t))
                    grad_y = np.mean(self.beta *
                                     2 * (r - t).reshape((x.shape[0], 1)) * x - self.budgets / y -
                                     self.delta * self.expectation * x, axis=0)

                    # Descent
                    t = t - eta_t * grad_t
                    y = y - eta_y * grad_y
                    y = np.where(y <= 0, proj_y, y)

                    if k + 1 > sum_k_first:
                        y_sum += y

                    if store:
                        y_.append(y)
                        t_.append(t)

                    k += 1

        elif self.risk_measure == 'median_absolute_deviation':
            # Initialize t
            if t_init is None:
                t = np.dot(np.dot(y, np.cov(X, rowvar=False)), y)
            else:
                t = t_init
            t_ = [t]
            for s in range(epochs):
                np.random.shuffle(X)
                for i in range(0, n, minibatch_size):

                    # Mini-batch
                    x = X[i:i + minibatch_size]

                    # Step size schedule
                    eta_t = eta_0_t / (1 + k) ** c
                    eta_y = eta_0_y / (1 + k) ** c

                    # Gradient
                    indicator_pos = (np.dot(y, x.T) - t >= 0).reshape((x.shape[0], 1))
                    indicator_neg = 1 - indicator_pos
                    grad_t = np.mean(self.beta * -1 * indicator_pos + indicator_neg)
                    grad_y = np.mean(self.beta * x * (
                            indicator_pos - indicator_neg) - self.budgets / y - self.delta * self.expectation * x,
                                     axis=0)

                    # Descent
                    t = t - eta_t * grad_t
                    y = y - eta_y * grad_y
                    y = np.where(y <= 0, proj_y, y)

                    if k + 1 > sum_k_first:
                        y_sum += y

                    if store:
                        y_.append(y)
                        t_.append(t)

                    k += 1

        elif self.risk_measure == 'expected_shortfall':
            # Initialize t
            if t_init is None:
                t = -np.dot(y, np.mean(X, axis=0)) + np.dot(np.dot(y, np.cov(X, rowvar=False)), y) * norm.ppf(
                    self.alpha)
            else:
                t = t_init
            t_ = [t]
            for s in range(epochs):
                np.random.shuffle(X)
                for i in range(0, n, minibatch_size):

                    # Mini-batch
                    x = X[i:i + minibatch_size]

                    # Step size schedule
                    eta_t = eta_0_t / (1 + k) ** c
                    eta_y = eta_0_y / (1 + k) ** c

                    # Gradient
                    indicator = (-np.dot(y, x.T) - t >= 0).reshape((x.shape[0], 1))
                    grad_t = np.mean(self.beta * 1 - (1 / (1 - self.alpha)) * indicator)
                    grad_y = np.mean(self.beta *
                                     (-x / (
                                             1 - self.alpha)) * indicator - self.budgets / y + self.delta * self.expectation * x,
                                     axis=0)

                    # Descent
                    t = t - eta_t * grad_t
                    y = y - eta_y * grad_y
                    y = np.where(y <= 0, proj_y, y)

                    if k + 1 > sum_k_first:
                        y_sum += y

                    if store:
                        y_.append(y)
                        t_.append(t)

                    k += 1

        elif self.risk_measure == 'power_spectral_risk_measure':
            # Initialize t
            if discretize is None:
                discretize = {'step': 50, 'bounds': (.5, .99)}
            u = np.linspace(discretize['bounds'][0], discretize['bounds'][1], discretize['step'])
            w = (self.gamma * u ** (self.gamma - 1))  # power law
            delta_w = np.diff(w)
            u = u[1:]
            if t_init is None:
                t = -np.dot(y, np.mean(X, axis=0)) + np.dot(np.dot(y, np.cov(X, rowvar=False)), y) * norm.ppf(u)
            else:
                t = t_init
            t_ = [t]
            for s in range(epochs):
                np.random.shuffle(X)
                for i in range(0, n, minibatch_size):

                    # Mini-batch
                    x = X[i:i + minibatch_size]

                    # Step size schedule
                    eta_t = eta_0_t / (1 + k) ** c
                    eta_y = eta_0_y / (1 + k) ** c

                    # Gradient
                    indicator = (-np.dot(y, x.T)[:, None] - t >= 0)
                    grad_t = np.mean(self.beta * delta_w * (1 - u) - delta_w * indicator, axis=0)
                    grad_y = np.mean(self.beta *
                                     -np.dot(delta_w, indicator.T).reshape(
                                         (x.shape[0], 1)) * x - self.budgets / y + self.delta * self.expectation * x,
                                     axis=0)

                    # Descent
                    t = t - eta_t * grad_t
                    y = y - eta_y * grad_y
                    y = np.where(y <= 0, proj_y, y)

                    if k + 1 > sum_k_first:
                        y_sum += y

                    if store:
                        y_.append(y)
                        t_.append(t)

                    k += 1

        elif self.risk_measure == 'variantile':
            # Initialize t
            if t_init is None:
                t = -np.dot(np.dot(y, np.cov(X, rowvar=False)), y)
            else:
                t = t_init
            t_ = [t]
            for s in range(epochs):
                np.random.shuffle(X)
                for i in range(0, n, minibatch_size):
                    # Mini-batch
                    x = X[i:i + minibatch_size]
                    # Step size schedule
                    eta_t = eta_0_t / (1 + k) ** c
                    eta_y = eta_0_y / (1 + k) ** c
                    # Gradient
                    loss = -np.dot(y, x.T)
                    indicator_pos = (loss - t >= 0).reshape((x.shape[0], 1))
                    indicator_neg = (loss - t < 0).reshape((x.shape[0], 1))
                    grad_t = np.mean(
                        self.beta * -2 * self.alpha * (loss - t) * indicator_pos + -2 * (1 - self.alpha) * (
                                loss - t) * indicator_neg)
                    grad_y = np.mean(self.beta *
                                     -2 * self.alpha * (loss - t).reshape((x.shape[0], 1)) * x * indicator_pos + -2 * (
                                             1 - self.alpha) * (loss - t).reshape(
                        (x.shape[0], 1)) * x * indicator_neg - self.budgets / y, axis=0)

                    # Descent
                    t = t - eta_t * grad_t
                    y = y - eta_y * grad_y
                    y = np.where(y <= 0, proj_y, y)

                    if k + 1 > sum_k_first:
                        y_sum += y

                    if store:
                        y_.append(y)
                        t_.append(t)

                    k += 1

        else:
            raise ValueError('The given risk measure is not applicable.')

        y_sgd = y_sum / int(polyak_ruppert * (epochs * n / minibatch_size))
        theta_sgd = y_sgd / y_sgd.sum()

        self.x = theta_sgd

        if store:
            self.ys = y_
            self.ts = t_
