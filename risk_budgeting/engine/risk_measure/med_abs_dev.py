import numpy as np

def median_absolute_deviation_method(rb_params, solve_params):
    # Initialize t
    if solve_params.t_init is None:
        t = np.dot(np.dot(y, np.cov(solve_params.X, rowvar=False)), y)
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
            eta_t = solve_params.eta_0_t / (1 + k) ** solve_params.c
            eta_y = solve_params.eta_0_y / (1 + k) ** solve_params.c

            # Gradient
            indicator_pos = (np.dot(y, x.T) - t >= 0).reshape((x.shape[0], 1))
            indicator_neg = 1 - indicator_pos
            grad_t = np.mean(rb_params.beta * -1 * indicator_pos + indicator_neg)
            grad_y = np.mean(rb_params.beta * x * (
                    indicator_pos - indicator_neg) - rb_params.budgets / y - rb_params.delta * rb_params.expectation * x,
                                axis=0)

            # Descent
            t = t - eta_t * grad_t
            y = y - eta_y * grad_y
            y = np.where(y <= 0, solve_params.proj_y, y)

            if k + 1 > solve_params.sum_k_first:
                y_sum += y

            if solve_params.store:
                y_.append(y)
                t_.append(t)

            k += 1
    return solve_params, t_, y_