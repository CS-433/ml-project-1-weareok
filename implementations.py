import numpy as np


def compute_loss_MSE(y, tx, w):
    """Calculate the MSE for a linear model.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e = y - np.dot(tx, w)
    return np.mean(e**2) / 2


def compute_gradient_least_squares(y, tx, w):
    """Computes the gradient at w for least squares.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    e = y - np.dot(tx, w)
    return -np.dot(tx.T, e) / len(y)


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm for least squares for least squares.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losse: scalar
        w: numpy array of shape=(2, )
    """
    w = initial_w
    for n_iter in range(max_iters):
        loss = compute_loss_MSE(y, tx, w)
        grad = compute_gradient_least_squares(y, tx, w)
        w -= gamma * grad
    loss = compute_loss_MSE(y, tx, w)
    return w, loss


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Alias for the previous function
    """
    return least_squares_GD(y, tx, initial_w, max_iters, gamma)


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def compute_stoch_gradient_least_squares(y, tx, w, n):
    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.
        n: batch size

    Returns:
        A numpy array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """
    for minibatch_y, minibatch_tx in batch_iter(y, tx, n):
        grad = compute_gradient_least_squares(y, tx, w)
        break
    return grad


def least_squares_SGD(y, tx, initial_w, max_iters, gamma, batch_size=1):
    """The Stochastic Gradient Descent algorithm (SGD) for least squares.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient

    Returns:
        losse: scalar
        w: numpy array of shape=(2, )
    """
    w = initial_w

    for n_iter in range(max_iters):
        loss = compute_loss_MSE(y, tx, w)
        grad = compute_stoch_gradient_least_squares(y, tx, w, batch_size)
        w -= gamma * grad
    loss = compute_loss_MSE(y, tx, w)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Alias for the previous function
    """
    return least_squares_SGD(y, tx, initial_w, max_iters, gamma)


def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.
    """
    w = np.linalg.solve(np.dot(tx.T, tx), np.dot(tx.T, y))
    loss = compute_loss_MSE(y, tx, w)


    return w, loss


def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
    """
    N, D = np.shape(tx)

    # covariance_inv = np.linalg.inv(covariance_reg)

    w = np.linalg.lstsq(tx.T.dot(tx) + 2 * N * lambda_ * np.eye(D), tx.T.dot(y), rcond=None)[0]
   # loss = compute_loss_MSE(y, tx, w_start)


    #w = np.linalg.lstsq(np.dot(tx.T, tx) + 2 * N * lambda_ * np.identity(D), np.dot(tx.T, y), rcond=None)[0]

    loss = compute_loss_MSE(y, tx, w)

    return w, loss


def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array
    """
    t = np.array(t)
    out = t * 0
    out[t < 0] = np.exp(t[t < 0]) / (1 + np.exp(t[t < 0]))
    out[t >= 0] = 1 / (1 + np.exp(-t[t >= 0]))

    return out

def calculate_loss_lr(y, tx, w):
    """compute the cost by negative log likelihood for logistic regression.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]
    e = np.dot(tx, w)
    return -np.mean(y * e - np.log(1 + np.exp(e)))


def calculate_gradient_lr(y, tx, w):
    """compute the gradient of loss for logistic regression.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a vector of shape (D, 1)
    """
    return np.dot(tx.T, sigmoid(np.dot(tx, w)) - y) / len(y)


def lr_step(y, tx, w, gamma):
    """
    Do one step of gradient descent, using the logistic regression.
    Return the loss and updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: scalar

    Returns:
        loss: scalar number
        w: shape=(D, 1)
    """
    loss, gradient = calculate_loss_lr(y, tx, w), calculate_gradient_lr(y, tx, w)
    w -= gamma * gradient
    return loss, w


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression on y and features tx.
    Return the loss and the optimal w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        initial_w:  shape=(D, 1)
        max_iters:  scalar
        gamma: scalar

    Returns:
        loss: scalar number
        w: shape=(D, 1)
    """
    threshold = 1e-8
    w = initial_w
    losses = []

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = lr_step(y, tx, w, gamma)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w, calculate_loss_lr(y, tx, w)


def reg_lr_step(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: scalar
        lambda_: scalar

    Returns:
        loss: scalar number
        w: shape=(D, 1)
    """
    loss, gradient = calculate_loss_lr(y, tx, w), calculate_gradient_lr(y, tx, w)
    gradient += 2 * lambda_ * w
    w -= gamma * gradient
    return loss, w


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression on y and features tx.
    Return the loss and the optimal w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        lambda: scalar
        initial_w:  shape=(D, 1)
        max_iters:  scalar
        gamma: scalar

    Returns:
        loss: scalar number
        w: shape=(D, 1)
    """
    threshold = 1e-8
    w = initial_w
    losses = []

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = reg_lr_step(y, tx, w, gamma, lambda_)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 2 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w, calculate_loss_lr(y, tx, w)


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)