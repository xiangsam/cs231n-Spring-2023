from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # write by samrito
    # log_sum_exp is not a good method for numeric stability here, see the other implementation in assignment3.

    num_train = X.shape[0]
    num_class = W.shape[1]
    for i in range(num_train):
        score = X[i] @ W  # [1, C]
        loss -= score[y[i]]
        c = np.max(score)
        sum_exp = np.sum(np.exp(score - c))
        log_sum_exp = np.log(sum_exp)
        mask = np.argmax(score)
        dW[:, mask] += X[i, :]
        loss += c + log_sum_exp  # logsumexp trick for solving numeric instability.
        dW[:, y[i]] -= X[i, :]
        d_log = 1 / sum_exp
        d_exp = np.exp(score - c) * d_log
        for j in range(num_class):
            dW[:, mask] -= d_exp[j] * X[i, :]
            dW[:, j] += d_exp[j] * X[i, :]

    loss /= num_train
    loss += reg * np.sum(W * W)  # add regularization.
    dW /= num_train
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # write by samrito
    # log_sum_exp is not a good method for numeric stability here see the other implementation in assignment3.

    num_train = X.shape[0]
    num_class = W.shape[1]
    score = X @ W  # [N, C]
    upper = -score[np.arange(num_train), y]  # [N, ]
    c = np.max(score, axis=1)  # [N, ]
    mask = np.argmax(score, axis=1)  # [N, ]
    exp = np.exp(score - c.reshape(-1, 1))  # [N, C]
    sum_exp = np.sum(exp, axis=1)  # [N, ]
    log_sum_exp = np.log(sum_exp)  # [N, ]
    loss += np.sum(upper + c + log_sum_exp)
    loss /= num_train
    loss += reg * np.sum(W * W)

    d_upper, d_c, d_log_sum_exp = 1.0 / num_train, 1.0 / num_train, 1.0 / num_train
    select_gate = np.zeros_like(score)  # [N, C]
    select_gate[np.arange(num_train), y] = 1
    dW -= X.T @ (select_gate * d_upper)
    d_sum_exp = d_log_sum_exp / sum_exp  # [N, ]
    d_before_exp = d_sum_exp.reshape(-1, 1) * exp  # [N, C]
    dW += X.T @ d_before_exp

    d_c += np.sum((-1) * d_before_exp, axis=1)  # [N, ]
    max_gate = np.zeros_like(score)  # [N, C]
    max_gate[np.arange(num_train), mask] = 1
    dW += X.T @ (max_gate * d_c.reshape(-1, 1))

    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
