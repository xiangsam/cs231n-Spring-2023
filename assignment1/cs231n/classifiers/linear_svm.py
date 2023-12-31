from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, y[i]] = dW[:, y[i]] - X[i, :]  # by samrito
                dW[:, j] = dW[:, j] + X[i, :]  # by samrito

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW = dW / num_train  # by samrito

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW = dW + reg * 2 * W  # by samrito

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass  # done

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]  # by samrito
    num_train = X.shape[0]  # by samrito
    scores = X @ W  # [N, C] by samrito
    correct_class_score = scores[np.arange(num_train), y]  # [N] by samrito
    margin = np.maximum(0, scores - correct_class_score.reshape(-1, 1) +
                        1)  # note delta = 1 by samrito
    margin[np.arange(num_train), y] = 0  # [N, C]
    loss += np.sum(margin)
    loss /= num_train

    loss += reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    max_gate = np.zeros_like(margin)  # by samrito
    max_gate[margin > 0] = 1  # by samrito
    count = np.sum(max_gate, axis=1)  # by samrito
    max_gate[np.arange(num_train),
             y] = -count  # [N, C] for correct class. by samrito
    dW = X.T @ max_gate  # [D, C] by samrito
    dW = dW / num_train  # by samrito
    dW = dW + reg * 2 * W  # by samrito

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


if __name__ == '__main__':
    np.random.seed(40)
    W = np.random.randn(5, 3)
    X = np.random.randn(2, 5)
    y = np.array([0, 1])
    reg = 1e-1
    print(svm_loss_naive(W, X, y, reg))
    print(svm_loss_vectorized(W, X, y, reg))
