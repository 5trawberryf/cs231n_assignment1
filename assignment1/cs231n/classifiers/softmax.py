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
    N = X.shape[0]
    D = X.shape[1]
    C = W.shape[1]
    score = X.dot(W) # (N,C)
    score_exp = np.exp(score)
    softmax = score_exp / np.sum(score_exp,1).reshape(-1,1) # (N,C)
    logsoftmax = np.log(softmax)
    for i in range(N):
        j = y[i]
        loss -= logsoftmax[i][j]
        dW[np.arange(D),j] -= X[i,np.arange(D)]
        dW += softmax[i].reshape(1,-1) * X[i].reshape(-1,1)
    # loss = - logsoftmax[np.arange(N),y]
    dW /= N
    loss /= N
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W
    pass
                        
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
    N = X.shape[0]
    D = X.shape[1]
    C = W.shape[1]
    score = X.dot(W) # (N,C)
    score_exp = np.exp(score)
    softmax = score_exp / np.sum(score_exp,1).reshape(-1,1)  # (N,C)
    logsoftmax = np.log(softmax)
    dS = softmax 
    dS[np.arange(N),y] -= 1
    dW = X.transpose().dot(dS)
    loss = - np.sum(logsoftmax[np.arange(N),y])
    dW /= N
    loss /= N
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
