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
  # TODO: Compute the softmax loss and its gradient using explicit loops.    #
  # Store the loss in loss and the gradient in dW. If you are not careful    #
  # here, it is easy to run into numeric instability. Don't forget the      #
  # regularization!                                        #
  #############################################################################
  for i in range(X.shape[0]): # training examples
    score = np.dot(X[i,:], W)
    max_score = np.max(score)
    score -= max_score
    score = np.exp(score)
    loss -= np.log(score[y[i]] / np.sum(score))
    for j in range(W.shape[1]): # classes
      if j == y[i]:
        dW[:,j] += -X[i,:].T + (X[i,:].T * score[j]) / np.sum(score)
      else:
        dW[:,j] += (X[i,:].T * score[j]) / np.sum(score)
  #############################################################################
  #                          END OF YOUR CODE              #
  #############################################################################
  loss = loss / X.shape[0] + reg * np.sum(W*W)
  dW = dW / X.shape[0] + 2 * reg * W
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  train_num, dim = X.shape
  class_num = W.shape[1]
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful    #
  # here, it is easy to run into numeric instability. Don't forget the      #
  # regularization!                                        #
  #############################################################################
  scores = np.dot(X, W) # X N * D
  scores = scores - np.max(scores, axis=1).reshape(train_num,1) # normalize
  scores = np.exp(scores)  # do exponential  N * C
  loss = np.mean(-np.log(scores[np.arange(train_num),y] / np.sum(scores, axis=1)))
    
  coeff = np.zeros([train_num, class_num])
  coeff[np.arange(train_num), y] = 1
  
  midcalc = scores / np.sum(scores, axis=1).reshape(train_num,1)
  
  dW = (np.dot(X.T, midcalc) - np.dot(X.T, coeff)) / train_num + 2 * reg * W
  #############################################################################
  #                         END OF YOUR CODE               #
  #############################################################################

  return loss, dW

