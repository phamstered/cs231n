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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] += X[i].T
        dW[:,y[i]] -= X[i].T # 
  # for every exmaple x, if its margin is larger than 0,dW's j-th row is x and y-th row is -x 

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss. 
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  dW = dW / X.shape[0] + 2 * reg * W
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  scores = np.dot(X, W) # shape N * C
  nums = np.shape(X)[0] # example numbers
  ground_v = (scores[np.arange(nums), y]).reshape(nums, 1) 
  delta = np.ones(scores.shape)
  delta[np.arange(nums), y] = 0
  loss_matrix = scores - ground_v + delta # add delta to all examples
  loss_matrix[loss_matrix < 0] = 0 # if loss less than zero, set it to 0
  loss = np.mean(np.sum(loss_matrix, axis=1))   
  loss += reg * np.sum(W * W)
  #############################################################################
  # TODO:                                               #
  # Implement a vectorized version of the structured SVM loss, storing the   #
  # result in loss.                                        #
  #############################################################################
  # hinge loss in multiclass svm
  # http://www.ttic.edu/sigml/symposium2011/papers/Moore+DeNero_Regularization.pdf
  # https://ipfs.io/ipfs/QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco/wiki/Hinge_loss.html#cite_note-4   
  # http://www.mit.edu/~rakhlin/6.883/lectures/lecture05.pdf  
  #############################################################################  
  indicator = loss_matrix > 0 # N * C
  coeff = np.zeros(scores.shape)
  coeff[np.arange(X.shape[0]), y] = -np.sum(indicator, axis=1)
  coeff2 = np.zeros(scores.shape)
  coeff2[indicator] = 1
  coeff2[np.arange(X.shape[0]), y] = 0
  dW = np.dot(X.T, coeff + coeff2) / X.shape[0] + 2 * reg * W 
  #############################################################################
  #                   END OF YOUR CODE                     #
  #############################################################################


  #############################################################################
  # TODO:                                              #
  # Implement a vectorized version of the gradient for the structured SVM   #
  # loss, storing the result in dW.                             #
  #                                                   #
  # Hint: Instead of computing the gradient from scratch, it may be easier   #
  # to reuse some of the intermediate values that you used to compute the    #
  # loss.                                                                     #
  #############################################################################
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
