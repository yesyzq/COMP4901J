import numpy as np
from random import shuffle
#from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    loss = 0.0
    dW = np.zeros_like(W)

    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i, :].dot(W)
        scores -= np.max(scores) # avoid numeric instability
        scoresExp = np.exp(scores)
        scoresNorm = scoresExp / scoresExp.sum()  
        loss += -np.log(scoresNorm[y[i]])
        for j in range(num_classes):
            dW[:, j] += -X[i] * ((j == y[i]) - scoresNorm[j])

    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)

    dW /= num_train
    dW += reg * W
        
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
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
  num_train = X.shape[0]
  scores = X.dot(W)
  scores -= np.max(scores, axis=1, keepdims=True)
  scoresExp = np.exp(scores)
  scoresNorm = scoresExp / np.sum(scoresExp, axis=1, keepdims=True)
  
  loss += -np.log(scoresNorm[np.arange(num_train), y]).sum()
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  mask = np.zeros(scores.shape)
  mask[np.arange(num_train), y] = 1
  # print(mask.shape, scoresNorm.shape)
  dW += -X.T.dot(mask - scoresNorm)
  dW /= num_train
  dW += reg * W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

