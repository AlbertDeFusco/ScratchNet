import numpy as np
from scipy.special import logsumexp
from pandas import get_dummies

class CrossEntropy(object):
    def __call__(self, y, T):
        return -(T * np.log(y) + (1 - T) * np.log(1 - y)).mean()

    def gradient(self, y, T):
        return (y - T) / ((y) * (1 - y)) / y.shape[0]

class MSE(object):
     def __call__(self, y, T):
        return np.sum((y - T)*(y - T))/(y.shape[0])
     def gradient(self, y, T):
        return  ( y - T) / y.shape[0]

class CrossEntropyWithLogits(object):
    ## more numerically stable __call__ and gradient method
    def __call__(self, y, T):
        return -( T*y - np.log( np.exp(y) + 1.0)).mean()
    def gradient(self, y, T):
        s = np.exp(y)
        s /= (1 + s)
        return (s - T) / y.shape[0]

class SoftmaxCrossEntropyWithLogits(object):
    def __call__(self, y, T):
        ## y is logits from previous layers and
        ## T is the multiclass labels and should range in value 0 to k
        ## with k + 1 equal to total number of unique labels
        ym = y.max(1)[:, np.newaxis]
        z = y-ym
        zt = z[np.arange(T.shape[0]), T]
        return -(zt - logsumexp(z, 1)).mean(0)
    def gradient(self, y, T):
        ym = y.max(1)[:, np.newaxis]
        eym = np.exp(y-ym)
        sm = eym / eym.sum(1)[:,np.newaxis]
        sm[np.arange(T.shape[0]), T] -= 1
        return sm / y.shape[0]
