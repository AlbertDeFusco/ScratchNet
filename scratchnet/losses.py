import numpy as np

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
    from pandas import get_dummies
    def __call__(self, y, T):
        ym = y.max(1)[:, np.newaxis]
        T_zip_y = zip(T,y-ym)
        return -np.array([e2[e1] - logsumexp(e2) for e1,e2 in T_zip_y]).mean()
    def gradient(self, y, T):
        ym = y.max(1)[:, np.newaxis]
        eym = np.exp(y-ym)
        sm = eym / eym.sum(1)[:,np.newaxis]
        return sm - get_dummies(y)
