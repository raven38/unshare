import numpy as np

import chainer
from chainer import functions as F
from chainer import links as L
from chainer import optimizers
from chainer import cuda


class Unsharing(object):
    def __init__(self, rate, source):
        self.rate = rate
        self.source = source

    def __call__(self, opt):
        for t_param, s_param in zip(opt.target.params(False), self.source.params(False)):
            t_data, t_grad = t_param.data, t_param.grad
            s_data = s_param.data
            xp = cuda.get_array_module(p)
            grad += self.rate * -1 * xp.absolute(t_data - s_data)
                

class MLP(chainer.Chain):
    def __init__(self):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(2, 2)
            self.l2 = L.Linear(2, 1)

    def __call__(self, x):
        h = self.l1(x)
        h = self.l2(h)
        return h


net = MLP()
optimizer = optimizers.SGD(lr=0.1)
optimizer.setup(net)

x = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
t = np.asarray([[0], [1], [1], [0]], dtype=np.int32)




                                                                                    
