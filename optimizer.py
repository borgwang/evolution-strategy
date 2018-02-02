import numpy as np


class Optimizer(object):
    def __init__(self, num_params, epsilon=1e-8):
        self.dim = num_params
        self.epsilon = epsilon
        self.t = 0
        self.stepsize = None

    def compute_step(self, origin_g, stepsize):
        self.t += 1
        grads = self._compute_grads(origin_g, stepsize)
        ratio = np.linalg.norm(grads) / (np.linalg.norm(grads) + self.epsilon)
        infos = {'ratio': ratio, 'max': np.max(grads), 'min': np.min(grads),
                 'mean': np.mean(grads), 'std': np.std(grads)}
        return grads, infos

    def _compute_grads(self, origin_g):
        raise NotImplementedError


class BasicSGD(Optimizer):
    def __init__(self, num_params):
        Optimizer.__init__(self, num_params)

    def _compute_grads(self, origin_g, stepsize):
        step = -stepsize * origin_g
        return step


class SGD(Optimizer):
    def __init__(self, num_params, momentum=0.9):
        Optimizer.__init__(self, num_params)
        self.v = np.zeros(self.dim, dtype=np.float32)
        self.momentum = momentum

    def _compute_grads(self, origin_g, stepsize):
        self.v = self.momentum * self.v + (1.0 - self.momentum) * origin_g
        step = - stepsize * self.v

        return step


class Adam(Optimizer):
    def __init__(self, num_params, beta1=0.99, beta2=0.999):
        Optimizer.__init__(self, num_params)
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = np.zeros(self.dim, dtype=np.float32)
        self.v = np.zeros(self.dim, dtype=np.float32)

    def _compute_grads(self, origin_g, stepsize):
        a = stepsize * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * origin_g
        self.v = self.beta2 * self.v + (1 - self.beta2) * (origin_g * origin_g)
        step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step
