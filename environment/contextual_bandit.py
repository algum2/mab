import sys
import random
import numpy


class ContextualBandit(object):
    """
    Contextual Bandits Environment Initialize with dimention and average
    """

    def __init__(self):
        """
        Need to explicit initialization
        """
        self._ready = False

    def make(self, act_num=200, ctx_num=1000, dim=8, scale=1.0):
        """
        Generate action features
        """
        self._act_num = act_num
        self._ctx_num = ctx_num
        self._dim = dim
        self._scale = scale 
        self._actions = {}
        self._qualities = {}
        self._biases = {}
        self._ctxs = {}
        for i in range(act_num):
            self._actions[i] = self._gen_normal_vec(dim)
            self._qualities[i] = numpy.random.exponential(scale)
            self._biases[i] = numpy.random.normal()
        for i in range(ctx_num):
            self._ctxs[i] = self._gen_normal_vec(dim)
        self._ready = True

    def dump(self, path):
        """
        Dump env to file
        """
        if not self._ready:
          self.make()
        numpy.savez(path, actions=self._actions, qualities=self._qualities, biases=self._biases, scale=self._scale, contexts=self._ctxs)

    def load(self, path):
        """
        Get env from file
        """
        data = numpy.load(path)
        self._actions = data['actions'].item()
        self._qualities = data['qualities'].item()
        self._biases = data['biases'].item()
        self._scale = data['scale'].item()
        self._ctxs = data['contexts'].item()
        self._act_num = len(self._actions)
        self._ctx_num = len(self._ctxs)
        if self._act_num > 0:
            self._dim = len(self._actions[0])
        else:
            self._dim = 0
        self._ready = True

    def get_context(self):
        """
        Generate contextual features
        """
        idx = numpy.random.randint(0, self._ctx_num)
        return self._ctxs[idx]

    def get_reward(self, context, aid, islinear=True):
        """
        Args:
            context: numpy array, context feature
            candidate: int, id of candidates
            islinear: simulate linear or nonlinear env
            return  click or no click
        """
        if islinear:
            ctr = self._get_linear_ctr(context, aid)
        else:
            ctr = self._get_nonlinear_ctr(context, aid)
        #print(ctr)
        return random.random() < ctr

    def _gen_normal_vec(self, dim):
        """
        Generate a random normalized vector
        """
        vec = numpy.random.normal(size=dim)
        return 1.0 / numpy.linalg.norm(vec) * vec

    def _get_linear_ctr(self, context, aid):
        """
        Args:
            context: numpy array, context feature
            candidate: int, id of candidates
        Return:
            linear ctr
        """
        out = numpy.dot(self._actions[aid], context)
        ctr = 0.5 + out / 2
        return ctr

    def _get_nonlinear_ctr(self, context, aid):
        """
        Args:
            context: numpy array, context feature
            candidate: int, id of candidates
        Return:
            nonlinear ctr
        """
        #out = numpy.dot(self._actions[aid], context) * self._qualities[aid] + self._biases[aid]
        out = numpy.dot(self._actions[aid], context) * 2
        ctr = 1. / (1 + numpy.exp(-out))
        return ctr


if __name__ == '__main__':
    cb = ContextualBandit()
    #cb.dump('./data/bandits.npz')
    cb.load('./data/bandits.npz')
    ctx = cb.get_context()
    for i in range(200):
        rwd = cb.get_reward(ctx, i)
        rwd1 = cb.get_reward(ctx, i, False)

