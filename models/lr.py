import sys
import random
import numpy
import math


class LinearUCB(object):
    """
    A Linear UCB Bandits Algorithm
    """

    def  __init__(self, action_num, context_dim, alpha):
        self.action_num = action_num
        self.context_dim = context_dim
        self.alpha = alpha

        self._disp_matrix = [numpy.eye(self.context_dim + 1, dtype='float32') \
            for i in range(self.action_num)]
        self._clk_matrix = [numpy.zeros((self.context_dim + 1,), dtype='float32') \
            for i in range(self.action_num)]

    def choose_samples(self, context):
        sel_idx = -1
        max_priority = -1e10
        x = numpy.concatenate(
            [context, numpy.asarray([1.0], dtype='float32')],
            axis=0)
        for idx in range(self.action_num):
            d_inv = numpy.linalg.inv(self._disp_matrix[idx])
            tmp = numpy.matmul(d_inv, x)
            est = numpy.dot(tmp, self._clk_matrix[idx])
            unc = numpy.dot(tmp, x)
            priority = est + self.alpha * numpy.sqrt(unc)
            if(max_priority < priority):
                max_priority = priority
                sel_idx = idx
        return sel_idx

    def update_one_sample(self, context, aid, reward, weight=1.0):
        x = numpy.expand_dims(numpy.concatenate(
            [context, numpy.asarray([1.0], dtype='float32')],
            axis=0), 1)
        self._disp_matrix[aid] += numpy.matmul(x, x.transpose()) * weight
        self._clk_matrix[aid] += reward * numpy.squeeze(x) * weight
        
