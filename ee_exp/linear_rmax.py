"""
linear rmax
"""

import sys
import random
import numpy
import math


class LinearRmax(object):

    def  __init__(self, candidates, env, alpha, typ='inverse_t'):
        self._type = typ
        self._env = env
        self._count = 0
        self._candidates = candidates
        self._candidate_number = len(candidates) 
        self._dimension = len(candidates[0])
        self._alpha = alpha
        self._disp_matrix = [numpy.eye(self._dimension + 1, dtype='float32') \
            for i in range(self._candidate_number)]
        self._clk_matrix = [numpy.zeros((self._dimension + 1,), dtype='float32') \
            for i in range(self._candidate_number)]

        self._fake_disp_matrix = [numpy.zeros((self._dimension + 1, self._dimension + 1), dtype='float32') \
            for i in range(self._candidate_number)]
        self._fake_clk_matrix = [numpy.zeros((self._dimension + 1,), dtype='float32') \
            for i in range(self._candidate_number)]

    def choose_samples(self, context_feature):
        sel_idx = -1
        max_priority = -1.0
        x = numpy.concatenate(
            [context_feature, numpy.asarray([1.0], dtype='float32')],
            axis=0)
        for idx in range(self._candidate_number):
            d_inv = numpy.linalg.inv(self._disp_matrix[idx] + self._fake_disp_matrix[idx])
            tmp = numpy.matmul(d_inv, x)
            priority = numpy.dot(tmp, self._clk_matrix[idx] + self._fake_clk_matrix[idx])
            if(max_priority < priority):
                max_priority = priority
                sel_idx = idx
        return sel_idx

    def update_one_sample(self, context_feature, candidate_id, reward):
        self._count += 1
        x = numpy.expand_dims(numpy.concatenate(
            [context_feature, numpy.asarray([1.0], dtype='float32')],
            axis=0), 1)
        self._disp_matrix[candidate_id] += numpy.matmul(
            x, x.transpose())
        self._clk_matrix[candidate_id] += reward * numpy.squeeze(x)
        
        self._fake_disp_matrix = [numpy.zeros((self._dimension + 1, self._dimension + 1), dtype='float32') \
            for i in range(self._candidate_number)]
        self._fake_clk_matrix = [numpy.zeros((self._dimension + 1,), dtype='float32') \
            for i in range(self._candidate_number)]
        if self._type == 'inverse_t':
          fake_ = self._alpha * 1.0 / self._count
        elif self._type == 'log_t':
          fake_ = self._alpha * math.log(self._count)
        elif self._type == 'sqrt_log_t':
          fake_ = self._alpha * math.sqrt(math.log(self._count))
        fnum = int(fake_)
        if random.random() < fake_ % 1:
          fnum += 1
        print >> sys.stderr, self._count, fnum, fake_
        for i in range(self._candidate_number):
          for _ in range(fnum):
            x = numpy.expand_dims(numpy.concatenate(
                [self._env.gen_context(), numpy.asarray([1.0], dtype='float32')], axis=0), 1)
            self._fake_disp_matrix[i] += numpy.matmul(x, x.transpose())
            self._fake_clk_matrix[i] += 1.0 * numpy.squeeze(x)



