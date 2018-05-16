"""
linear ucb
"""

import sys
import random
import numpy
import math


class LinearBanditsAlgorithm(object):
    """
    A Linear UCB Bandits Algorithm
    Initialize with dimension and exploration ratio
    """

    def  __init__(self, candidates, alpha):
        self._candidates = candidates
        self._candidate_number = len(candidates) 
        self._dimension = len(candidates[0])
        self._alpha = alpha
        self._disp_matrix = [numpy.eye(self._dimension + 1, dtype='float32') \
            for i in range(self._candidate_number)]
        self._clk_matrix = [numpy.zeros((self._dimension + 1,), dtype='float32') \
            for i in range(self._candidate_number)]

    def choose_samples(self, context_feature):
        sel_idx = -1
        max_priority = -1.0
        x = numpy.concatenate(
            [context_feature, numpy.asarray([1.0], dtype='float32')],
            axis=0)
        for idx in range(self._candidate_number):
            d_inv = numpy.linalg.inv(self._disp_matrix[idx])
            tmp = numpy.matmul(d_inv, x)
            est = numpy.dot(tmp, self._clk_matrix[idx])
            unc = numpy.dot(tmp, x)
            priority = est + self._alpha * unc
            #priority = est + self._alpha * math.sqrt(unc)
            if(max_priority < priority):
                max_priority = priority
                sel_idx = idx
        return sel_idx

    def update_one_sample(self, context_feature, candidate_id, reward):
        x = numpy.expand_dims(numpy.concatenate(
            [context_feature, numpy.asarray([1.0], dtype='float32')],
            axis=0), 1)
        self._disp_matrix[candidate_id] += numpy.matmul(
            x, x.transpose())
        self._clk_matrix[candidate_id] += reward * numpy.squeeze(x)
        
