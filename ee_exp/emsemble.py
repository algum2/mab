"""
emsemble
"""

import sys
import numpy
import collections

import rmax
import replay_memory

class Emsemble(object):
    """
    emsemble
    """
    def __init__(self, candidates, emsemble_num=10,dropout_prob=1, batch_size=64, replay_memory_size=10000):
        self._candidates = [candidates[i] for i in range(len(candidates))]
        self._emsemble_num = emsemble_num
        self._candidate_number = len(candidates)
        self._dimension = len(candidates[0])
        self._droput_prob = dropout_prob
        self._batch_size = batch_size
        self._replay_memory_size = replay_memory_size
        self._replay_memory = replay_memory.ReplayMemory(replay_memory_size, [1, self._dimension, self._dimension, 1])
        self._count = 0
        self._model = [rmax.Rmax(self._candidates) for i in range(self._emsemble_num)]

    def choose_samples(self, context_feature):
      idx_list = []
      for model in self._model:
          idx_list.append(model.choose_samples(context_feature))
      idx_dict = collections.Counter(idx_list)
      idx = max(idx_dict.items(), key=lambda x: x[1])[0]
      return idx

    def update_one_sample(self, context_feature, nid, reward):
        self._replay_memory.push([nid], self._candidates[nid], context_feature, [reward]) 
        for model in self._model:
            train_set = self._replay_memory.draw(512)
            model.update_one_sample(context_feature, nid, reward, model_type='emsemble', train_set=train_set)

