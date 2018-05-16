"""
A Linear Bandits Simulator Environment
"""

import sys
import random
import numpy


class LinearBanditsEnv(object):
    """
    Linear Bandits Environment
    Initialize with dimention and average
    """

    def __init__(self):
        pass

    def dump_document_candidates(self, candidate_number, dim, avg, path):
      self.gen_document_candidates(candidate_number, dim, avg)
      numpy.savez(path, candidates=self._candidates, qualities=self._qualities, biases=self._biases)

    def load_document_candidates(self, path):
      """get doc candidates from file"""
      data = numpy.load(path)
      self._candidates = data['candidates'].item()
      self._qualities = data['qualities'].item()
      self._biases = data['biases'].item()
      #print type(self._biases)

      return self._candidates

    def gen_document_candidates(self, candidate_number, dim, avg):
        """
        generating candidate documents
        """
        self._candidates = {}
        self._qualities = {}
        self._biases = {}
        for i in range(candidate_number):
            self._candidates[i] = self._gen_random_vec(dim)
            self._qualities[i] = min(0.80, numpy.random.exponential(0.80 * avg))
            self._biases[i] = min(0.20, numpy.random.exponential(0.20 * avg))
        return self._candidates

    def gen_context(self):
        """
        generating context feature
        """
        context = self._gen_random_vec(len(self._candidates[0]))
        return context

    def get_reward(self, context, candidate_id):
        """
        context: numpy array, context feature
        candidate: int, id of candidates
        return  click or no click
        """
        if(random.random() < self._get_true_ctr(context, candidate_id)):
            return 1
        return 0

    def _gen_random_vec(self, dim):
        """
        Generating a random normalized vector
        """
        vec = numpy.random.normal(size=dim)
        return 1.0 / numpy.linalg.norm(vec) * vec

    def _get_true_ctr(self, context, candidate_id):
        """
        context: numpy array, context feature
        candidate: int, id of candidates
        return True ctr
        """
        #print numpy.dot(self._candidates[candidate_id], context)
        #print self._qualities[candidate_id]
        #print self._biases[candidate_id]
        dot = numpy.dot(self._candidates[candidate_id], context)
        out = 1. / (1 + numpy.exp(-dot))
        res = out * self._qualities[candidate_id] + self._biases[candidate_id]
        return res

    def gen_fake_data(self, num):
      d = {}
      nids = []
      docs = []
      ctxs = []
      labels = []
      for cnt in range(num):
        for nid, doc in self._candidates.iteritems():
          ctx = self.gen_context()
          nids.append([nid])
          docs.append(doc)
          ctxs.append(ctx)
          labels.append([1])
      d['nid'] = numpy.array(nids, dtype=numpy.int32)
      d['doc'] = numpy.array(docs, dtype=numpy.float32)
      d['ctx'] = numpy.array(ctxs, dtype=numpy.float32)
      d['label'] = numpy.array(labels, dtype=numpy.float32)
      return d

    def gen_fake_data_step(self, num, nid):
        d = {}
        nids = []
        docs = []
        ctxs = []
        labels = []
        for cnt in range(num):
            ctx = self.gen_context()
            nids.append([nid])
            docs.append(self._candidates[nid])
            ctxs.append(ctx)
            labels.append([1])
        d['nid'] = numpy.array(nids, dtype=numpy.int32)
        d['doc'] = numpy.array(docs, dtype=numpy.float32)
        d['ctx'] = numpy.array(ctxs, dtype=numpy.float32)
        d['label'] = numpy.array(labels, dtype=numpy.float32)
        return d

if __name__ == '__main__':
  lbe = LinearBanditsEnv()
  #lbe.dump_document_candidates(200, 5, 0.20, './data/doc.npz')
  lbe.load_document_candidates('./data/doc.npz')

