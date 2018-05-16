"""
rmax
"""

import sys
import numpy
import tensorflow as tf
import replay_memory

class DropoutMLP(object):
    """
    mlp class
    """

    def  __init__(self, candidates, dropout_prob=0.7, batch_size=64, replay_memory_size=10000):
        self._candidates = [candidates[i] for i in range(len(candidates))]
        self._candidate_number = len(candidates) 
        self._dimension = len(candidates[0])
        self._droput_prob = dropout_prob
        self._batch_size = batch_size
        self._replay_memory_size = replay_memory_size
        self._replay_memory = replay_memory.ReplayMemory(replay_memory_size, [1, self._dimension, self._dimension, 1])
        self._count = 0

        # define a MLP
        self.nid = tf.placeholder(tf.int32, [None, 1], name='nid')
        self.doc = tf.placeholder(tf.float32, [None, 5], name='doc')
        self.ctx = tf.placeholder(tf.float32, [None, 5], name='ctx')
        self.label = tf.placeholder(tf.float32, [None, 1], name='label') 

        self.dpp = tf.placeholder(tf.float32, shape=(), name='droput_prob')
        self.batch = tf.placeholder(tf.int64, shape=(), name='batch')

        self.ds = tf.contrib.data.Dataset.from_tensor_slices((self.nid, self.doc, self.ctx, self.label))
        self.ds = self.ds.repeat(1).batch(self.batch)
        self.itr = self.ds.make_initializable_iterator()
        self.nxt = self.itr.get_next()

        with tf.variable_scope('nid_embedding'):
          self.nid_embs = tf.get_variable('embedding', initializer=tf.random_uniform([200, 16], -1.0, 1.0))
          self.nid_emb = tf.nn.relu(tf.reduce_sum(tf.nn.embedding_lookup(self.nid_embs, self.nxt[0]), 1))
          print >> sys.stderr, self.nid_emb.get_shape().as_list()

        with tf.variable_scope('dot'):
          self.dot = tf.reduce_sum(
              tf.multiply(tf.nn.dropout(self.nxt[1], self.dpp), tf.nn.dropout(self.nxt[2], self.dpp)), 
              1, keep_dims=True)
          print >> sys.stderr, self.dot.get_shape().as_list()

        with tf.variable_scope('FC'):
          self.weight = tf.get_variable('weight', [17, 1], tf.float32, tf.random_normal_initializer(stddev=0.05))
          self.bias = tf.get_variable('bias', [1], tf.float32, tf.constant_initializer(0.0))
          self.feats = tf.concat([self.nid_emb, self.dot], 1)
          print >> sys.stderr, self.feats.get_shape().as_list()
          self.fc = tf.matmul(self.feats, self.weight) + self.bias
          print >> sys.stderr, self.fc.get_shape().as_list()
          self.pred_score = tf.nn.sigmoid(self.fc)
          self.max_idx = tf.argmax(self.pred_score)
          print >> sys.stderr, self.pred_score.get_shape().as_list()

        with tf.variable_scope('optimizer'):
          self.t_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.nxt[3], logits=self.fc)
          print >> sys.stderr, self.t_loss.get_shape().as_list()
          self.loss = tf.reduce_mean(self.t_loss)
          print >> sys.stderr, self.loss.get_shape().as_list()
          self.optim = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def choose_samples(self, context_feature):
        nid = [[i] for i in range(self._candidate_number)]
        ctx = [context_feature for i in range(self._candidate_number)]
        label = [[1.0] for i in range(self._candidate_number)]
        self.sess.run(self.itr.initializer, feed_dict={self.nid: nid, self.doc: self._candidates, self.ctx: ctx, self.batch: self._candidate_number, self.label: label})
        nxt, score, idx = self.sess.run([self.nxt, self.pred_score, self.max_idx], feed_dict={self.dpp: self._droput_prob})
        return idx[0]

    def update_one_sample(self, context_feature, nid, reward):
        self._count += 1
        self._replay_memory.push([nid], self._candidates[nid], context_feature, [reward]) 
        train_set = self._replay_memory.draw(512)
        if not train_set:
          return
        self.sess.run(self.itr.initializer, feed_dict={self.nid: train_set[0], self.doc: train_set[1], self.ctx: train_set[2], self.label: train_set[3], self.batch: self._batch_size})
        if train_set:
          while True:
            try:
              _, nxt, t_loss = self.sess.run([self.optim, self.nxt, self.t_loss], feed_dict={self.dpp: self._droput_prob})
              print >> sys.stderr, 'loss: %f' % (numpy.sum(t_loss))
            except tf.errors.OutOfRangeError:
              break

