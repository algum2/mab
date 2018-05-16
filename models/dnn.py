#! /bin/env python
# encoding=utf8

import sys
import numpy
import math
import random
from collections import namedtuple
import tensorflow as tf
import replay_memory
import tensorflow.contrib.layers as layers


class Agent(object):
    """
    Agent
    """

    def  __init__(self, sess, action_num, context_dim, lr=0.001, keep_prob=1.0, model_num=1, scope='MLP'):
        self.action_num = action_num
        self.context_dim = context_dim
        self.sess = sess
        self.models = []
        Model = namedtuple('Model', ['ctxs', 'aids', 'wgts', 'labels', 'eps', 'pred', 'act', 'loss', 'optim'])
        #self.graph = tf.Graph()
        #with self.graph.as_default():
        for i in range(model_num):
            with tf.variable_scope('MLP_{}'.format(i)):
              with tf.variable_scope('input'):
                  ctxs = tf.placeholder(tf.float32, [None, self.context_dim], name='ctxs')
                  aids = tf.placeholder(tf.int32, [None, 1], name='aids')
                  wgts = tf.placeholder(tf.float32, [None, 1], name='wgts')
                  labels = tf.placeholder(tf.float32, [None, 1], name='labels')
                  eps = tf.placeholder(tf.float32, (), name='epsilon')

              with tf.variable_scope('embedding'):
                  aid_embs = tf.get_variable('embeddings', initializer=tf.random_uniform([action_num, 16], -1.0, 1.0))
                  aid_emb = tf.nn.relu(tf.reduce_sum(tf.nn.embedding_lookup(aid_embs, aids), 1))
                  print >> sys.stderr, aid_emb.get_shape().as_list()

              with tf.variable_scope('concat'):
                  concat = tf.concat([ctxs, aid_emb], 1)
                  print >> sys.stderr, concat.get_shape().as_list()

              with tf.variable_scope('kernel'):
                  kernel = layers.fully_connected(concat, num_outputs=32, activation_fn=tf.nn.tanh)

              with tf.variable_scope('fc1'):
                  out = layers.fully_connected(kernel, num_outputs=32, activation_fn=tf.nn.relu)
                  fc1 = tf.nn.dropout(out, keep_prob=keep_prob)
                  print >> sys.stderr, fc1.get_shape().as_list()

              with tf.variable_scope('fc2'):
                  out = layers.fully_connected(fc1, num_outputs=32, activation_fn=tf.nn.relu)
                  fc2 = tf.nn.dropout(out, keep_prob=keep_prob)
                  print >> sys.stderr, fc2.get_shape().as_list()

              with tf.variable_scope('head'):
                  head = layers.fully_connected(fc2, num_outputs=1, activation_fn=None)
                  print >> sys.stderr, head.get_shape().as_list()

              with tf.variable_scope('optimizer'):
                  losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=head)
                  print >> sys.stderr, 'losses: ', losses.get_shape().as_list()
                  loss = tf.reduce_mean(losses * wgts)
                  print >> sys.stderr, 'loss: ', loss.get_shape().as_list()
                  optim = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

              with tf.variable_scope('predict'):
                  pred = tf.nn.sigmoid(head)
                  greedy_act = tf.argmax(pred)
                  print >> sys.stderr, 'greedy_act: ', greedy_act.get_shape().as_list()
                  random_act = tf.random_uniform((1,), minval=0, maxval=action_num, dtype=tf.int64)
                  print >> sys.stderr, 'random_act: ', random_act.get_shape().as_list()
                  explore = tf.random_uniform((1,), minval=0, maxval=1, dtype=tf.float32) < eps
                  print >> sys.stderr, 'explore: ', explore.get_shape().as_list()
                  act = tf.where(explore, random_act, greedy_act)
                  print >> sys.stderr, 'act: ', act.get_shape().as_list()
            self.models.append(Model(ctxs, aids, wgts, labels, eps, pred, act, loss, optim))
            print >> sys.stderr, self.models[-1]
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

    def predict(self, ctxs, aids, eps, model_id=1):
        model = self.models[model_id]
        score, aid = self.sess.run([model.pred, model.act], 
            feed_dict={model.ctxs: ctxs, model.aids: aids, model.eps: eps})
        return score, aid

    def train(self, labels, ctxs, aids, wgts, model_id=1):
        model = self.models[model_id]
        loss, _ = self.sess.run([model.loss, model.optim], 
            feed_dict={model.ctxs: ctxs, model.aids: aids, model.wgts: wgts, model.labels: labels})
        return loss


if __name__ == '__main__':
    tf_config = tf.ConfigProto(
        #gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.3),
        inter_op_parallelism_threads=4,
        intra_op_parallelism_threads=4,
        )
    tf_config.gpu_options.allow_growth = True
    session =  tf.Session(config=tf_config)
    agent = Agent(session, 200, 8)

