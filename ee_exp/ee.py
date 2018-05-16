"""
ee
"""

import sys
import numpy
import math
import linear_env
import linear_ucb
import dropout
import rmax
import emsemble
import wrmax
import wrmax_ini
import wrmax_random
import wrmax_new
import wrmax_new2
import epsilon_greedy
import wrmax_sqrt_tlogt
import wrmax_inverse_t
import linear_rmax

if __name__ == '__main__':
    dim = 5
    cand_num = 200
    avg = 0.20

    env = linear_env.LinearBanditsEnv()
    #candidates = env.gen_document_candidates(cand_num, dim, avg)
    candidates = env.load_document_candidates('./data/doc.npz')

    if sys.argv[1] == 'ucb':
      alg = linear_ucb.LinearBanditsAlgorithm(candidates, alpha=0.85)
    elif sys.argv[1] == 'dropout':
      alg = dropout.DropoutMLP(candidates, dropout_prob=1.0)
    elif sys.argv[1] == 'rmax':
      d = env.gen_fake_data(20)
      alg = rmax.Rmax(candidates)
      alg.init(d)
    elif sys.argv[1] == 'emsemble':
      alg = emsemble.Emsemble(candidates, emsemble_num=10)

    elif sys.argv[1] == 'wrmax':
      alg = wrmax.Wrmax(candidates, env=env, weight= 1.2)
    elif sys.argv[1] == 'wrmax_ini':
      alg = wrmax_ini.WrmaxIni(candidates, env=env, weight= 1.4)
    elif sys.argv[1] == 'wrmax_new':
      alg = wrmax_new.WrmaxNew(candidates, env=env, weight=0.018)
    elif sys.argv[1] == 'egreedy':
      alg = epsilon_greedy.EpslionGreedy(candidates, env=env, epsilon=0.012)
    elif sys.argv[1] == 'stlogt':
      alg = wrmax_sqrt_tlogt.WrmaxSqrtTlogt(candidates, env=env, weight=0.013)
    elif sys.argv[1] == 'inverset':
      alg = wrmax_inverse_t.WrmaxInverset(candidates, env=env, weight=0.05)
    elif sys.argv[1] == 'lin_rmax':
      #alg = linear_rmax.LinearRmax(candidates, env=env, alpha=490, typ='inverse_t')
      #alg = linear_rmax.LinearRmax(candidates, env=env, alpha=0.02, typ='log_t')
      alg = linear_rmax.LinearRmax(candidates, env=env, alpha=0.55, typ='sqrt_log_t')

    elif sys.argv[1] == 'wrmax_random':
      alg = wrmax_random.WrmaxRandom(candidates, env=env, weight= 4.5)
    elif sys.argv[1] == 'wrmax_new2':
      alg = wrmax_new2.WrmaxNew2(candidates, env=env, weight=0.6, fake_prob=0.7)
    else:
      print >> sys.stderr, 'Usage: python %s [ucb|dropout|rmax|emsemble]' % sys.argv[0]
      sys.exit(1)

    avg_reward = 0.0
    total_reward = 0.0
    prt_interval = 100
    max_iteration = 100000
    for step in range(max_iteration):
        context = env.gen_context()
        sel_id = alg.choose_samples(context)
        reward = env.get_reward(context, sel_id)
        alg.update_one_sample(context, sel_id, reward)

        avg_reward += reward
        total_reward += reward
        #print '%d,%d,%d' % (step, sel_id, total_reward)

        if(step % prt_interval == 0):
          print 'sum: %d,%d,%f' % (step, total_reward, avg_reward / float(prt_interval))
          avg_reward = 0.0

