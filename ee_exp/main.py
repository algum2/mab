"""
linear ucb
"""

import sys
import random
import numpy
import math
import linear_env
import linear_ucb
import dropout


if __name__ == '__main__':
    dim = 5
    cand_num = 200
    avg = 0.20
    alpha = 1.0

    env = linear_env.LinearBanditsEnv()
    candidates = env.gen_document_candidates(cand_num, dim, avg)

    ucb = linear_ucb.LinearBanditsAlgorithm(candidates, alpha)
    dro = dropout.DropoutMLP(candidates, 0.7)

    avg_reward = 0.0
    prt_interval = 100
    max_iteration = 100000
    total_reward = 0.0
    for step in range(max_iteration):
        context = env.gen_context()

        sel_id = dro.choose_samples(context)
        reward = env.get_reward(context, sel_id)
        dro.update_one_sample(context, sel_id, reward)

        #sel_id = ucb.choose_samples(context)
        #reward = env.get_reward(context, sel_id)
        #ucb.update_one_sample(context, sel_id, reward) 

        step_plus = step + 1
        avg_reward += reward
        total_reward += reward

        if(step_plus % prt_interval == 0):
            print step_plus, total_reward, avg_reward / float(prt_interval)
            avg_reward = 0.0

