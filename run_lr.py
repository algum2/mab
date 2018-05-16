import sys
import random
import numpy
import math
import argparse
from collections import deque
from environment import contextual_bandit
from models import lr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--linear', dest='is_linear_env', action='store_true')
    parser.add_argument('--no-linear', dest='is_linear_env', action='store_false')
    parser.set_defaults(is_linear_env=True)

    parser.add_argument('--alpha', help='alpha', type=float, default=0.0)
    parser.add_argument('--dummy_lr', help='dummy_lr', type=float, default=0.0)
    args = parser.parse_args()

    is_linear_env = args.is_linear_env 
    alpha = args.alpha
    dummy_lr = args.dummy_lr
    dim = 8
    action_num = 200
    context_num = 1000

    env = contextual_bandit.ContextualBandit()
    env.load('./data/bandits.npz')

    agent = lr.LinearUCB(action_num, dim, alpha)

    #------------run----------
    rewards = deque(maxlen=context_num)
    accumulated_reward = 0.0
    print_freq = 10
    max_iteration = 100000
    for step in range(max_iteration):
        context = env.get_context()
        aid = agent.choose_samples(context)
        reward = env.get_reward(context, aid, is_linear_env)
        agent.update_one_sample(context, aid, reward)

        if dummy_lr > 0:
            context = env.get_context()
            aid = numpy.random.randint(action_num)
            agent.update_one_sample(context, aid, 1, dummy_lr)

        rewards.append(reward)
        accumulated_reward += reward
        
        stepp1 = step + 1
        if(stepp1 % print_freq == 0):
            print '{}\talpha={}\tdummy_lr={}\tislinear={}\tacc_rwd={}\tavg_rwd={}'.format(
                stepp1, 
                alpha,
                dummy_lr, 
                is_linear_env, 
                accumulated_reward, 
                float(sum(rewards)) / len(rewards))

