import sys
import random
import numpy
import math
import argparse
import tensorflow as tf
from collections import deque
from environment import contextual_bandit
from models import dnn
from models import replay_memory


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--linear', dest='is_linear_env', action='store_true')
    parser.add_argument('--no-linear', dest='is_linear_env', action='store_false')
    parser.set_defaults(is_linear_env=True)

    parser.add_argument('--lr', help='lr', type=float, default=0.001)
    parser.add_argument('--dummy_lr', help='dummy_lr', type=float, default=0.001)

    parser.add_argument('--epsilon', help='epsilon', type=float, default=0.0)
    parser.add_argument('--keep_prob', help='keep_prob', type=float, default=1.0)
    parser.add_argument('--model_num', help='model num', type=int, default=1)
    parser.add_argument('--action_dummy', help='need action dummy', type=bool, default=False)
    parser.add_argument('--input_dummy', help='need input dummy', type=bool, default=False)
    args = parser.parse_args()

    is_linear_env = args.is_linear_env
    lr = args.lr
    dummy_lr = args.dummy_lr
    epsilon = args.epsilon
    keep_prob = args.keep_prob
    model_num = args.model_num
    need_input_dummy = args.action_dummy
    need_action_dummy = args.input_dummy

    dim = 8
    action_num = 200
    context_num = 1000
    batch_size = 32
    begin_learning = 100
    max_iteration = 100000
    replay_buffer_size = 100000

    env = contextual_bandit.ContextualBandit()
    env.load('./data/bandits.npz')

    tf_config = tf.ConfigProto(
        #gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.3),
        inter_op_parallelism_threads=4,
        intra_op_parallelism_threads=4,
        )
    tf_config.gpu_options.allow_growth = True
    session =  tf.Session(config=tf_config)
    session.__enter__()

    agent = dnn.Agent(sess=session, action_num=action_num, context_dim=dim, lr=lr, keep_prob=keep_prob, model_num=model_num)

    replay_buffer = replay_memory.ReplayMemory(replay_buffer_size, [dim, 1, 1])

    # ------------run-------------
    buffer_ready = True 
    rewards = deque(maxlen=context_num)
    accumulated_reward = 0.0
    losses = []
    print_freq = 10
    for step in range(max_iteration):
        context = env.get_context()
        ctxs = numpy.array([context for _ in range(action_num)])
        aids = numpy.array([numpy.asarray([i]) for i in range(action_num)])
        if step > begin_learning:
            eps = numpy.asarray(epsilon)
        else:
            eps = numpy.asarray(1.0)

        model_id = numpy.random.randint(model_num)
        score, sel_id = agent.predict(ctxs, aids, eps, model_id=model_id)
        reward = env.get_reward(context, numpy.asscalar(sel_id), is_linear_env)
        #print context, sel_id, reward
        replay_buffer.push(context, sel_id, numpy.asarray([reward]))
        if step > begin_learning:
            #if not buffer_ready:
            #    print >> sys.stderr, 'Initialing dataset...'
            #    ds = tf.data.Dataset.range(max_iteration * 10).map(lambda x: replay_buffer.draw(batch_size))
            #    batch_samples = ds.make_one_shot_iterator().get_next()
            #    buffer_ready = True
            losses = []
            for i in range(model_num):
                ctxs, aids, labs = replay_buffer.draw(batch_size)
                wgts = numpy.ones((batch_size, 1))
                loss = agent.train(labs, ctxs, aids, wgts, model_id=i)
                losses.append(loss)
                if need_input_dummy:
                    ctxs = numpy.array([env.get_context() for _ in range(batch_size)])
                    aids = numpy.random.randint(action_num, size=(batch_size, 1))
                    wgts = numpy.full((batch_size, 1), dummy_lr)
                    labs = numpy.ones((batch_size, 1))
                    loss = agent.train(labs, ctxs, aids, wgts, model_id=i)
                    losses.append(loss)
                if need_action_dummy:
                    ctxs, _, _ = replay_buffer.draw(batch_size)
                    aids = numpy.random.randint(action_num, size=(batch_size, 1))
                    wgts = numpy.full((batch_size, 1), dummy_lr)
                    labs = numpy.ones((batch_size, 1))
                    loss = agent.train(labs, ctxs, aids, wgts, model_id=i)
                    losses.append(loss)


        step_plus = step + 1
        rewards.append(reward)
        accumulated_reward += reward
        if step_plus % print_freq == 0:
            print >> sys.stderr, losses
            print '{}\tlr={}\tdummy_lr={}\tepsilon={}\tdropout={}\tmodel_num={}\tact_dummy={}\tinput_dummy={}\tacc_rwd={}\tavg_rwd={}\tlinear_env={}'.format(
                step_plus,
                lr,
                dummy_lr,
                epsilon,
                keep_prob,
                model_num,
                need_action_dummy,
                need_input_dummy,
                accumulated_reward,
                float(sum(rewards)) / len(rewards),
                is_linear_env)


if __name__ == '__main__':
    main()

