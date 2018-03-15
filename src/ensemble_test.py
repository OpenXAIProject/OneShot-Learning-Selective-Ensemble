import tensorflow as tf
import numpy as np
import argparse

import _init_paths
from utils import Avg, tf_acc, to1hot, cross_entropy, get_proto_model_output
from networks import Baseline_CNN
from dataset import MiniImageNet
from episode import Episode_Generator

def parse_args():
    parser = argparse.ArgumentParser(description='baseline method')
    parser.add_argument('--n', dest='nway', default=5)
    parser.add_argument('--k', dest='kshot', default=5)
    parser.add_argument('--q', dest='qnum', default=15)
    parser.add_argument('-td', dest='test_dataset',
            default=['imgnet'])
    parser.add_argument('-mn', dest='model_names',
            default=[
                'p84_5w5s_imgnet',
                'p84_5w5s_cifar',
#                'p84_5w1s_mnist',
                'b84_imgnet',
                'b84_cifar',
#                'b84_mnist'
                ])
#                'bline_cifar',
#                'bline_mnist',
#                'proto_imgnet',
#                'proto_cifar',
#                'proto_mnist'])
    parser.add_argument('--ss', dest='initial_step', default=0)
    parser.add_argument('--ms', dest='max_iter', default=500)
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = parse_args()
    print ('-'*50)
    print ('args::')
    for arg in vars(args):
        print ('%15s : %s'%(arg, getattr(args, arg)))
    print ('-'*50)
    nway = args.nway
    kshot = args.kshot
    qnum = args.qnum

    sx_ph = tf.placeholder(tf.float32, [nway*kshot,None,None,3])
    qx_ph = tf.placeholder(tf.float32, [nway*qnum,None,None,3])
    qy_ph = tf.placeholder(tf.float32, [nway*qnum,nway])
#    isTr = tf.constant(False)
    isTr = True

    models = [Baseline_CNN(model) for model in args.model_names]
    logits = []
    for model in models:
        logit = get_proto_model_output(model, sx_ph, qx_ph, nway, isTr)
        logits.append(logit)
    pred = tf.reduce_sum(logits, axis=0)
    acc = tf_acc(qy_ph, pred)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    ep_gen = []
    for dset in args.test_dataset:
        ep = Episode_Generator(nway, kshot, dset, phase='test_test')
        ep_gen.append(ep)

    for model in models:
        model.saver_init()
        model.saver_load(sess)

    avg = Avg(desc=['train acc'])

    for j in range(5):
        f_l = .0
        a_l = .0
        for i in range(1+args.initial_step, 1+args.max_iter):
            rnd_dset = np.random.randint(len(ep_gen))
            sx, sy, qx, qy = ep_gen[rnd_dset].next_batch(qnum)
            fd = {sx_ph:sx, qx_ph:qx, qy_ph: to1hot(qy, nway)}

    #        loss_lists = sess.run(losses, fd)
     #       acc_lists = sess.run(acces, fd)
            a = sess.run(acc, fd)
#            print (a)
            avg.add( a, 0)
        avg.show(j)
