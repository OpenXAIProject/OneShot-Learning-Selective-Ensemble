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
    parser.add_argument('--n', dest='nway', default=20)
    parser.add_argument('--k', dest='kshot', default=1)
    parser.add_argument('--q', dest='qnum', default=15)
    parser.add_argument('--d', dest='dataset', default='imgnet')
    parser.add_argument('--m', dest='model_name', default='xai_imgnet')
    parser.add_argument('--ss', dest='initial_step', default=0)
    parser.add_argument('--ms', dest='max_iter', default=10000)
    parser.add_argument('--lr', dest='initial_lr', default=1e-3)
    parser.add_argument('--fm', dest='from_ckpt', default=False)
    parser.add_argument('--me', dest='max_epoch', type=int, default=150)
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
    lr_ph = tf.placeholder(tf.float32)
    #isTr = tf.constant(True)
    isTr = True
    model = Baseline_CNN(args.model_name)

    y = get_proto_model_output(model, sx_ph, qx_ph, nway, isTr, trainable=True)
    loss = cross_entropy(qy_ph, y)
    opt = tf.train.AdamOptimizer(lr_ph)
    train_op = opt.minimize(loss)
    update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_step = tf.group(train_op, update_op)
    acc = tf_acc(qy_ph, y)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    avg = Avg(desc=['train acc', 'train loss', 'lr'])

    ep_gen = Episode_Generator(nway, kshot, args.dataset, phase='train')
    model.saver_init()
    if args.from_ckpt:
        model.saver_load(sess)

    max_step = int(500 * 64 / nway / qnum)
    for j in range(args.max_epoch):
        lr = 1e-3 if j < 100 else 1e-4
        for i in range(max_step):
            sx, sy, qx, qy = ep_gen.next_batch(qnum)
            fd = {sx_ph:sx, qx_ph:qx, qy_ph:to1hot(qy, nway), lr_ph:lr}
            p1, p2, _ = sess.run([acc, loss, train_step], fd)

            avg.add(p1, 0)
            avg.add(p2, 1)
            if (i+1) % 100 == 0:
                avg.show(j)
        if j % 10 == 0:
            avg.description()
        if j == args.max_epoch - 1:
            model.saver_save(sess)

#    ep_gen = Episode_Generator(nway, kshot, args.dataset, phase='test_train')
#    lr = 1e-3
#    for i in range(5000):
#        sx, sy, qx, qy = ep_gen.next_batch(qnum)
#        fd = {sx_ph:sx, qx_ph:qx, qy_ph:to1hot(qy, nway), lr_ph:lr}
#        p1,p2,_ = sess.run([acc, loss, train_step], fd)
#
#        avg.add(p1, 0)
#        avg.add(p2, 1)
#        if (i+1) % 50 == 0 :
#            avg.show(i)
#
#    ep_gen = Episode_Generator(20, kshot, args.dataset, phase='test_test')
#    for i in range(1000):
#        sx, sy, qx, qy = ep_gen.next_batch(qnum)
#        fd = {sx_ph:sx, qx_ph:qx, qy_ph:to1hot(qy, 20)}
#        p1,p2 = sess.run([acc, loss], fd)
#
#        avg.add(p1, 0)
#        avg.add(p2, 1)
#        if (i+1) % 50 == 0 :
#            avg.show(i)
