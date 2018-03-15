import tensorflow as tf
import numpy as np
import argparse

import _init_paths
from utils import Avg, tf_acc, to1hot, cross_entropy
from networks import Baseline_CNN
from dataset import MiniImageNet, Cifar, MNIST, SubDataSet

def parse_args():
    parser = argparse.ArgumentParser(description='baseline method')
    parser.add_argument('--d', dest='dataset', default='imgnet')
    parser.add_argument('--m', dest='model_name', default='b84_cifar')
    parser.add_argument('--ss', dest='initial_step', default=0)
    parser.add_argument('--mi', dest='max_iter', default=None)
    parser.add_argument('--me', dest='max_epoch', type=int, default=100)
    parser.add_argument('--bs', dest='batch_size', default=64)
    parser.add_argument('--lr', dest='initial_lr', default=1e-3)
    parser.add_argument('--fm', dest='from_ckpt', default=True)
    parser.add_argument('--nw', dest='nway', type=int, default=5)
    parser.add_argument('--ks', dest='kshot', type=int, default=1)
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = parse_args()
    print ('-'*50)
    print ('args::')
    # data size : 64 * 500
    args.max_iter = int(args.max_epoch * (64 * 500 / args.batch_size))
    for arg in vars(args):
        print ('%15s : %s'%(arg, getattr(args, arg)))
    print ('-'*50)

    x_ph = tf.placeholder(tf.float32, [None,None,None,3])
    y_ph = tf.placeholder(tf.float32, [None,100])
    lr_ph = tf.placeholder(tf.float32)
    isTr = tf.constant(True)

    model = Baseline_CNN(args.model_name)
    feature = model.network(x_ph, isTr)
    logits = model.get_logit(feature)
    y = tf.nn.softmax(logits)

    loss = cross_entropy(y_ph, y)
    opt = tf.train.AdamOptimizer(lr_ph)
    train_op = opt.minimize(loss)
    update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_step = tf.group(train_op, update_op)
    acc = tf_acc(y_ph, y)


    logit2 = model.get_another_logit(feature)
    y2 = tf.nn.softmax(logit2)
    loss2 = cross_entropy(y_ph, y2)
    train_step2 = tf.train.AdamOptimizer(lr_ph, name='opt2').minimize(loss2)
    acc2 = tf_acc(y_ph, y2)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    avg = Avg(desc=['train acc', 'train loss', 'lr', 'test acc'])

#    if args.dataset=='imgnet':
#        train_data = MiniImageNet('train')
#        val_data = MiniImageNet('train_val')
#    if args.dataset=='cifar':
#        train_data = Cifar('cifar100', 'train')
#        val_data = Cifar('cifar100', 'test')
#    if args.dataset=='mnist':
#        train_data = MNIST('notMNIST', 'train')
#        val_data = MNIST('notMNIST', 'test')
#
#
    model.saver_init()
    if args.from_ckpt:
        model.saver_load(sess)
#
#    for i in range(1+args.initial_step, 1+args.max_iter):
#        batch_x, batch_y = train_data.next_batch(args.batch_size)
#        lr = (1e-0 if i < args.max_iter * 2.0 / 3.0 else 1e-1) * args.initial_lr
#        fd = {x_ph: batch_x, y_ph: batch_y, lr_ph: lr}
#
#        if sess.run(isTr):
#            p1, p2, _ = sess.run([acc, loss, train_step], fd)
#        else:
#            p1, p2 = sess.run([acc, loss], fd)
#        avg.add(p1, 0)
#        avg.add(p2, 1)
#        if i % 100 == 0:
#            batch_x, batch_y = val_data.next_batch(128)
#            fd = {x_ph: batch_x, y_ph: batch_y}
#            avg.add(1.0/lr, 2)
#            avg.add(sess.run(acc, fd), 3)
#            avg.show(i)
#
#        if i % 5000 == 0:
#            avg.description()
#            model.saver_save(sess)

    # fine tune the model
    meta_test_Dtrain = MiniImageNet('test')
    meta_test_Dtest = MiniImageNet('test_val')

    if args.dataset=='imgnet':
        meta_test_Dtrain = MiniImageNet('test')
        meta_test_Dtest = MiniImageNet('test_val')
    if args.dataset=='cifar':
        meta_test_Dtrain = Cifar('cifar10', 'train')
        meta_test_Dtest = Cifar('cifar10', 'test')
    if args.dataset=='mnist':
        meta_test_Dtrain = MNIST('MNIST', 'train')
        meta_test_Dtest = MNIST('MNIST', 'test')

#    random_class_nway = np.random.choice(20,
#            size=args.nway, replace=False)
#    random_class_nway = [2,6,3,14,11]
    random_class_nway = [1,2,3,4,5]
    Dtrain = SubDataSet(random_class_nway,
            (meta_test_Dtrain.x, meta_test_Dtrain.y), k=500)
    Dtest = SubDataSet(random_class_nway,
            (meta_test_Dtest.x, meta_test_Dtest.y))
    for i in range(1, 5001):
        lr = 1e-4
        batch_x, batch_y = Dtrain.next_batch(32)
        fd = {x_ph: batch_x, y_ph:batch_y, lr_ph: lr}
        p1, p2, _  = sess.run([acc2, loss2, train_step2], fd)
        avg.add(p1, 0)
        avg.add(p2, 1)
        if i % 100 == 0:
            batch_x, batch_y = Dtest.next_batch(100)
            fd = {x_ph: batch_x, y_ph:batch_y, lr_ph: lr}
            p1 = sess.run(acc2, fd)
            avg.add(p1, 2)
            avg.show(i)
