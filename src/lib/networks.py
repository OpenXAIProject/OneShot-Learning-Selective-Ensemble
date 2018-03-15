import tensorflow as tf
import numpy as np
from utils import conv, pool, fc, get_wb, batch_norm, layer_bn

nway = 5
kshot = 1
qnum = 10
one_shot_test = True
pretrain_trainset = False
model_name = 'baseline_nn_mini32.ckpt'
initial_step = 0

class Baseline_CNN():
    def __init__(self, name, set_type=None, loc=None):
        self.name = name
        if not loc:
            self.model_loc = '/home/mike/models/'+name+'/model.ckpt'
        else:
            self.model_loc = loc
        if set_type==None:
            set_type = name.split('_')[-1]
        if set_type=='cifar':
            self.wh = 84
        elif set_type=='imgnet':
            self.wh = 84
        elif set_type=='mnist':
            self.wh = 84

    def conv_block(self, x, name, reuse, isTr, trainable=True):
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            l = tf.layers.conv2d(x, 64, kernel_size=3, padding='SAME')
            l = tf.contrib.layers.batch_norm(l,
                    is_training=isTr, decay=.99, scale=True, center=True)
            l = tf.nn.relu(l)
            l = tf.contrib.layers.max_pool2d(l, 2)
        return l

    def network(self, in_x, isTr, reuse=False, trainable=True):
        with tf.variable_scope(self.name):
            l = tf.image.resize_images(in_x, [self.wh, self.wh])
            l = self.conv_block(l, 'c1', reuse, isTr)
            l = self.conv_block(l, 'c2', reuse, isTr)
            l = self.conv_block(l, 'c3', reuse, isTr)
            l = self.conv_block(l, 'c4', reuse, isTr)
            l = tf.contrib.layers.flatten(l)
            #l = tf.layers.dense(l, 1024, reuse=reuse)
        return l


    def _network(self, in_x, isTr, reuse=False, trainable=True):
        with tf.variable_scope(self.name):
            l = tf.image.resize_images(in_x, [self.wh, self.wh])
            l = conv('c1', [3,3,3,64], l, reuse=reuse, trainable=trainable)
            l = layer_bn('b1', l, isTr, reuse, trainable)
            l = tf.nn.relu(l)
            l = pool(l)
            l = conv('c2', [3,3,64,64], l, reuse=reuse, trainable=trainable)
            l = layer_bn('b2', l, isTr, reuse, trainable)
            l = tf.nn.relu(l)
            l = pool(l)
            l = conv('c3', [3,3,64,64], l, reuse=reuse, trainable=trainable)
            l = layer_bn('b3', l, isTr, reuse, trainable)
            l = tf.nn.relu(l)
            l = pool(l)
            l = conv('c4', [3,3,64,64], l, reuse=reuse, trainable=trainable)
            l = layer_bn('b4', l, isTr, reuse, trainable)
            l = tf.nn.relu(l)
            l = pool(l)
            l = tf.contrib.layers.flatten(l)
            #l = fc('f1', 1024, l, r=reuse, t=trainable)
        return l

    def get_logit(self, feat):
        with tf.variable_scope(self.name):
            l = tf.nn.relu(feat)
            l = fc('f3', 100, l)
        return l

    def get_another_logit(self, feat):
        with tf.variable_scope(self.name):
            l = tf.nn.relu(feat)
            l = fc('f4', 100, l)
        return l

    def saver_init(self):
        model_var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                scope=self.name+'*')
        #model_var_lists += tf.get_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES)
        self.saver = tf.train.Saver(model_var_lists)

    def saver_load(self, sess, loc=None):
        if not loc:
            print ('load model from ', self.model_loc)
            self.saver.restore(sess, self.model_loc)
        else:
            print ('load model from ', loc)
            self.saver.restore(sess, loc)


    def saver_save(self, sess):
        print ('save model at ', self.model_loc)
        self.saver.save(sess, self.model_loc)

class BidirectionalLSTM(Baseline_CNN):
    def __init__(self, name, loc=None):
        ay = 0
        self.name = name
        self.num_hidden = 128
        self.num_output_dim = 1

        self.w, self.b = get_wb(name, [self.num_hidden * 2,
            self.num_output_dim])

        if not loc:
            self.model_loc = '/home/mike/models/'+name+'/model.ckpt'
        else:
            self.model_loc = loc

    def get_output(self, x_ph, reuse=False):
        with tf.variable_scope(self.name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            x = tf.expand_dims(x_ph, 0)
            x = tf.unstack(x, axis=1)

            lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.num_hidden, forget_bias=1.0)
            lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.num_hidden, forget_bias=1.0)

            outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
                    lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)

        return tf.matmul(outputs[-1], self.w) + self.b
