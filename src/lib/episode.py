import numpy as np
import h5py
from dataset import Cifar, MiniImageNet, MNIST, SubDataSet
import time

class Episode_Generator():
    def __init__(self, Nway, Kshot, dataset, phase):
        if dataset=='cifar':
            '''
            use ciafr100 to train base predictors and meta-classifier
            use cifar10 to test one-shot classification
            '''
            if phase=='train':
                self.train_data = Cifar('cifar100', 'train')
                self.class_list = [i for i in range(80)]
            elif phase=='val':
                self.train_data = Cifar('cifar100', 'train')
                self.class_list = [i for i in range(80)]
            elif phase=='test_train':
                self.train_data = Cifar('cifar10', 'train')
                self.class_list = [i for i in range(10)]
            elif phase=='test_test':
                self.train_data = Cifar('cifar10', 'test')
                self.class_list = [i for i in range(10)]
            else:
                print (name_error)

        elif dataset=='imgnet':
            '''
            use miniImagenet 'train' and 'train_val' to train base predictors
            'train' and 'train_val' contains 64 classes of mini imagenet
            use 'val' to train meta classifier. val has 16 classes
            use 'test' to test one-shot classification
            '''
            if phase=='train':
                self.train_data = MiniImageNet('train')
                self.class_list = [i for i in range(64)]
            elif phase=='val':
                self.train_data = MiniImageNet('val')
                self.class_list = [i for i in range(16)]
            elif phase=='test_train':
                self.train_data = MiniImageNet('test')
                self.class_list = [i for i in range(20)]
            elif phase=='test_test':
                self.train_data = MiniImageNet('test_val')
                self.class_list = [i for i in range(20)]
            else:
                print (name_error)

        elif dataset=='mnist':
            '''
            use notMNIST 'train' to train base predictors
            use notMNIST 'test' to train meta classifier
            use MNIST 'train' and 'test' to test one-shot classification
            '''
            if phase=='train':
                self.train_data = MNIST('notMNIST', 'train')
            elif phase=='val':
                self.train_data = MNIST('notMNIST', 'test')
            elif phase=='test_train':
                self.train_data = MNIST('MNIST', 'train')
            elif phase=='test_test':
                self.train_data = MNIST('MNIST', 'test')
            else:
                print (name_error)
            self.class_list = [i for i in range(10)]
        else:
            print (dataset_name_error)

        self.dataset = dataset
        self.phase = phase
        self.N = Nway
        self.K = Kshot
        self.train_subset = []
        for i in self.class_list:
            self.train_subset.append(SubDataSet([i],
                (self.train_data.x, self.train_data.y)))
        self.episode_set = []

    def next_batch(self, q_size):
        random_class = np.random.choice(len(self.class_list),
                size = self.N, replace=False)
        self.episode_set = []
        for v in random_class:
            self.episode_set.append(self.train_subset[v])
        for n in range(self.N):
            if n == 0:
                x, _ = self.episode_set[n].next_batch(self.K + q_size)
                Skx = x[:self.K]
                Qkx = x[self.K:]
                Sky = [n for _ in range(self.K)]
                Qky = [n for _ in range(q_size)]
            else:
                x, _ = self.episode_set[n].next_batch(self.K + q_size)
                Skxi = x[:self.K]
                Qkxi = x[self.K:]
                Skyi = [n for _ in range(self.K)]
                Qkyi = [n for _ in range(q_size)]

                Skx = np.concatenate((Skx, Skxi), axis=0)
                Sky = np.concatenate((Sky, Skyi), axis=0)
                Qkx = np.concatenate((Qkx, Qkxi), axis=0)
                Qky = np.concatenate((Qky, Qkyi), axis=0)
        return Skx, Sky, Qkx, Qky


    def next(self, q_size):
        random_class_V = np.random.choice(len(self.class_list),
                size = self.N, replace=False)
        Q_class_set = []
        S_class_set = []
        for v in random_class_V:
            Q_class_set.append(self.train_subset[v])
            S_class_set.append(self.test_subset[v])

        for n in range(self.N):
            if n == 0:
                Skx, _ = S_class_set[n].next_batch(self.K)
                Qkx, _ = Q_class_set[n].next_batch(q_size)
                Sky = [n for _ in range(self.K)]
                Qky = [n for _ in range(q_size)]

            else:
                Skxi, _ = S_class_set[n].next_batch(self.K)
                Qkxi, _ = Q_class_set[n].next_batch(q_size)
                Skyi = [n for _ in range(self.K)]
                Qkyi = [n for _ in range(q_size)]

                Skx = np.concatenate((Skx, Skxi), axis=0)
                Sky = np.concatenate((Sky, Skyi), axis=0)
                Qkx = np.concatenate((Qkx, Qkxi), axis=0)
                Qky = np.concatenate((Qky, Qkyi), axis=0)
        return Skx, Sky, Qkx, Qky
