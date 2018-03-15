import numpy as np
import h5py
from tflearn.data_augmentation import ImageAugmentation

class DataSet():
    def __init__(self):
        do_nothing=False

    def _get_root_loc(self):
        return '/home/mike/DataSet/'

    def next_batch(self, batch_size, onehot=True, aug=False):
        x, y = self.x, self.y
        seq_ind = np.random.randint(x.shape[0], size=batch_size)

        batch_x = x[seq_ind]
        batch_y = y[seq_ind]
        if onehot:
            batch_1hot = np.zeros([batch_size, 100])
            for i in range(batch_size):
                batch_1hot[i, batch_y[i]] = 1
            batch_y = batch_1hot
        if aug:
            batch_x = s.aug.apply(batch_x)
        return batch_x, batch_y

    def dataset_load(self):
        fo = h5py.File(self.loc, 'r')
        train_list = {'data': np.array(fo['data']),
                'labels':np.array(fo['labels'])}
        fo.close()

        self.x = np.array(train_list['data'])
        self.y = np.array(train_list['labels'])

    def to_git_style(self):
        d_all = []
        print (self.y.shape)
        print (self.y[:100])
        uu = np.unique(self.y)
        for _y in range(uu):
            d = self.x[self.y==_y]
            d_all.append(d)
        print (np.shape(d_all))
        return np.array(d_all)

class MNIST(DataSet):
    def __init__(self, mnist_or_not, set_type):
        self.loc = self._get_root_loc() + mnist_or_not + '/' + set_type + '.h5'
        self.dataset_load()

class Cifar(DataSet):
    def __init__(self, cifar_n, set_type):
        self.loc = self._get_root_loc() + cifar_n + '/' + set_type + '.h5'
        self.dataset_load()

class MiniImageNet32(DataSet):
    def __init__(self, set_type):
        self.loc = self._get_root_loc() + 'miniImageNet32/' + set_type + '.h5'
        self.dataset_load()

class MiniImageNet(DataSet):
    def __init__(self, set_type):
        self.loc = self._get_root_loc() + 'miniImageNet/' + set_type + '.h5'
        self.dataset_load()

class SubDataSet(DataSet):
    def __init__(self, class_list, full_dataset, k=0):
        self.class_list = class_list
        x, y = full_dataset

        ind = []
        for i in range(len(y)):
            if y[i] in class_list:
                ind.append(i)
        if k > 0:
            np.random.shuffle(ind)
            self.x = x[ind][:k]
            self.y = y[ind][:k]
        else:
            self.x = x[ind]
            self.y = y[ind]
