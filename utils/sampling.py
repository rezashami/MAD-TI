import numpy as np


class Sampling:
    def __init__(self, pos_len, neg_len,batch_size =60, shuffle=True, ratio=0):
        """ This function is constructor of Sampling class

        Parameters
        ----------
        batch_size : int
            Size of batch
        pos_ind : numpy array
            List of positive indices
        neg_ind : numpy array
            List of negative indices
        ratio : int, optional
            ration of sampling. 0 means 1:1, 1 means 1:3, 2 means 1:5, and 3 means 1:10, by default 0
        """
        self.batch_size = batch_size
        self.pos_indices = np.arange(pos_len)
        self.neg_indices = np.arange(neg_len)
        self.ratio = ratio
        self.pos_num = pos_len
        self.neg_num = neg_len

        if ratio ==0:
            self.pc = 30
            self.nc = 30
        elif ratio ==1:
            self.pc = 15
            self.nc = 45
        elif ratio ==2:
            self.pc = 10
            self.nc = 50
        elif ratio ==3:
            self.pc = 6
            self.nc = 54
        
        self.iter_counter = 0
        self.shuffle = shuffle
        if shuffle:
            np.random.shuffle(self.pos_indices)
            np.random.shuffle(self.neg_indices)

    def next(self):
        if self.num_iterations_left() <= 0:
            self.reset()
        self.iter_counter += 1
        
        pos_ind = self.pos_indices[(self.iter_counter - 1) * self.pc:self.iter_counter * self.pc]
        neg_ind = self.neg_indices[(self.iter_counter - 1) * self.nc:self.iter_counter * self.nc]
        pos_ind.sort()
        neg_ind.sort()
        return pos_ind, neg_ind
    def num_iterations(self):
        return int(self.neg_num // self.batch_size)

    def num_iterations_left(self):
        return self.num_iterations() - self.iter_counter

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.pos_indices)
            np.random.shuffle(self.neg_indices)
        self.iter_counter = 0
