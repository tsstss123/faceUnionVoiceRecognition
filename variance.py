import numpy as np

class VectorQueue(object):

    def __init__(self, dim, len):
        self.length = int(len)
        self.dim = int(dim)
        self.array = None
        self.debug = False

    def push(self, data):
        if len(data.shape) != 1 or data.shape[0] != self.dim:
            print('vector queue insert ERROR!')
        data.shape = (data.shape[0], 1)
        if self.debug:
            print('add data:')
            print(data)
        if self.array is None:
            if self.debug:
                print('add first batch')
            self.array = data
        else:
            self.array = np.concatenate((self.array, data), axis = 1)
            # print('append new data')
            # print(self.array)
            if self.array.shape[1] >= self.length:
                self.array = self.array[:, 1:]
                # print(self.array.shape)
                # print('out of range')

    def var(self):
        # print('var :')
        # print(np.var(self.array, axis = 1))
        return np.var(self.array, axis = 1)

    def max(self):
        return np.max(self.array, axis = 1)

    def mean(self):
        return np.mean(self.array, axis = 1)

if __name__ == '__main__':
    q = VectorQueue(dim = 5, len = 3)
    for i in range(10):
        q.push(np.random.rand(5))
        q.var()
        q.max()