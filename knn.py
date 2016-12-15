# -*- coding: utf-8 -*-

import random


class NearestNeighbor:
    
    def __init__(self, _k=1, _x=(), _y=()):
        self.k = _k     # for k-nn
        self.x = _x     # training data x
        self.y = _y     # training data y
        self.length = len(self.x)
        self.maximum = max(self.y)
    
    def predict(self, x):
        
        y_pred = []
        
        for i in range(len(x)):
            dist = []
            for j in range(self.length):
                d = 0
                for k in range(len(x[0])):
                    d += (x[i][k]-self.x[j][k])**2
                # j is index
                # d is distance^2
                dist.append((j, d))
            
            # sort by distance value
            # we will use index value of k top values
            dist.sort(key=lambda a: a[1])
            
            # y_temp contains lists of index and count
            y_temp = [[v+1, 0] for v in range(self.maximum)]
            for k in range(self.k):
                idx = self.y[dist[k][0]] - 1
                y_temp[idx][1] += 1
            
            # sort by count value
            y_temp.sort(key=lambda a: a[1], reverse=True)
            y_pred.append(y_temp[0][0])
        return y_pred


def load_data(filename):
    f = open(filename, 'r')
    x = []
    y = []
    for l in f:
        raw = list(map(float, l.strip().split(',')))
        x.append(raw[1:-1])
        y.append(int(raw[-1]))
    return x, y


def normalize(_x):
    for i in range(len(_x)):
        _x[i][0] = (_x[i][0]-1.5184)/0.0030
        _x[i][1] = (_x[i][1]-13.4079)/0.8166
        _x[i][2] = (_x[i][2]-2.6845)/1.4424
        _x[i][3] = (_x[i][3]-1.4449)/0.4993
        _x[i][4] = (_x[i][4]-72.6509)/0.7745
        _x[i][5] = (_x[i][5]-0.4971)/0.6522
        _x[i][6] = (_x[i][6]-8.9570)/1.4232
        _x[i][7] = (_x[i][7]-0.1750)/0.4972
        _x[i][8] = (_x[i][8]-0.0570)/0.0974


def nn(flag=False):
    if flag:
        print('\nNormalizing...')
        normalize(x)
    for k in range(1, 26):
        accuracy = 0
        
        # for 10-fold cross validation
        for i in range(10):
            x_test = []
            y_test = []
            x_train = []
            y_train = []
            
            l = length // 10
            # i-th part gonna be test part
            # rest parts of them gonna be training parts
            for j in r[:l * i]:
                x_train.append(x[j])
                y_train.append(y[j])
            for j in r[l * i:l * (i+1)]:
                x_test.append(x[j])
                y_test.append(y[j])
            for j in r[l * (i+1):]:
                x_train.append(x[j])
                y_train.append(y[j])
            
            # k-nn with train data
            knn = NearestNeighbor(k, x_train, y_train)
            
            # result of k-nn
            y_pred = knn.predict(x_test)
            
            # calculate accuracy
            correct = 0
            for j in range(len(y_test)):
                correct += 1 if y_test[j] == y_pred[j] else 0
            
            accuracy += (correct / len(y_test)) / 10
        
        print('k: %d, accuracy: %f' % (k, accuracy))
        val_acc.append((k, accuracy))


if __name__ == "__main__":

    val_acc = []
    
    x, y = load_data('glass.data')
    length = len(x)
    
    # make index list and shake it in order to select training data
    # call shuffle only one time.
    r = list(range(length))
    random.shuffle(r)

    nn(False)
    nn(True)

