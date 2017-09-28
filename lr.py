import numpy as np
import math
import optimizer
import sys
import scipy.io as sio

from ranking import auc
def logit(x):
    temp = 1.0 + math.exp(-x)
    return 1.0 / temp

class logistic_regression(object):

    def __init__(self):
        self.solver = optimizer.lbfgs(10)
        self.instance = []
        self.label = []

    def add_instance(self, i, l):
        # read instance
        self.instance.append(i)
        self.label.append(l)

    def evaluate(self):
        i = 0
        right = 0
        pred = []
        while i < len(self.instance):
            temp = logit(np.dot(self.instance[i], self.w))
            pred.append(temp)
#            if int(round(temp)) == int(self.label[i]):
            if (temp > 0.5 and self.label[i] == 1) or (temp <= 0.5 and self.label[i] == 0):
                right += 1
#                print '%e' % temp, self.label[i], self.instance[i]
            else:
                print '%e' % temp, self.label[i], self.instance[i]
            i += 1

        print 'right: %d, total: %d, ratio: %f' % (right, len(self.instance), float(right) / float(len(self.instance)))
        print self.w
#        sio.mmwrite('ins.mat', self.instance)
#        sio.mmwrite('label.mat', np.array(self.label).reshape(len(self.label),1))
        print auc(self.label, pred)

    def predict(self):
        return

    def train(self):
#        self.w = self.solver.minimize(self, np.random.normal(size=self.instance[0].size))
        self.w = self.solver.minimize(self, np.zeros(self.instance[0].size))
       
    def grad(self, w):
        # compute gradient
        return

    def value(self, w):
        # compute loss value
        i = 0
        loss = 0.0
        gradient = np.zeros(w.size)
        pred = []
        while i < len(self.instance):
            temp = - np.dot(self.instance[i], w)
            pred.append(1.0 / (1.0 + math.exp(temp)))
            if self.label[i] == 0:
                temp = - temp
            if temp > 30:
                ins_loss = -temp
                ins_prob = 1.0
            elif temp < -30:
                ins_loss = 0.0
                ins_prob = 0.0
            else:
                ins_loss = 1.0 / (1.0 + math.exp(temp))
                ins_prob = 1.0 - ins_loss
                ins_loss = math.log(ins_loss)
            # loss and gradient for minimization
            if self.label[i] == 0:
                ins_prob = - ins_prob

            loss += -ins_loss
            gradient += -ins_prob * self.instance[i]
            i += 1
        print 'AUC', auc(self.label, pred), loss
        return loss, gradient
