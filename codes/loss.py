from __future__ import division
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        '''Your codes here'''
        #print (input)
        temp = input - target
        temp = temp * temp
        temp /= 2
        loss = temp.sum(axis=0)
        loss /= len(input)
        return loss

    def backward(self, input, target):
        '''Your codes here'''
        loss = input - target
        #loss = temp.sum(axis=0)
        loss /= len(input)
        return loss
