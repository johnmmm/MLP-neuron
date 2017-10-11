from __future__ import division
from __future__ import print_function
import numpy as np
from datetime import datetime
import csv

c=open("dan.csv","w")
writer=csv.writer(c)
d=open("test.csv","w")
writer2=csv.writer(d)
writer2.writerow(['test','loss_list','acc_list'])

def onehot_encoding(label, max_num_class):
    encoding = np.eye(max_num_class)
    encoding = encoding[label]
    return encoding


def calculate_acc(output, label):
    correct = np.sum(np.argmax(output, axis=1) == label)
    return correct / len(label)


def LOG_INFO(msg):
    now = datetime.now()
    display_now = str(now).split(' ')[1][:-3]
    print(display_now + ' ' + msg)
