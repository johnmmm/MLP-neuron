from network import Network
from utils import LOG_INFO
from layers import Relu, Sigmoid, Linear
from loss import EuclideanLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d
import csv
from utils import writer

train_data, test_data, train_label, test_label = load_mnist_2d('data')


writer.writerow(['type','middle_layers','learning_rate','weight_decay','momentum'])

# Your model defintion here
# You should explore different model architecture
middle_layers1 = 500
middle_layers2 = 300

model = Network()
model.add(Linear('fc1', 784, middle_layers1, 0.01))
model.add(Relu('sig1'))
model.add(Linear('fc2', middle_layers1, middle_layers2, 0.01))
model.add(Sigmoid('sig2'))
model.add(Linear('fc3', middle_layers2, 10, 0.01))

loss = EuclideanLoss(name='loss')

# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.

config = {
    'learning_rate': 0.01,
    'weight_decay': 0.0001,
    'momentum': 0.9,
    'batch_size': 100,
    'max_epoch': 100,
    'disp_freq': 50,
    'test_epoch': 5
}

rlist=[]
rlist.append('Relu')
rlist.append(middle_layers1)
rlist.append(config['learning_rate'])
rlist.append(config['weight_decay'])
rlist.append(config['momentum'])
writer.writerow(rlist)
writer.writerow(['iter_counter','loss_list','acc_list'])

for epoch in range(config['max_epoch']):
    LOG_INFO('Training @ %d epoch...' % (epoch))
    train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])

    if epoch % config['test_epoch'] == 0:
        LOG_INFO('Testing @ %d epoch...' % (epoch))
        test_net(model, loss, test_data, test_label, config['batch_size'])
