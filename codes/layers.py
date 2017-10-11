import numpy as np
import math

class Layer(object):
    def __init__(self, name, trainable=False):
        self.name = name
        self.trainable = trainable
        self._saved_tensor = None

    def forward(self, input):
        pass

    def backward(self, grad_output):
        pass

    def update(self, config):
        pass

    def _saved_for_backward(self, tensor):
        '''The intermediate results computed during forward stage
        can be saved and reused for backward, for saving computation'''

        self._saved_tensor = tensor


class Relu(Layer):
    def __init__(self, name):
        super(Relu, self).__init__(name)

    def forward(self, input):
        '''Your codes here'''
        output = input
        output[input < 0] = 0
        super(Relu, self)._saved_for_backward(output)
        return output

    def backward(self, grad_output):
        '''Your codes here'''
        fu = self._saved_tensor
        fup = fu
        fup[fu < 0] = 0
        fup[fu > 0] = 1
        #fup = fup.sum(axis = 0)
        output = grad_output * fup
        # print('~~~~~')
        # print(grad_output)
        # print('~~~~~')
        # print(fup)
        # print('~~~~~')
        # print(output)
        return output


class Sigmoid(Layer):
    def __init__(self, name):
        super(Sigmoid, self).__init__(name)

    def forward(self, input):
        '''Your codes here'''
        output = 1 / (1 + np.exp(-input))
        super(Sigmoid, self)._saved_for_backward(output)
        return output

    def backward(self, grad_output):
        '''Your codes here'''
        fu = self._saved_tensor
        fup = fu * (1 - fu)
        #fup = fup.sum(axis = 0)
        output = grad_output * fup
        # print('~~~~~')
        # print(grad_output)
        # print('~~~~~')
        # print(fup)
        # print('~~~~~')
        # print(output)
        return output


class Linear(Layer):
    def __init__(self, name, in_num, out_num, init_std):
        super(Linear, self).__init__(name, trainable=True)
        self.in_num = in_num
        self.out_num = out_num
        self.W = np.random.randn(in_num, out_num) * init_std
        self.b = np.zeros(out_num)

        self.grad_W = np.zeros((in_num, out_num))
        self.grad_b = np.zeros(out_num)

        self.diff_W = np.zeros((in_num, out_num))
        self.diff_b = np.zeros(out_num)

    def forward(self, input):
        '''Your codes here'''
        super(Linear, self)._saved_for_backward(input)

        output = np.dot(input, self.W) + self.b

        return output

    def backward(self, grad_output):
        '''Your codes here'''
        delta = grad_output

        inputy = self._saved_tensor
        #inputy = inputy.sum(axis=0)
        output = np.dot(delta, self.W.T)
        self.grad_W = np.dot(inputy.T, delta)
        self.grad_b = delta.sum(axis=0)
        # for i in range(0, self.in_num):
        #     for j in range(0, self.out_num):
        #         self.grad_W[i][j] = inputy[i] * delta[j]

        # output = np.zeros(self.in_num)
        # output = np.dot(self.W, delta)
        return output

    def update(self, config):
        mm = config['momentum']
        lr = config['learning_rate']
        wd = config['weight_decay']

        self.diff_W = mm * self.diff_W + (self.grad_W + wd * self.W)
        self.W = self.W - lr * self.diff_W

        self.diff_b = mm * self.diff_b + (self.grad_b + wd * self.b)
        self.b = self.b - lr * self.diff_b
