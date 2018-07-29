import numpy as np
import functions as f

'''
単レイヤーのクラス群
'''

class MatMul:
    """
    乗算レイヤー
    """
    def __init__(self, W):
        self.params = [W]
        self.grads = np.zeros_like(W)
        self.x = None


    def forward(self, x):
        W, = self.params
        out = np.dot(x, W)
        self.x = x
        return out


    def backward(self, dout):
        W, = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        self.grads[0][...] = dW
        return dx


class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * ( 1.0 - self.out) * self.out
        return dx


class Relu:
    def __init__(self):
        self.params, self.grads = [], []
        self.mask = None


    def forward(self, x):
        self.mask = (x <=0)
        out = x.copy() # 値渡し
        out[self.mask] = 0

        return out


    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

class Affine:
    """
    全結合層レイヤ
    """

    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        self.x = x

        return out

    def backward(self, dout):
        W, b = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 損失
        self.pred = None # softmaxの出力 それぞれのカテゴリの分類確率
        self.true = None # 教師データ one-hot vector

    def forward(self, x, true):
        self.true = true
        self.pred = f.softmax(x)
        self.loss = f.cross_entropy_error(self.pred, self.true)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.true.shape[0]
        dx = (self.pred - self.true)/ batch_size

        return dx
