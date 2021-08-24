import tensorflow as tf
import numpy as np
import sys, os, cv2
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from skimage.transform import resize
from matplotlib.pyplot import imread
from imgaug import augmenters as iaa
import nibabel as nib
import imgaug as ia
from scipy.ndimage import zoom
import matplotlib.animation as animation

plt.style.use('seaborn-white')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.random.seed(6278)
tf.random.set_seed(6278)
ia.seed(6278)


# layers
def tf_elu(x):
    """
    Exponential Linear Unit based on the ICCV 2015 paper:
    https://arxiv.org/pdf/1511.07289.pdf
    Parameters
    ----------
    x: float
        Floating point number applied to ELU activation .

    Returns
    -------
    float
        Data with same dimensions as input after ELU
    """
    return tf.nn.elu(x)


def d_tf_elu(x):
    """
    Derivative of ELU activation
    Parameters
    ----------
    x: type
        Description

    Returns
    -------
    type
        Description
    """
    return tf.cast(tf.greater(x, 0), tf.float64)


def tf_logcosh(x): return tf.math.log(tf.cosh(x))
def d_tf_logcosh(x): return tf.tanh(x)
def tf_tanh(x): return tf.tanh(x)
def d_tf_tanh(x): return 1.0 - tf.tanh(x) ** 2
def tf_sigmoid(x): return tf.nn.sigmoid(x)
def d_tf_sigmoid(x): return tf_sigmoid(x) * (1.0 - tf_sigmoid(x))


# ----FastICA special function ----
def tf_cube(x): return x ** 3
def d_tf_cube(x): return 2 * x ** 2
def tf_exp(x): return x * tf.exp(-(x ** 2) / 2.0)
def d_tf_exp(x): (1 - x ** 2) * tf.exp(-(x ** 2) / 2.0)


class CNN:
    """
    Deep Network

    Parameters
    ----------
    k: int
        Description
    inc: int
        Description
    out: int
        Description
    act: function
        Description
    d_act: function
        Description
    batch_size: int
        Description
    beta1: float
        Description
    beta2: float
        Description
    adam_e: float
        Description
    learning_rate: float
        Description

    """
    def __init__(self, k, inc, out, act=tf_elu, d_act=d_tf_elu, batch_size=1,
                 beta1=0.9, beta2=0.999, adam_e=1e-8, learning_rate=0.005):
        self.w = tf.Variable(tf.random.normal([k, k, inc, out], stddev=0.05, seed=2, dtype=tf.float64))
        self.m, self.v = tf.Variable(tf.zeros_like(self.w)), tf.Variable(tf.zeros_like(self.w))
        self.act, self.d_act = act, d_act
        self.batch_size = batch_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.adam_e = adam_e
        self.learning_rate = learning_rate

    def getw(self): return self.w

    def feedforward(self, inp, stride=1, padding='SAME'):
        self.inp = inp
        self.layer = tf.nn.conv2d(self.inp, self.w, strides=stride, padding=padding)
        # self.layer = tf.keras.layers.Conv2D(inp, self.w, strides=[stride, stride], padding=padding)
        self.layerA = self.act(self.layer)
        return self.layerA

    def backprop(self, gradient, stride=1, padding='SAME', l2_reg=False):
        grad_part_1 = gradient
        grad_part_2 = self.d_act(self.layer)
        grad_part_3 = self.inp
        # print(f"grad_part_1: {grad_part_1.shape}, grad_part_2: {grad_part_2.shape}, grad_part_3: {grad_part_3.shape}")

        grad_middle = grad_part_1 * grad_part_2

        grad = tf.nn.depthwise_conv2d_backprop_filter(input=grad_part_3, filter_sizes=self.w.shape,
                                                      out_backprop=grad_middle, strides=[1, stride, stride, 1],
                                                      padding=padding) / self.batch_size

        grad_pass = tf.nn.depthwise_conv2d_backprop_input(input_sizes=[self.batch_size] + list(grad_part_3.shape[1:]),
                                                          filter=self.w, out_backprop=grad_middle,
                                                          strides=[1, stride, stride, 1], padding=padding)

        update_w = []
        update_w.append(tf.compat.v1.assign(self.m, self.m * self.beta1 + (1 - self.beta1) * grad))
        update_w.append(tf.compat.v1.assign(self.v, self.v * self.beta2 + (1 - self.beta2) * (grad ** 2)))
        m_hat = self.m / (1 - self.beta1)
        v_hat = self.v / (1 - self.beta2)
        adam_middel = self.learning_rate / (tf.sqrt(v_hat) + self.adam_e)
        update_w.append(tf.compat.v1.assign(self.w, tf.subtract(self.w, tf.multiply(adam_middel, m_hat))))
        return grad_pass, update_w


class TfPCALayer:
    def __init__(self, n_components):
        self.n_components = n_components

    def feedforward(self, inp):
        self.inp = inp
        # print(inp.shape)
        self.cov = tf.matmul(self.inp, tf.transpose(self.inp)) / (inp.shape[0] - 1)
        self.eigval, self.pc = tf.linalg.eigh(self.cov)
        self.pc_projection = self.pc[:, -self.n_components:]
        self.layer = tf.matmul(tf.transpose(self.pc_projection), inp)
        return self.layer

    def backprop(self, grad):
        # print(grad.shape)
        mat_shape = self.inp.shape[0]
        d_pc_project = tf.transpose(tf.matmul(grad, tf.transpose(self.inp)))
        diff = mat_shape - self.n_components
        added_mat = tf.zeros([mat_shape, diff], dtype=tf.float64)
        d_pc = tf.concat([d_pc_project, added_mat], 1)
        E = tf.matmul(tf.ones([mat_shape, 1], dtype=tf.float64), tf.transpose(self.eigval)[tf.newaxis, :]) - \
            tf.matmul(self.eigval[:, tf.newaxis], tf.ones([1, mat_shape], dtype=tf.float64))
        F = 1.0 / (E + tf.eye(mat_shape, dtype=tf.float64)) - tf.eye(mat_shape, dtype=tf.float64)
        d_cov = tf.matmul(tf.linalg.inv(tf.transpose(self.pc)),
                          tf.matmul(F * (tf.matmul(tf.transpose(self.pc), d_pc)), tf.transpose(self.pc)))
        d_x = tf.matmul(self.pc_projection, grad) + \
            (tf.matmul(d_cov, self.inp) + tf.matmul(tf.transpose(d_cov), self.inp)) / (mat_shape - 1)
        return d_x


class FastICALayer:
    """
    Performs ICA vis FastICA method

    Parameters
    ----------
    inc: int
        Description
    outc: int
        Description
    act: function
        Description
    d_act: function
        Description

    Attributes
    ----------
    w: type
        Description
    sym_decorelation: type
        Description
    m: type
        Description
    v: type
        Description
    self, matrix: type
        Description
    act: type
        Description
    d_act: type
        Description
    """

    def __init__(self, inc, outc, act, d_act,
                beta1=0.9, beta2=0.999, adam_e=1e-8, learning_rate=0.005):
        self.w = tf.Variable(self.sym_decorrelation(tf.random.normal(shape=[inc, outc], stddev=0.05, dtype=tf.float64, seed=2)))
        self.m = tf.Variable(tf.zeros_like(self.w))
        self.v = tf.Variable(tf.zeros_like(self.w))
        self.act = act
        self.d_act = d_act
        self.beta1 = beta1
        self.beta2 = beta2
        self.adam_e = adam_e
        self.learning_rate = learning_rate

    def sym_decorrelation(self, matrix):
        s, u = tf.linalg.eigh(tf.matmul(matrix, tf.transpose(matrix)))
        decor_matrix = tf.matmul(u * (1.0 / tf.sqrt(s)), tf.transpose(u))
        m = tf.matmul(decor_matrix, matrix)
        # print(matrix.shape)
        return m

    def getw(self): return self.w

    def feedforward(self, inp):
        self.inp = inp
        #print(inp.shape)
        self.layer = tf.matmul(self.w, inp)
        return self.layer

    def backprop_ica(self):
        self.layerA = self.act(tf.matmul(self.w, self.inp))
        self.layerDA = tf.reduce_mean(self.d_act(tf.matmul(self.w, self.inp)), -1)
        grad_pass = tf.matmul(tf.transpose(self.w), self.layer)
        # print(self.layer)

        grad_w = tf.matmul(self.layerA, tf.transpose(self.inp)) / \
            self.inp.shape[1] - self.layerDA[:, tf.newaxis] * self.w
        grad = self.sym_decorrelation(grad_w)

        update_w = []
        # ==== Correct Method of Weight Update ====
        # update_w.append(tf.compat.v1.assign(self.w, grad))

        # ==== Wrong Method of Weight Update ====
        update_w.append(tf.compat.v1.assign(self.m, self.m * self.beta1 + (1 - self.beta1) * grad))
        update_w.append(tf.compat.v1.assign(self.v, self.v * self.beta2 + (1 - self.beta2) * (grad ** 2)))
        m_hat = self.m / (1 - self.beta1)
        v_hat = self.v / (1 - self.beta2)
        adam_middle = m_hat * 0.01 * self.learning_rate / (tf.sqrt(v_hat) + self.adam_e)
        update_w.append((tf.compat.v1.assign(self.w, tf.subtract(self.w, adam_middle))))
        return grad_pass, update_w


class TfMeanLayer:

    def __init__(self):
        pass

    def feedforward(self, inp):
        self.mean = tf.reduce_mean(inp, 1)
        return inp - self.mean[:, tf.newaxis]

    def backprop(self, grad):
        return grad * (1 + 1.0 / grad.shape[0].value)