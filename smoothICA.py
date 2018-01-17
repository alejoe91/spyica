import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
import time
from sklearn.decomposition import PCA
import matplotlib.pylab as plt

def weight_variable(shape, name, seed):
    initial = tf.truncated_normal(shape, stddev=0.1, seed=seed)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def my_pow(X, exp):
    if exp >= 1:
        return tf.pow(X, exp)
    else:
        X_neg = tf.negative(tf.pow(tf.cast(X < 0, X.dtype)*tf.negative(X), exp))
        X_pos = tf.pow(tf.cast(X > 0, X.dtype)*X, exp)
        X_new = tf.add(X_pos, X_neg)

    return X_new


def svd_whiten(X):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)

    # U and Vt are the singular matrices, and s contains the singular values.
    # Since the rows of both U and Vt are orthonormal vectors, then U * Vt
    # will be white
    X_white = np.dot(U, Vt)

    return X_white


def smoothICA(X, n_comp='all', L=1, lamb=0, mu=0, n_iter=500, EM=False):
    '''

    Parameters
    ----------
    X
    n_comp
    L
    mu

    Returns
    -------

    '''

    if n_comp == 'all':
        n_comp = X.shape[0]
    else:
        n_comp = n_comp

    n_features = X.shape[0]

    # whiten data
    data = svd_whiten(X)
    learning_rate = 5e-2
    display_step = 50

    seed=np.random.seed(2308)
    # Launch the raph
    sess = tf.Session()

    if L == 1:
        Z = tf.constant(data, dtype=np.float32)
        W = weight_variable((n_comp, n_features), name='demixing', seed=seed)
        I = tf.constant(np.eye(n_comp, n_features), dtype=np.float32)
        
        y = tf.matmul(W, Z)
        # nonlin = tf.divide(tf.matmul(my_pow(y, 3), my_pow(y, 1./3.), transpose_b=True),
        #                    tf.constant(n_features, dtype=np.float32))
        nonlin = tf.divide(tf.matmul(tf.pow(y, 3), tf.tanh(y), transpose_b=True),
                           tf.constant(n_features, dtype=np.float32))

        square = tf.square(tf.subtract(nonlin, I))
        err = tf.reduce_sum(tf.square(tf.subtract(nonlin, I)))

        train_step = tf.train.AdamOptimizer(learning_rate).minimize(err)
        sess.run(tf.global_variables_initializer())

        ############
        # TRAINING #
        ############
        t_start = time.time()
        for epoch in range(n_iter):
            # print sess.run(nonlin)[:5, :5]
            sess.run(train_step)
            # Display logs per epoch step
            if (epoch + 1) % display_step == 0:
                train_err = sess.run(err)
                print "Step:", '%04d' % (epoch + 1), "Cost=", "{:.9f}".format(train_err)
                print 'Elapsed time: ', time.time() - t_start

        W_opt = sess.run(W)
        y_opt = sess.run(y)
        A_opt = np.linalg.inv(W_opt)
    else:
        W_opt = []
        y_opt = []
        A_opt = []

    return y_opt, A_opt, W_opt



