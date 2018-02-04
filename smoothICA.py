import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
import time
from sklearn.decomposition import PCA
import matplotlib.pylab as plt
from scipy.optimize import curve_fit
import scipy.stats as stat

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

def tri_norm(x, m1, m2, m3, s1, s2, s3, k1, k2, k3):
    ret = k1*stat.norm.pdf(x, loc=m1 ,scale=s1)
    ret += k2*stat.norm.pdf(x, loc=m2 ,scale=s2)
    ret += k3*stat.norm.pdf(x, loc=m3 ,scale=s3)
    return ret

def bi_norm(x, m1, m2, s1, s2, k1, k2):
    ret = k1 * stat.norm.pdf(x, loc=m1, scale=s1)
    ret += k2 * stat.norm.pdf(x, loc=m2, scale=s2)
    return ret


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
    n_obs = X.shape[1]
    batch_size = int(1.*n_obs)
    # whiten data
    pca = PCA(n_components=n_comp, whiten=True)
    pca.fit(X)
    data = pca.components_

    learning_rate = 1e2
    display_step = 20

    seed=np.random.seed(2308)
    # Launch the raph
    sess = tf.Session()

    nonlin_mat = []
    # todo approximate pdf with sum of gaussians and then compute cdf
    hb = np.array([np.histogram(rec, bins=200) for rec in data])
    hist = hb[:, 0]
    bins = hb[:, 1]
    bin_center = [(bins[0][i] + bins[0][i + 1]) / 2. for i in range(len(bins[0]) - 1)]
    h_center = hist[np.argmax([stat.skew(h) for h in hist])]
    popt, pcov = curve_fit(tri_norm, bin_center, h_center)

    fit_values = tri_norm(bin_center, *popt)

    return bin_center, h_center, fit_values, popt

    # if L == 1:
    #     # Z = tf.constant(data, dtype=np.float32)
    #     Z = tf.placeholder("float", shape=[n_comp, None])
    #     W = weight_variable((n_comp, n_features), name='demixing', seed=seed)
    #     # W = tf.divide(W, tf.norm(W))
    #     I = tf.constant(np.eye(n_comp, n_features), dtype=np.float32)
    #
    #     y = tf.matmul(W, Z)
    #     a = tf.constant(4.)
    #
    #     # use skewed distr TOO SLOW
    #     # todo approximate pdf with sum of gaussians and then compute cdf
    #
    #
    #     term_2 = tf.cast(tf.py_func(skewnorm.cdf, [y, a], tf.double), dtype=tf.float32)
    #
    #
    #     # # TODO: try fastICA non-gaussianity
    #     # v = tf.random_normal((1, 1000000))
    #     #
    #     # def G(tensor):
    #     #     # return tf.log(tf.cosh(tensor))
    #     #     #return tf.negative(tf.exp(tf.divide(tf.negative(tensor), 2)))
    #     #     return tf.pow(tensor, 3)
    #     #
    #     # # def G_kurt(tensor):
    #     # #     return tf.pow(tensor, 4)
    #     #
    #     # err = tf.divide(1., tf.square(tf.subtract(tf.reduce_mean(G(y)), tf.reduce_mean(G(v)))))
    #     # # err = tf.divide(1., tf.reduce_mean(G_kurt(y)))
    #     # train_step = tf.train.AdamOptimizer(learning_rate).minimize(err)
    #
    #     #term_1 = tf.divide(1-tf.exp(-y), 1+tf.exp(-y))
    #     #term_1 = tf.pow(y, 3)
    #     # term_1 = tf.exp(y)-1
    #     term_1 = y
    #     # term_2 = tf.subtract(y, tf.tanh(y))
    #
    #     nonlin = tf.matmul(term_1, term_2, transpose_b=True)
    #     nonlin = tf.divide(nonlin, n_obs)
    #
    #     square = tf.square(tf.subtract(nonlin, I))
    #     err = tf.reduce_sum(square)
    #     # err = tf.reduce_sum(tf.square(tf.subtract(nonlin, tf.diag_part(nonlin))))
    #     train_step = tf.train.AdamOptimizer(learning_rate).minimize(err)
    #
    #     sess.run(tf.global_variables_initializer())
    #     sess.as_default()
    #
    #     raise Exception()
    #     ############
    #     # TRAINING #
    #     ############
    #     t_start = time.time()
    #     for epoch in range(n_iter):
    #         idxs = np.random.permutation(n_obs)[:batch_size]
    #         train_batch = data[:, idxs]
    #         sess.run(train_step, feed_dict={Z: train_batch})
    #         # Display logs per epoch step
    #         if (epoch + 1) % display_step == 0:
    #             train_err = sess.run(err, feed_dict={Z: train_batch})
    #             print "Step:", '%04d' % (epoch + 1), "Cost=", "{:.9f}".format(train_err)
    #             print 'Elapsed time: ', time.time() - t_start
    #
    #             # print sess.run(tf.reduce_mean(v))
    #             # print 'nonlin diag: ', sess.run(tf.diag_part(nonlin))
    #
    #             nonlin_mat.append(sess.run(nonlin, feed_dict={Z: train_batch}))
    #
    #     # todo: show Gert that it doesn't work because it optimizes the chosen function (if y**2)
    #     # sources are note independent
    #
    #     W_opt = sess.run(W)
    #     y_opt = sess.run(y, feed_dict={Z: data})
    #     A_opt = np.linalg.inv(W_opt)
    # else:
    #     W_opt = []
    #     y_opt = []
    #     A_opt = []
    #
    # return y_opt, A_opt, W_opt, nonlin_mat

def return_training_data(num, n_obs):
    # random_idxs = np.random.choice(self.num_train_spikes, num)
    # elems = tf.convert_to_tensor(range(self.num_train_spikes))
    samples = tf.multinomial(tf.log([[10.] * n_obs]), num)  # note log-prob
    random_idxs = samples[0].eval()
    # print random_idxs

    return random_idxs



