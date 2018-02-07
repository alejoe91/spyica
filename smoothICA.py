import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
import time
from sklearn.decomposition import PCA
import matplotlib.pylab as plt
from scipy.optimize import curve_fit
import scipy.stats as stat
from scipy.integrate import simps

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


def whiten(X, n_comp):
    # whiten data
    pca = PCA(n_components=n_comp, whiten=True)
    data = pca.fit_transform(np.transpose(X))

    return np.transpose(data), pca.components_


def tri_norm(x, m1, m2, m3, s1, s2, s3, k1, k2, k3):
    ret = k1*stat.norm.pdf(x, loc=m1 ,scale=s1)
    ret += k2*stat.norm.pdf(x, loc=m2 ,scale=s2)
    ret += k3*stat.norm.pdf(x, loc=m3 ,scale=s3)
    return ret

def bi_norm(x, m1, m2, s1, s2, k1, k2):
    ret = k1 * stat.norm.pdf(x, loc=m1, scale=s1)
    ret += k2 * stat.norm.pdf(x, loc=m2, scale=s2)
    return ret


def smoothICA(X, n_comp='all', L=1, lamb=0, mu=0, n_iter=1000, EM=False):
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
    data, pc = whiten(X, n_comp)

    learning_rate = 1e2
    display_step = 20

    seed=np.random.seed(2308)
    # Launch the raph
    sess = tf.Session()

    nonlin_mat = []
    # # todo approximate pdf with sum of gaussians and then compute cdf
    hb = np.array([np.histogram(rec, bins=200) for rec in data])
    hist = hb[:, 0]
    bins = hb[:, 1]
    bin_center = [(bins[0][i] + bins[0][i + 1]) / 2. for i in range(len(bins[0]) - 1)]
    h_center = hist[np.argmax([abs(stat.skew(h)) for h in hist])]
    popt, pcov = curve_fit(tri_norm, bin_center, h_center)

    fit_values = tri_norm(bin_center, *popt)
    auc = simps(fit_values, dx=0.01)

    m1, m2, m3, s1, s2, s3, k1, k2, k3 = popt
    k1 /= auc
    k2 /= auc
    k3 /= auc

    fit_values_norm = tri_norm(bin_center, m1, m2, m3, s1, s2, s3, k1, k2, k3)
    auc_norm = simps(fit_values_norm, dx=0.01)

    # print auc, auc_norm

    if L == 1:
        Z = tf.placeholder("float", shape=[n_comp, None])
        W = weight_variable((n_comp, n_features), name='demixing', seed=seed)
        I = tf.constant(np.eye(n_comp, n_features), dtype=np.float32)

        y = tf.matmul(W, Z)
        a = tf.constant(4.)

        # todo approximate pdf with sum of gaussians and then compute cdf
        # term_1 = 0.5*(3 + k1*tf.erf((y - m1)/(np.sqrt(2)*s1))
        #               + k2*tf.erf((y - m2)/(np.sqrt(2)*s2))
        #               + k3*tf.erf((y - m3)/(np.sqrt(2)*s3)))

        def exp_X2_taylor(X):
            return 1 - tf.pow(X, 2) + 0.5*tf.pow(X, 4)

        # term_1 = tf.divide(-2*(1./s1 * (y - m1) * tf.exp(-(y - m1)/(2*(s1**2)))
        #                        + 1./s2 * (y - m2) * tf.exp(-(y - m2)/(2*(s1**2)))
        #                        + 1./s3 * (y - m3) * tf.exp(-(y - m3)/(2*(s3**2)))),
        #                    ((1./s1 * tf.exp(-(y - m1)/(2*(s1**2)))
        #                        + 1./s2 * tf.exp(-(y - m2)/(2*(s1**2)))
        #                        + 1./s3 * tf.exp(-(y - m3)/(2*(s3**2))))))

        # term_1 = tf.divide(-2*(1./s1 * (y - m1) * exp_X2_taylor((y - m1)/(2*(s1**2)))
        #                        + 1./s2 * (y - m2) * exp_X2_taylor((y - m2)/(2*(s1**2)))
        #                        + 1./s3 * (y - m3) * exp_X2_taylor((y - m3)/(2*(s3**2)))),
        #                    ((1./s1 * exp_X2_taylor((y - m1)/(2*(s1**2)))
        #                        + 1./s2 * exp_X2_taylor((y - m2)/(2*(s1**2)))
        #                        + 1./s3 * exp_X2_taylor((y - m3)/(2*(s3**2))))))

        term_1 = tf.pow(y,3)
        term_2 = tf.atan(y)
        # term_2 = y
        # term_2 = 2*tf.tanh(y)

        nonlin = tf.matmul(term_1, term_2, transpose_b=True)
        nonlin = tf.divide(nonlin, n_obs)

        square = tf.square(tf.subtract(nonlin, I))
        err = tf.reduce_sum(square)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(err)

        sess.run(tf.global_variables_initializer())
        sess.as_default()

        ############
        # TRAINING #
        ############
        t_start = time.time()
        for epoch in range(n_iter):
            idxs = np.random.permutation(n_obs)[:batch_size]
            train_batch = data[:, idxs]
            sess.run(train_step, feed_dict={Z: train_batch})
            # Display logs per epoch step
            if (epoch + 1) % display_step == 0:
                train_err = sess.run(err, feed_dict={Z: train_batch})
                print "Step:", '%04d' % (epoch + 1), "Cost=", "{:.9f}".format(train_err)
                print 'Elapsed time: ', time.time() - t_start

                nonlin_mat.append(sess.run(nonlin, feed_dict={Z: train_batch}))

        W_opt = sess.run(W)
        y_opt = sess.run(y, feed_dict={Z: data})
        A_opt = np.linalg.inv(W_opt)
    else:
        W_opt = []
        y_opt = []
        A_opt = []

    return y_opt, A_opt, W_opt, nonlin_mat, data