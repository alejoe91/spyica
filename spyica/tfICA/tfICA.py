import quantities as pq
import numpy as np
import tensorflow as tf
from spikeinterface import NumpySorting
import matplotlib.pyplot as plt
from .tools import TfMeanLayer, FastICALayer, TfPCALayer, CNN, \
    tf_sigmoid, d_tf_sigmoid, tf_cube, d_tf_cube
from ..tools import clean_sources, detect_and_align, reject_duplicate_spiketrains, \
    cluster_spike_amplitudes


class TfFastICA:

    def __init__(self, train_batch, num_samples=320000, num_channels=32, num_epoch=10, learning_rate=0.005, print_size=100,
                 batch_size=1, beta1=0.9, beta2=0.999, adam_e=1e-8):

        self.train_batch = train_batch
        self._num_epoch = num_epoch
        self._num_channels = num_channels
        self._num_samples = num_samples
        self._learning_rate = learning_rate
        self._print_size = print_size
        self._batch_size = batch_size
        self._beta1 = beta1
        self._beta2 = beta2
        self._adam_e = adam_e

    def run(self):
        l1 = CNN(3, 1, 1, act=tf_sigmoid, d_act=d_tf_sigmoid)
        l2 = CNN(3, 1, 1, act=tf_sigmoid, d_act=d_tf_sigmoid)
        l3 = CNN(3, 1, 1, act=tf_sigmoid, d_act=d_tf_sigmoid)
        l4 = CNN(3, 1, 1, act=tf_sigmoid, d_act=d_tf_sigmoid)

        # pca_l_1 = TfPCALayer(8)
        # pca_l_2 = TfPCALayer(8)
        # pca_l_3 = TfPCALayer(8)
        # pca_l_4 = TfPCALayer(8)
        ica_l_1 = FastICALayer(self._num_channels - 8, self._num_channels - 8, act=tf_cube, d_act=d_tf_cube)
        ica_l_2 = FastICALayer(self._num_channels - 8, self._num_channels - 8, act=tf_cube, d_act=d_tf_cube)
        ica_l_3 = FastICALayer(self._num_channels - 8, self._num_channels - 8, act=tf_cube, d_act=d_tf_cube)
        ica_l_4 = FastICALayer(self._num_channels - 8, self._num_channels - 8, act=tf_cube, d_act=d_tf_cube)

        x = tf.compat.v1.placeholder(tf.float64, shape=[self._batch_size, self._num_channels, self._num_samples, 1])
        layer1 = l1.feedforward(x, padding='VALID')
        layer2 = l2.feedforward(layer1, padding='VALID')
        layer3 = l3.feedforward(layer2, padding='VALID')
        layer4 = l4.feedforward(layer3, padding='VALID')
        # layer_flat = tf.reshape(layer3, [self._batch_size, -1])
        layer4 = tf.squeeze(layer4)
        print(f"layer4: {layer4.shape}")

        # pca_layer_1 = pca_l_1.feedforward(layer_flat[:int(self._batch_size/4), :])
        # pca_layer_2 = pca_l_2.feedforward(layer_flat[int(self._batch_size/4):int(self._batch_size/2), :])
        # pca_layer_3 = pca_l_3.feedforward(layer_flat[int(self._batch_size/2):int(self._batch_size*(3/4)), :])
        # pca_layer_4 = pca_l_4.feedforward(layer_flat[int(self._batch_size*(3/4)):, :])

        ica_layer_1 = ica_l_1.feedforward(layer4[:, :int(layer4.shape[1] / 4)])
        ica_layer_2 = ica_l_2.feedforward(layer4[:, int(layer4.shape[1] / 4): int(layer4.shape[1] / 2)])
        ica_layer_3 = ica_l_3.feedforward(layer4[:, int(layer4.shape[1] / 2): int(layer4.shape[1] / 4 * 3)])
        ica_layer_4 = ica_l_4.feedforward(layer4[:, int(layer4.shape[1] / 4 * 3):])
        # print(f"ica1: {ica_layer_1.shape}, ica2: {ica_layer_2.shape}, ica3: {ica_layer_3.shape}, ica4: {ica_layer_4.shape}")
        all_ica_section = ica_layer_1 + ica_layer_2 + ica_layer_3 + ica_layer_4
        print(f"ICAlayer4: {ica_layer_4.shape}")

        grad_ica_1, grad_ica_up_1 = ica_l_1.backprop_ica()
        grad_ica_2, grad_ica_up_2 = ica_l_2.backprop_ica()
        grad_ica_3, grad_ica_up_3 = ica_l_3.backprop_ica()
        grad_ica_4, grad_ica_up_4 = ica_l_4.backprop_ica()

        # grad_pca_1 = pca_l_1.backprop(grad_ica_1)
        # grad_pca_2 = pca_l_2.backprop(grad_ica_2)
        # grad_pca_3 = pca_l_3.backprop(grad_ica_3)
        # grad_pca_4 = pca_l_4.backprop(grad_ica_4)

        # grad_pca_reshape = tf.reshape(tf.concat([grad_pca_1, grad_pca_2, grad_pca_3, grad_pca_4], 0),
        #                               [self._batch_size, 26, 26, 1])
        grad_ica_reshape = tf.reshape(tf.concat([grad_ica_1, grad_ica_2, grad_ica_3, grad_ica_4], 1),
                                      [self._batch_size, self._num_channels - 8, self._num_samples - 8, 1])
        grad_4, grad_4_up = l4.backprop(grad_ica_reshape, padding='VALID')
        grad_3, grad_3_up = l3.backprop(grad_4, padding='VALID')
        grad_2, grad_2_up = l2.backprop(grad_3, padding='VALID')
        grad_1, grad_1_up = l1.backprop(grad_2, padding='VALID')
        grad_up = grad_ica_up_1 + grad_ica_up_2 + grad_ica_up_3 + grad_ica_up_4

        gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
        sess = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
        sess.run(tf.compat.v1.global_variables_initializer())
        results_for_animation = []
        print(type(x))
        dicty = {x: self.train_batch}
        for it in range(self._num_epoch):
            sess_results = sess.run([layer1, layer2, layer3, layer4,
                                     ica_layer_1, ica_layer_2, ica_layer_3, ica_layer_4,
                                     grad_1_up, grad_2_up, grad_3_up, grad_4_up,
                                     all_ica_section, grad_up],
                                    feed_dict=dicty)
            print(f"it: {it}, mean: {sess_results[4].shape}")
            self.data = []
            for x in sess_results:
                self.data.append(x)
        self.res = self.data[4:8]

        sess.close()

        # shows
        # fig = plt.figure(figsize=(30, 30))
        # columns = 10
        # rows = 10
        # for i in range(1, columns * rows + 1):
        #     fig.add_subplot(rows, columns, i)
        #     plt.imshow(np.squeeze(self.train_batch[:self._batch_size][i - 1]), cmap='gray')
        #     plt.axis('off')
        #     plt.title(str(i))
        # plt.show()
        # print('-------------------------------------')
        #
        # for temp in all_data_c:
        #     fig = plt.figure(figsize=(30, 30))
        #     columns = 10
        #     rows = 10
        #     for i in range(1, columns * rows + 1):
        #         fig.add_subplot(rows, columns, i)
        #         try:
        #             num = int(np.sqrt(temp[i - 1].shape[0]))
        #             plt.imshow(np.squeeze(temp[i - 1]).reshape(num, num), cmap='gray')
        #         except:
        #             break
        #         plt.axis('off')
        #         plt.title(str(i))
        #     plt.show()
        #     print('-------------------------------------')
        #
        # count = 0
        # for temp in all_data:
        #     fig = plt.figure(figsize=(30, 30))
        #     columns = 10
        #     rows = 10
        #     for i in range(1, columns * rows + 1):
        #         try:
        #             num = int(np.sqrt(temp[i - 1].shape[0]))
        #             fig.add_subplot(rows, columns, i)
        #             plt.imshow(np.squeeze(temp[i - 1]).reshape(num, num), cmap='gray')
        #             plt.axis('off')
        #             plt.title(str(i))
        #         except:
        #             break
        #     count = count + 1
        #     if count == 5:
        #         print('-------------------------------------')
        #     plt.show()

    def clean_sources_ica(self, kurt_thresh=1, skew_thresh=0.2, verbose=True):
        # clean sources based on skewness and correlation
        self.cleaned_sources_ica, source_idx = clean_sources(self.res, kurt_thresh=kurt_thresh, skew_thresh=skew_thresh)
        # cleaned_A_ica = A_ica[source_idx]
        # cleaned_W_ica = W_ica[source_idx]

        if verbose:
            print('Number of cleaned sources: ', self.cleaned_sources_ica.shape[0])

    def cluster(self, fs, num_frames, clustering='mog', spike_thresh=5,
                keep_all_clusters=False, features='amp', verbose=True):
        if verbose:
            print('Clustering Sources with: ', clustering)

        t_start = 0 * pq.s
        t_stop = num_frames / float(fs) * pq.s

        if clustering == 'kmeans' or clustering == 'mog':
            # detect spikes and align
            detected_spikes = detect_and_align(self.cleaned_sources_ica, fs, self.res,
                                               t_start=t_start, t_stop=t_stop, n_std=spike_thresh)
            spike_amps = [sp.annotations['ica_amp'] for sp in detected_spikes]
            spike_trains, amps, nclusters, keep, score = \
                cluster_spike_amplitudes(detected_spikes, metric='cal',
                                         alg=clustering, features=features, keep_all=keep_all_clusters)
            if verbose:
                print('Number of spike trains after clustering: ', len(spike_trains))
            self.sst, self.independent_spike_idx, dup = \
                reject_duplicate_spiketrains(spike_trains, sources=self.cleaned_sources_ica)
            if verbose:
                print('Number of spike trains after duplicate rejection: ', len(self.sst))
        else:
            raise Exception("Only 'mog' and 'kmeans' clustering methods are implemented")

    def set_times_labels(self, fs):
        times = np.array([], dtype=int)
        labels = np.array([])
        if 'ica_source' in self.sst[0].annotations.keys():
            self.independent_spike_idx = [s.annotations['ica_source'] for s in self.sst]
        for i_s, st in enumerate(self.sst):
            times = np.concatenate((times, (st.times.magnitude * fs).astype(int)))
            labels = np.concatenate((labels, np.array([i_s + 1] * len(st.times))))

        return NumpySorting.from_times_labels(times.astype(int), labels, fs)

