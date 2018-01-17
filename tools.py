# Helper functions

import numpy as np
import quantities as pq
from quantities import Quantity
import elephant
# import scipy.signal as ss
# from scipy.optimize import curve_fit
import os
from os.path import join
import matplotlib.pylab as plt

def load(filename):
    '''Generic loading of cPickled objects from file'''
    import pickle
    
    filen = open(filename,'rb')
    obj = pickle.load(filen)
    filen.close()
    return obj

# def apply_pca(data):
#     '''
#     :param data: T x N numpy array where N is neurons and T time or trials
#     :return: coeff, latent, projections
#     '''

#     pca = PCA(n_components=data.shape[1])
#     pca.fit(data)

#     coeff = pca.components_
#     latent = pca.explained_variance_ratio_
#     projections = np.dot(data, np.transpose(coeff))

#     return coeff, latent, projections




# from sklearn.utils.extmath                                                                                          

def svd_flip(u, v, u_based_decision=True):
    """Sign correction to ensure deterministic output from SVD.                                                       
    Adjusts the columns of u and the rows of v such that the loadings in the                                          
    columns in u that are largest in absolute value are always positive.                                              
    Parameters                                                                                                        
    ----------                                                                                                        
    u, v : ndarray                                                                                                    
        u and v are the output of `linalg.svd` or                                                                     
        `sklearn.utils.extmath.randomized_svd`, with matching inner dimensions                                        
        so one can compute `np.dot(u * s, v)`.                                                                        
    u_based_decision : boolean, (default=True)                                                                        
        If True, use the columns of u as the basis for sign flipping.                                                 
        Otherwise, use the rows of v. The choice of which variable to base the                                        
        decision on is generally algorithm dependent.                                                                 
    Returns                                                                                                           
    -------                                                                                                           
    u_adjusted, v_adjusted : arrays with the same dimensions as the input.                                            
    """
    if u_based_decision:
        # columns of u, rows of v                                                                                     
        max_abs_cols = np.argmax(np.abs(u), axis=0)
        signs = np.sign(u[max_abs_cols, xrange(u.shape[1])])
        u *= signs
        v *= signs[:, np.newaxis]
    else:
        # rows of v, columns of u                                                                                     
        max_abs_rows = np.argmax(np.abs(v), axis=1)
        signs = np.sign(v[xrange(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs[:, np.newaxis]
    return u, v


def apply_pca(data, n_components = None, bias=False, standardize_vars=False,flip=True):
    """                                                                                                               
    Principal component analysis                                                                                      
    """
    # Preprocessing the data                                                                                          
    data -= np.mean(data,0)     # subtract the mean (centering the columns)                                           
    if bias:
        data /= np.sqrt(data.shape[0]) # biased covariance PCA                                                        
    else:
        data /= np.sqrt(data.shape[0]-1) # unbiased covariance PCA                                                    
    if standardize_vars:
        data /= np.linalg.norm(data, 0) # correlation PCA                                                             
    # perform the SVD                                                                                                 
    P,Delta,Qt = np.linalg.svd(data,full_matrices=False)
    # flip eigenvectors' sign                                                                                         
    if flip:
        P,Qt = svd_flip(P,Qt,u_based_decision=False)

    explained_variance = (Delta ** 2)
    total_var = explained_variance.sum()
    explained_variance_ratio = explained_variance / total_var

    if n_components:
        Qt = Qt[:n_components]
        explained_variance_ratio = explained_variance_ratio[:n_components]
    scores=np.dot(data,Qt.T)
    return Qt.T,explained_variance_ratio,scores


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    
    from math import factorial
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


def load_EAP_data(spike_folder, cell_names, all_categories,samples_per_cat=None):

    print "Loading spike data ..."
    spikelist = [f for f in os.listdir(spike_folder) if f.startswith('e_spikes') and  any(x in f for x in cell_names)]
    loclist = [f for f in os.listdir(spike_folder) if f.startswith('e_pos') and  any(x in f for x in cell_names)]
    rotlist = [f for f in os.listdir(spike_folder) if f.startswith('e_rot') and  any(x in f for x in cell_names)]

    cells, occurrences = np.unique(sum([[f.split('_')[4]]*int(f.split('_')[2]) for f in spikelist],[]), return_counts=True)
    occ_dict = dict(zip(cells,occurrences))
    spikes_list = []

    loc_list = []
    rot_list = []
    category_list = []
    etype_list = []
    morphid_list = []

    spikelist = sorted(spikelist)
    loclist = sorted(loclist)
    rotlist = sorted(rotlist)

    loaded_categories = set()
    ignored_categories = set()

    for idx, f in enumerate(spikelist):
        category = f.split('_')[4]
        samples = int(f.split('_')[2])
        if samples_per_cat is not None:
            samples_to_read = int(float(samples)/occ_dict[category]*samples_per_cat)
        else:
            samples_to_read = samples
        etype = f.split('_')[5]
        morphid = f.split('_')[6]
        print 'loading ', samples_to_read , ' samples for cell type: ', f
        if category in all_categories:
            spikes = np.load(join(spike_folder, f)) # [:spikes_per_cell, :, :]
            spikes_list.extend(spikes[:samples_to_read])
            locs = np.load(join(spike_folder, loclist[idx])) # [:spikes_per_cell, :]
            loc_list.extend(locs[:samples_to_read])
            rots = np.load(join(spike_folder, rotlist[idx])) # [:spikes_per_cell, :]
            rot_list.extend(rots[:samples_to_read])
            category_list.extend([category] * samples_to_read)
            etype_list.extend([etype] * samples_to_read)
            morphid_list.extend([morphid] * samples_to_read)
            loaded_categories.add(category)
        else:
            ignored_categories.add(category)

    print "Done loading spike data ..."
    print "Loaded categories", loaded_categories
    print "Ignored categories", ignored_categories
    return np.array(spikes_list), np.array(loc_list), np.array(rot_list), np.array(category_list, dtype=str), \
        np.array(etype_list, dtype=str), np.array(morphid_list, dtype=int), loaded_categories

def load_validation_data(validation_folder,load_mcat=False):
    print "Loading validation spike data ..."

    spikes = np.load(join(validation_folder, 'val_spikes.npy'))  # [:spikes_per_cell, :, :]
    feat = np.load(join(validation_folder, 'val_feat.npy'))  # [:spikes_per_cell, :, :]
    locs = np.load(join(validation_folder, 'val_loc.npy'))  # [:spikes_per_cell, :]
    rots = np.load(join(validation_folder, 'val_rot.npy'))  # [:spikes_per_cell, :]
    cats = np.load(join(validation_folder, 'val_cat.npy'))
    if load_mcat:
        mcats = np.load(join(validation_folder, 'val_mcat.npy'))
        print "Done loading spike data ..."
        return np.array(spikes), np.array(feat), np.array(locs), np.array(rots), np.array(cats),np.array(mcats)
    else:
        print "Done loading spike data ..."
        return np.array(spikes), np.array(feat), np.array(locs), np.array(rots), np.array(cats)

def load_vm_im(vm_im_folder, cell_names, all_categories):
    
    print "Loading membrane potential and currents data ..."
    vmlist = [f for f in os.listdir(vm_im_folder) if f.startswith('v_spikes') and  any(x in f for x in cell_names)]
    imlist = [f for f in os.listdir(vm_im_folder) if f.startswith('i_spikes') and  any(x in f for x in cell_names)]
    print vmlist
    cat_list = [f.split('_')[3] for f in vmlist]
    entries_in_category = {cat: cat_list.count(cat) for cat in all_categories if cat in all_categories}
    # print "Number of cells in each category", entries_in_category

    vmlist = sorted(vmlist)
    imlist = sorted(imlist)

    vm_list = []
    im_list = []
    category_list = []

    loaded_categories = []
    ignored_categories = []

    for idx, f in enumerate(vmlist):
        category = f.split('_')[3]

        if category in all_categories:
            vm = np.load(join(vm_im_folder, f)) # [:spikes_per_cell, :, :]
            vm_list.append(vm)
            im = np.load(join(vm_im_folder, imlist[idx])) # [:spikes_per_cell, :]
            im_list.append(im)
            loaded_categories.append(category)
        else:
            ignored_categories.append(category)

    print "Done loading spike data ..."
    print "Loaded categories", loaded_categories
    print "Ignored categories", ignored_categories
    return np.array(vm_list), im_list, np.array(loaded_categories, dtype=str)



def get_EAP_features(EAP,feat_list,dt=None,EAP_times=None,threshold_detect=5.,normalize=False):
    ''' extract features specified in feat_list from EAP
    '''
    reference_mode = 't0'
    if EAP_times is not None and dt is not None:
        test_dt = (EAP_times[-1]-EAP_times[0])/(len(EAP_times)-1)
        if dt != test_dt:
            raise ValueError('EAP_times and dt do not match.')
    elif EAP_times is not None:
        dt = (EAP_times[-1]-EAP_times[0])/(len(EAP_times)-1)
    elif dt is not None:
        EAP_times = np.arange(EAP.shape[-1])*dt
    else:
        raise NotImplementedError('Please, specify either dt or EAP_times.')

    if len(EAP.shape)==1:
        EAP = np.reshape(EAP,[1,1,-1])
    elif len(EAP.shape)==2:
        EAP = np.reshape(EAP,[1,EAP.shape[0],EAP.shape[1]])
    if len(EAP.shape)!= 3:
        raise ValueError('Cannot handle EAPs with shape',EAP.shape)

    if normalize:
        signs = np.sign(np.min(EAP.reshape([EAP.shape[0],-1]),axis=1))
        norm = np.abs(np.min(EAP.reshape([EAP.shape[0],-1]),axis=1))
        EAP = np.array([EAP[i]/n if signs[i]>0 else EAP[i]/n-2. for i,n in enumerate(norm)])

    features = {}
    
    amps = np.zeros((EAP.shape[0], EAP.shape[1]))
    na_peak = np.zeros((EAP.shape[0], EAP.shape[1]))
    rep_peak = np.zeros((EAP.shape[0], EAP.shape[1]))
    if 'W' in feat_list:
        features['widths'] = np.zeros((EAP.shape[0], EAP.shape[1]))
    if 'F' in feat_list:
        features['fwhm'] = np.zeros((EAP.shape[0], EAP.shape[1]))
    if 'Aids' in feat_list:
        features['Aids'] = np.zeros((EAP.shape[0], EAP.shape[1],2),dtype=int)
    if 'Fids' in feat_list:
        features['Fids'] = []
    if 'FV' in feat_list:
        features['fwhm_V'] = np.zeros((EAP.shape[0], EAP.shape[1]))
    if 'Na' in feat_list:
        features['na'] = np.zeros((EAP.shape[0], EAP.shape[1]))
    if 'Rep' in feat_list:
        features['rep'] = np.zeros((EAP.shape[0], EAP.shape[1]))


    for i in range(EAP.shape[0]):
        # For spike with positive peak preceding sodium trough
        min_idx = np.array([np.unravel_index(EAP[i, e].argmin(), EAP[i, e].shape)[0] for e in
                                       range(EAP.shape[1])])
        na_peak[i, :] = np.array([EAP[i, e, min_idx[e]] for e in range(EAP.shape[1])])
        
        # max after NA trough
        max_idx = np.array([np.unravel_index(EAP[i, e, min_idx[e]:].argmax(),
                                                        EAP[i, e, min_idx[e]:].shape)[0]
                                       + min_idx[e] for e in range(EAP.shape[1])])
        rep_peak[i, :] = np.array([EAP[i, e, max_idx[e]] for e in range(EAP.shape[1])])

        if 'Aids' in feat_list:
            features['Aids'][i]=np.vstack((min_idx,max_idx)).T
            
        amps[i, :] = np.array([EAP[i, e, max_idx[e]]-EAP[i, e, min_idx[e]] for e in range(EAP.shape[1])])
        # If below 'detectable threshold, set amp and width to 0
        if normalize:
            too_low = np.where(amps[i, :] < threshold_detect/norm[i])
        else:
            too_low = np.where(amps[i, :] < threshold_detect)
        amps[i, too_low] = 0
           
        if 'W' in feat_list:
            features['widths'][i, :] = np.abs(EAP_times[max_idx] - EAP_times[min_idx])
            features['widths'][i, too_low] = EAP_times[-1]-EAP_times[0]
        if 'F' in feat_list:
            min_peak = np.min(EAP[i],axis=1)
            if reference_mode == 't0':
                # reference voltage is zeroth voltage entry
                fwhm_ref = np.array([EAP[i,e,0] for e in range(EAP.shape[1])])
            elif reference_mode == 'maxd2EAP':
                # reference voltage is taken at peak onset
                # peak onset is defined as id of maximum 2nd derivative of EAP
                peak_onset = np.array([np.argmax(savitzky_golay(EAP[i,e],5,2,deriv=2)[:min_idx[e]])
                                       for e in range(EAP.shape[1])])
                fwhm_ref = np.array([EAP[i,e,peak_onset[e]] for e in range(EAP.shape[1])])
            else:
                raise NotImplementedError('Reference mode ' + reference_mode + ' for FWHM calculation not implemented.')
            fwhm_V = (fwhm_ref + min_peak)/2. 
            id_trough = [np.where(EAP[i,e]<fwhm_V[e])[0] for e in range(EAP.shape[1])]
            if 'Fids' in feat_list:
                features['Fids'].append(id_trough)
            if 'FV' in feat_list:
                features['fwhm_V'][i,:]= fwhm_V

            # linear interpolation due to little number of data points during peak

            # features['fwhm'][i,:] = np.array([(len(t)+1)*dt+(fwhm_V[e]-EAP[i,e,t[0]-1])/(EAP[i,e,t[0]]-EAP[i,e,t[0]-1])*dt -(fwhm_V[e]-EAP[i,e,t[-1]])/(EAP[i,e,min(t[-1]+1,EAP.shape[2]-1)]-EAP[i,e,t[-1]])*dt if len(t)>0 else np.infty for e,t in enumerate(id_trough)])

            # no linear interpolation
            features['fwhm'][i,:] = [(id_trough[e][-1] - id_trough[e][0])*dt if len(id_trough[e])>1 \
                                     else EAP.shape[2] * dt for e in range(EAP.shape[1])]
            features['fwhm'][i, too_low] = EAP_times[-1]-EAP_times[0]

    if 'A' in feat_list:
        features.update({'amps': amps})
    if 'Na' in feat_list:
        features.update({'na': na_peak})
    if 'Rep' in feat_list:
        features.update({'rep': rep_peak})

    return features


def compute_maxspike_amp_width(spikes):
    
    dt = 2**-5

    widths = np.zeros(spikes.shape[0])
    amps = np.zeros(spikes.shape[0])
    fwhm = np.zeros(spikes.shape[0])

    for i in range(spikes.shape[0]):
        max_idx = spikes[i].argmax()
        min_idx = spikes[i].argmin()
        # For spike with positive peak preceding sodium trough
        if min_idx < max_idx:
            amps[i] = np.ptp(spikes[i])
            widths[i] = (max_idx - min_idx) * dt
        else:
            max_idx = spikes[i, min_idx:].argmax() + min_idx
            amps[i] = np.ptp(spikes[i, min_idx:-1])
            widths[i] = (max_idx - min_idx) * dt

        min_peak = np.min(spikes[i])
        id_trough = np.where(spikes[i] < 0.5*min_peak)[0]
        fwhm[i] = (id_trough[-1] - id_trough[0])*dt

    return amps, widths, fwhm


def get_binary_cat(categories, excit, inhib):
    binary_cat = []
    for i, cat in enumerate(categories):
        if cat in excit:
            binary_cat.append('EXCIT')
        elif cat in inhib:
            binary_cat.append('INHIB')

    return np.array(binary_cat, dtype=str)

def get_cat_from_label_idx(label_cat, categories):
    cat = []
    if len(label_cat) > 1:
        for i, cc in enumerate(label_cat):
            cat.append(categories[cc])
    elif len(label_cat) == 1:
        cat.append(categories[label_cat])

    return cat

def get_cat_from_hot_label(hot_label_cat, categories):
    cat = []
    if len(hot_label_cat.shape) == 2:
        for i, cc in enumerate(hot_label_cat):
            cat_id = int(np.where(cc == 1)[0])
            cat.append(categories[cat_id])
    elif len(hot_label_cat.shape) == 1:
        cat_id = int(np.where(hot_label_cat == 1)[0])
        cat.append(categories[cat_id])

    return cat


def convert_metype(mapfile,typ):
    ''' convert an m/e-type specifying string to integer or vice versa
    according to assignment in mapfile, or dictionary
    '''
    if type(filename)==str:
        mapdict = dict(np.loadtxt(mapfile,dtype=str))
    elif type(mapfile)==dict:
        mapdict = mapfile
    else:
        raise TypeError('Cannot handle type of mapfile.')

    if type(typ)==str and typ in mapdict.values():
        return int(float(mapdict.keys()[mapdict.values().index(typ)]))
    elif type(typ)==int:
        return mapdict.get(str(typ))
    else:
        raise ValueError('Can\'t handle m/e-type of the cell.')


###### STATISTICAL ANALYSIS #####

def annotate_stat_plot(x, y, df, ax=None, sig_val=0.05, ast_val=[0.001, 0.01, 0.05], keys=None):
    """
    Performs pairwise statistical analysis on pandas df attibute y based on
    factor x and annotate ax with statistical notes.
    -- shapiro test
    -- levene test
    -- pairwise t-test or mann-whitney U test
    Parameters
    ----------
    x, y : str
           factor and attribute in pandas databas
    df :  pandas dataframe
          dataframe used to perform test
    ax : matplotlib axis handles
         axis to be annotated
    Returns
    -------
    ax : annotated axis
    """
    import pandas as pd
    import scipy.stats as st

    group_by = df.groupby(x)
    ngroups = group_by.ngroups
    if keys is None:
        keys = group_by[y].groups.keys()

    pair_ttest = []
    pair_mannwhit = []
    pairs = []

    for idx_i, cat_i in enumerate(keys):
        for idx_j, cat_j in enumerate(keys):
            if idx_j > idx_i:
                # print idx_i, idx_j
                eq_pval = st.levene(group_by[y].get_group(cat_i), group_by[y].get_group(cat_j))[1]
                norm_pval_i = st.shapiro(group_by[y].get_group(cat_i))[1]
                norm_pval_j = st.shapiro(group_by[y].get_group(cat_j))[1]
                # print eq_pval, norm_pval_i, norm_pval_j
                if eq_pval > sig_val and norm_pval_i > sig_val and norm_pval_j > sig_val:
                    # print 'T-Test', idx_i, idx_j
                    pair_ttest.append([idx_i, idx_j])
                else:
                    # print 'Mann-Whitney', idx_i, idx_j
                    pair_mannwhit.append([idx_i, idx_j])

    # Perform ttest
    for pp_t in pair_ttest:
        pval = st.ttest_ind(group_by[y].get_group(keys[pp_t[0]]), group_by[y].get_group(keys[pp_t[1]]))[1]
        print 'pval t-test:  ', pval, np.mean(group_by[y].get_group(keys[pp_t[0]])), \
            np.mean(group_by[y].get_group(keys[pp_t[1]]))
        if pval < ast_val[0]:
            pp_t.append('***')
        elif pval < ast_val[1]:
            pp_t.append('**')
        elif pval < ast_val[2]:
            pp_t.append('*')
        else:
            pp_t.append('ns')

    # Perform mann-whitney
    for pp_m in pair_mannwhit:
        pval = st.mannwhitneyu(group_by[y].get_group(keys[pp_m[0]]), group_by[y].get_group(keys[pp_m[1]]))[1]
        # pval = st.ttest_ind(group_by[y].get_group(keys[pp_m[0]]), group_by[y].get_group(keys[pp_m[1]]),
        #                     equal_var=False)[1]

        print 'pval mann-whit:  ', pval, np.mean(group_by[y].get_group(keys[pp_m[0]])), \
            np.mean(group_by[y].get_group(keys[pp_m[1]]))
        if pval < ast_val[0]:
            pp_m.append('***')
        elif pval < ast_val[1]:
            pp_m.append('**')
        elif pval < ast_val[2]:
            pp_m.append('*')
        else:
            pp_m.append('ns')

    pairs = pair_ttest + pair_mannwhit

    # # annotate axis
    # for pp in pairs:
    #     # statistical annotation
    #     x1, x2 = pp[0], pp[1]  # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
    #     print x1, x2
    #     iqr = df[y].quantile(.75) - df[y].quantile(.25)
    #     y_shift, h, col = df[y].min() - (x1*(len(keys)-(x1-1)) + x2) * 0.1*iqr, 0.05*iqr, 'k'
    #     ax.plot([x1, x1, x2, x2], [y_shift, y_shift - h, y_shift - h, y_shift], lw=1.5, c=col)
    #     ax.text((x1 + x2) * .5, y_shift - 3*h, pp[2], ha='center', va='bottom', color=col)

    return pairs, keys


############ SPYICA ######################3

def filter_analog_signals(anas, freq, fs, filter_type='bandpass', order=3, copy_signal=False):
    """Filters analog signals with zero-phase Butterworth filter.
    The function raises an Exception if the required filter is not stable.

    Parameters
    ----------
    anas : np.array
           2d array of analog signals
    freq : list or float
           cutoff frequency-ies in Hz
    fs : float
         sampling frequency
    filter_type : string
                  'lowpass', 'highpass', 'bandpass', 'bandstop'
    order : int
            filter order

    Returns
    -------
    anas_filt : filtered signals
    """
    from scipy.signal import butter, filtfilt
    fn = fs / 2.
    fn = fn.rescale(pq.Hz)
    freq = freq.rescale(pq.Hz)
    band = freq / fn

    b, a = butter(order, band, btype=filter_type)

    if np.all(np.abs(np.roots(a)) < 1) and np.all(np.abs(np.roots(a)) < 1):
        print 'Filtering signals with ', filter_type, ' filter at ', freq, '...'
        if len(anas.shape) == 2:
            anas_filt = filtfilt(b, a, anas, axis=1)
        elif len(anas.shape) == 1:
            anas_filt = filtfilt(b, a, anas)
        return anas_filt
    else:
        raise ValueError('Filter is not stable')


def select_cells(loc, spikes, bin_cat, n_exc, n_inh, min_dist=25, bound_x=None, min_amp=None):
    pos_sel = []
    idxs_sel = []
    exc_idxs = np.where(bin_cat == 'EXCIT')[0]
    inh_idxs = np.where(bin_cat == 'INHIB')[0]

    if not bound_x:
        bound_x = [0, 100]
    if not min_amp:
        min_amp = 0

    for (idxs, num) in zip([exc_idxs, inh_idxs], [n_exc, n_inh]):
        n_sel = 0
        iter = 0
        while n_sel < num:
            # randomly draw a cell
            id_cell = idxs[np.random.permutation(len(idxs))[0]]
            dist = np.array([np.linalg.norm(loc[id_cell] - p) for p in pos_sel])

            iter += 1

            if np.any(dist < min_dist):
                # print 'NOPE! ', dist
                pass
            else:
                amp = np.max(np.ptp(spikes[id_cell]))
                if loc[id_cell][0] > bound_x[0] and loc[id_cell][0] < bound_x[1] and amp > min_amp:
                    # save cell
                    pos_sel.append(loc[id_cell])
                    idxs_sel.append(id_cell)
                    n_sel += 1

    return idxs_sel


def find_overlapping_spikes(spikes, thresh=0.7):
    overlapping_pairs = []

    for i in range(spikes.shape[0] - 1):
        temp_1 = spikes[i]
        max_ptp = (np.array([np.ptp(t) for t in temp_1]).max())
        max_ptp_idx = (np.array([np.ptp(t) for t in temp_1]).argmax())

        for j in range(i + 1, spikes.shape[0]):
            temp_2 = spikes[j]
            ptp_on_max = np.ptp(temp_2[max_ptp_idx])

            max_ptp_2 = (np.array([np.ptp(t) for t in temp_2]).max())

            max_peak = np.max([ptp_on_max, max_ptp])
            min_peak = np.min([ptp_on_max, max_ptp])

            if min_peak > thresh * max_peak and ptp_on_max > thresh * max_ptp_2:
                overlapping_pairs.append([i, j])  # , max_ptp_idx, max_ptp, ptp_on_max

    return np.array(overlapping_pairs)


def cubic_padding(spike, pad_len, fs, percent_mean=0.2):
    '''
    Cubic spline padding on left and right side to 0

    Parameters
    ----------
    spike
    pad_len
    fs

    Returns
    -------
    padded_template

    '''
    import scipy.interpolate as interp
    n_pre = int(pad_len[0] * fs)
    n_post = int(pad_len[1] * fs)

    padded_template = np.zeros((spike.shape[0], int(n_pre) + spike.shape[1] + n_post))
    splines = np.zeros((spike.shape[0], int(n_pre) + spike.shape[1] + n_post))

    for i, sp in enumerate(spike):
        # Remove inital offset
        padded_sp = np.zeros(n_pre + len(sp) + n_post)
        padded_t = np.arange(len(padded_sp))
        # initial_offset = np.mean(sp[0])
        # sp -= initial_offset

        x_pre = float(n_pre)
        x_pre_pad = np.arange(n_pre)
        x_post = float(n_post)
        x_post_pad = np.arange(n_post)[::-1]

        off_pre = sp[0]
        off_post = sp[-1]
        m_pre = sp[0] / x_pre
        m_post = sp[-1] / x_post

        padded_sp[:n_pre] = m_pre * x_pre_pad
        padded_sp[n_pre:-n_post] = sp
        padded_sp[-n_post:] = m_post * x_post_pad

        f = interp.interp1d(padded_t, padded_sp, kind='cubic')
        splines[i] = f(np.arange(len(padded_sp)))

        padded_template[i, :n_pre] = f(x_pre_pad)
        padded_template[i, n_pre:-n_post] = sp
        padded_template[i, -n_post:] = f(np.arange(n_pre + len(sp), n_pre + len(sp) + n_post))

    return padded_template, splines



def detect_and_align(sources, fs, recordings, t_start=None, t_stop=None, n_std=4, ref_period=2*pq.ms, upsample=8):
    '''

    Parameters
    ----------
    sources
    fs
    recordings
    t_start
    t_stop
    n_std
    ref_period
    upsample

    Returns
    -------

    '''
    import scipy.signal as ss
    import quantities as pq
    import neo

    idx_spikes = []
    idx_sources = []
    spike_trains = []
    times = (np.arange(sources.shape[1]) / fs).rescale('ms')
    unit = times[0].rescale('ms').units

    for s_idx, s in enumerate(sources):
        thresh = -n_std * np.median(np.abs(s) / 0.6745)
        # print s_idx, thresh
        idx_spike = np.where(s < thresh)[0]
        idx_spikes.append(idx_spike)

        n_pad = int(2 * pq.ms * fs.rescale('kHz'))
        sp_times = []
        sp_wf = []
        sp_rec_wf = []
        sp_amp = []

        for t in range(len(idx_spike) - 1):
            idx = idx_spike[t]
            # find single waveforms crossing thresholds
            if idx_spike[t + 1] - idx > 1 or t == len(idx_spike) - 2:  # single spike
                if idx - n_pad > 0 and idx + n_pad < len(s):
                    spike = s[idx - n_pad:idx + n_pad]
                    t_spike = times[idx - n_pad:idx + n_pad]
                    spike_rec = recordings[:, idx - n_pad:idx + n_pad]
                elif idx - n_pad < 0:
                    spike = s[:idx + n_pad]
                    spike = np.pad(spike, (np.abs(idx - n_pad), 0), 'constant')
                    t_spike = times[:idx + n_pad]
                    t_spike = np.pad(t_spike, (np.abs(idx - n_pad), 0), 'constant') * unit
                    spike_rec = recordings[:, :idx + n_pad]
                    spike_rec = np.pad(spike_rec, ((0, 0), (np.abs(idx - n_pad), 0)), 'constant')
                elif idx + n_pad > len(s):
                    spike = s[idx - n_pad:]
                    spike = np.pad(spike, (0, idx + n_pad - len(s)), 'constant')
                    t_spike = times[idx - n_pad:]
                    t_spike = np.pad(t_spike, (0, idx + n_pad - len(s)), 'constant') * unit
                    spike_rec = recordings[:, idx - n_pad:]
                    spike_rec = np.pad(spike_rec, ((0, 0), (0, idx + n_pad - len(s))), 'constant')

                # upsample and find minimume
                spike_up = ss.resample_poly(spike, upsample, 1)
                # times_up = ss.resample_poly(t_spike, upsample, 1)*unit
                times_up = np.linspace(t_spike[0].magnitude, t_spike[-1].magnitude, num=len(spike_up)) * unit

                min_idx_up = np.argmin(spike_up)
                min_amp_up = np.min(spike_up)
                min_time_up = times_up[min_idx_up]

                min_idx = np.argmin(spike)
                min_amp = np.min(spike)
                min_time = t_spike[min_idx]

                if len(sp_times) != 0:
                    if min_time_up - sp_times[-1] > ref_period:
                        sp_wf.append(spike_up)
                        sp_rec_wf.append(spike_rec)
                        sp_amp.append(min_amp_up)
                        sp_times.append(min_time_up)
                else:
                    sp_wf.append(spike_up)
                    sp_rec_wf.append(spike_rec)
                    sp_amp.append(min_amp_up)
                    sp_times.append(min_time_up)

        if t_start and t_stop:
            for i, sp in enumerate(sp_times):
                if sp.magnitude * unit < t_start:
                    sp_times[i] = t_start.rescale('ms')
                if  sp.magnitude * unit > t_stop:
                    sp_times[i] = t_stop.rescale('ms')
        elif t_stop:
            for i, sp in enumerate(sp_times):
                if sp > t_stop:
                    sp_times[i] = t_stop.rescale('ms')
        else:
            t_start = 0 * pq.s
            t_stop = sp_times[-1]

        spiketrain = neo.SpikeTrain([sp.magnitude for sp in sp_times] * unit, t_start=0 * pq.s, t_stop=t_stop,
                                    waveforms=np.array(sp_rec_wf))

        spiketrain.annotate(ica_amp=np.array(sp_amp))
        spiketrain.annotate(ica_wf=np.array(sp_wf))
        spike_trains.append(spiketrain)
        idx_sources.append(s_idx)

    return spike_trains


def reject_duplicate_spiketrains(sst, percent_threshold=0.8, min_spikes=3):
    # spiketrains
    import neo
    spike_trains = []
    idx_sources = []
    duplicates = []
    for i, sp_times in enumerate(sst):
        # check if overlapping with another source
        t_jitt = 2 * pq.ms
        counts = []
        for j, sp in enumerate(sst):
            count = 0
            if i != j:
                for t_i in sp_times:
                    id_over = np.where((sp > t_i - t_jitt) & (sp < t_i + t_jitt))[0]
                    if len(id_over) != 0:
                        count += 1
                if count >= percent_threshold * len(sp_times):
                    if [i, j] not in duplicates and [j, i] not in duplicates:
                        print 'Found duplicate spike trains: ', i, j
                        duplicates.append([i, j])
                counts.append(count)

    duplicates = np.array(duplicates)
    if len(duplicates) > 0:
        for i, sp_times in enumerate(sst):
            if i not in duplicates[:, 1]:
                # rej ect spiketrains with less than 3 spikes...
                if len(sp_times) >= min_spikes:
                    spike_trains.append(sp_times)
                    idx_sources.append(i)
    else:
        spike_trains = sst

    return spike_trains, idx_sources, duplicates


def integrate_sources(sources):
    '''

    Parameters
    ----------
    sources

    Returns
    -------

    '''
    integ_source = np.zeros_like(sources)

    for s, sor in enumerate(sources):
        partial_sum = 0
        for t, s_t in enumerate(sor):
            partial_sum += s_t
            integ_source[s, t] = partial_sum

    return integ_source


def clean_sources(sources, corr_thresh=0.4, skew_thresh=0.5):
    '''

    Parameters
    ----------
    s
    corr_thresh
    skew_thresh

    Returns
    -------

    '''
    import scipy.stats as stat
    import scipy.signal as ss

    sk = stat.skew(sources, axis=1)
    ku = stat.kurtosis(sources, axis=1)

    high_sk = np.where(np.abs(sk) >= skew_thresh)[0]
    low_sk = np.where(np.abs(sk) < skew_thresh)[0]
    sources_sp = sources[high_sk]
    sources_disc = sources[low_sk]

    # compute correlation matrix
    corr = np.zeros((sources_sp.shape[0], sources_sp.shape[0]))
    max_lag = np.zeros((sources_sp.shape[0], sources_sp.shape[0]))
    for i in range(sources_sp.shape[0]):
        s_i = sources_sp[i]
        for j in range(i + 1, sources_sp.shape[0]):
            s_j = sources_sp[j]
            cmat = crosscorrelation(s_i, s_j, maxlag=50)
            # cmat = ss.correlate(s_i, s_j)
            corr[i, j] = np.max(np.abs(cmat))
            max_lag[i, j] = np.argmax(np.abs(cmat))

    sources_keep, sources_discard = sources[high_sk], sources[low_sk]

    corr_idx = np.argwhere(corr > corr_thresh)
    sk_keep = stat.skew(sources_keep, axis=1)

    # remove smaller skewnesses
    remove_ic = []
    for idxs in corr_idx:
        sk_pair = sk_keep[idxs]
        remove_ic.append(idxs[np.argmin(np.abs(sk_pair))])
    remove_ic = np.array(remove_ic)

    if len(remove_ic) != 0:
        mask = np.array([True] * len(sources_keep))
        mask[remove_ic] = False

        spike_sources = sources_keep[mask]
        source_idx = high_sk[mask]
    else:
        spike_sources = sources_keep
        source_idx = high_sk


    sk_sp = stat.skew(spike_sources, axis=1)
    # invert sources with positive skewness
    spike_sources[sk_sp > 0] = -spike_sources[sk_sp > 0]

    return spike_sources, source_idx, corr_idx



def crosscorrelation(x, y, maxlag):
    """
    Cross correlation with a maximum number of lags.

    `x` and `y` must be one-dimensional numpy arrays with the same length.

    This computes the same result as
        numpy.correlate(x, y, mode='full')[len(a)-maxlag-1:len(a)+maxlag]

    The return vaue has length 2*maxlag + 1.
    """
    from numpy.lib.stride_tricks import as_strided

    py = np.pad(y.conj(), 2*maxlag, mode='constant')
    T = as_strided(py[2*maxlag:], shape=(2*maxlag+1, len(y) + 2*maxlag),
                   strides=(-py.strides[0], py.strides[0]))
    px = np.pad(x, maxlag, mode='constant')
    return T.dot(px)


def bin_spiketimes(spike_times, fs=None, T=None, t_stop=None):
    '''

    Parameters
    ----------
    spike_times
    fs
    T

    Returns
    -------

    '''
    import elephant.conversion as conv
    import neo
    resampled_mat = []
    binned_spikes = []
    spiketrains = []

    if isinstance(spike_times[0], neo.SpikeTrain):
        unit = spike_times[0].units
        spike_times = [st.times.magnitude for st in spike_times]*unit
    for st in spike_times:
        if t_stop:
            t_st = t_stop.rescale(pq.ms)
        else:
            t_st = st[-1].rescale(pq.ms)
        st_pq = [s.rescale(pq.ms).magnitude for s in st]*pq.ms
        spiketrains.append(neo.SpikeTrain(st_pq, t_st))
    if not fs and not T:
        print 'Provide either sampling frequency fs or time period T'
    elif fs:
        if not isinstance(fs, Quantity):
            raise ValueError("fs must be of type pq.Quantity")
        binsize = 1./fs
        binsize.rescale('ms')
        resampled_mat = []
        spikes = conv.BinnedSpikeTrain(spiketrains, binsize=binsize)
        for sts in spiketrains:
            spikes = conv.BinnedSpikeTrain(sts, binsize=binsize)
            resampled_mat.append(np.squeeze(spikes.to_array()))
            binned_spikes.append(spikes)
    elif T:
        binsize = T
        if not isinstance(T, Quantity):
            raise ValueError("T must be of type pq.Quantity")
        binsize.rescale('ms')
        resampled_mat = []
        for sts in spiketrains:
            spikes = conv.BinnedSpikeTrain(sts, binsize=binsize)
            resampled_mat.append(np.squeeze(spikes.to_array()))
            binned_spikes.append(spikes)


    return np.array(resampled_mat), binned_spikes


def ISI_amplitude_modulation(st, mrand=1, sdrand=0.05, n_spikes=1, exp=0.5, mem_ISI = 10*pq.ms):

    import elephant.statistics as stat

    ISI = stat.isi(st).rescale('ms')
    # mem_ISI = 2*mean_ISI
    amp_mod = np.zeros(len(st))
    amp_mod[0] = sdrand*np.random.randn() + mrand
    cons = np.zeros(len(st))

    for i, isi in enumerate(ISI):
        if n_spikes == 1:
            if isi > mem_ISI:
                amp_mod[i + 1] = sdrand * np.random.randn() + mrand
            else:
                amp_mod[i + 1] = isi.magnitude ** exp * (1. / mem_ISI.magnitude ** exp) + sdrand * np.random.randn()
        else:
            consecutive = 0
            bursting=True
            while consecutive < n_spikes and bursting:
                if i-consecutive >= 0:
                    if ISI[i-consecutive] > mem_ISI:
                        bursting = False
                    else:
                        consecutive += 1
                else:
                    bursting = False

            if consecutive == 0:
                amp_mod[i + 1] = sdrand * np.random.randn() + mrand
            elif consecutive==1:
                amp = (isi / float(consecutive)) ** exp * (1. / mem_ISI.magnitude ** exp)
                # scale std by amp
                amp_mod[i + 1] = amp + amp * sdrand * np.random.randn()
            else:
                if i != len(ISI):
                    isi_mean = np.mean(ISI[i-consecutive+1:i+1])
                else:
                    isi_mean = np.mean(ISI[i - consecutive + 1:])
                amp = (isi_mean/float(consecutive)) ** exp * (1. / mem_ISI.magnitude ** exp)
                # scale std by amp
                amp_mod[i + 1] = amp + amp * sdrand * np.random.randn()

            cons[i+1] = consecutive

    return amp_mod, cons

def cluster_spike_amplitudes(spike_amps, sst, metric='cal', min_sihlo=0.8, min_cal=150, max_clusters=4,
                             alg='kmeans'):
    #TODO keep remove cluster as separate spike trains and search among them if unique
    '''

    Parameters
    ----------
    spike_amps
    sst
    metric
    min_sihlo
    min_cal
    max_clusters

    Returns
    -------

    '''
    from sklearn.metrics import silhouette_score, calinski_harabaz_score
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture

    import neo

    nclusters = np.zeros(len(spike_amps))

    silhos = np.zeros(len(spike_amps))
    cal_hars = np.zeros(len(spike_amps))

    reduced_amps = []
    reduced_sst = []
    keep_id = []

    for i, amps in enumerate(spike_amps):
        silho = 0
        cal_har = 0
        keep_going = True

        if len(amps) > 2:
            for k in range(2, max_clusters):
                if alg=='kmeans':
                    kmeans_new = KMeans(n_clusters=k, random_state=0)
                    kmeans_new.fit(amps.reshape(-1, 1))
                    labels = kmeans_new.predict(amps.reshape(-1, 1))
                elif alg=='mog':
                    gmm_new = GaussianMixture(n_components=k, covariance_type='full')
                    gmm_new.fit(amps.reshape(-1, 1))
                    labels = gmm_new.predict(amps.reshape(-1, 1))

                if len(np.unique(labels)) > 1:
                    silho_new = silhouette_score(amps.reshape(-1, 1), labels)
                    cal_har_new = calinski_harabaz_score(amps.reshape(-1, 1), labels)
                    if silho_new > silho:
                        silho = silho_new
                        if metric == 'silho':
                            nclusters[i] = k
                            if alg == 'kmeans':
                                kmeans = kmeans_new
                            elif alg == 'mog':
                                gmm = gmm_new
                            keep_going=False
                    if cal_har_new > cal_har:
                        cal_har = cal_har_new
                        if metric == 'cal':
                            nclusters[i] = k
                            if alg == 'kmeans':
                                kmeans = kmeans_new
                            elif alg == 'mog':
                                gmm = gmm_new
                            keep_going=False
                else:
                    keep_going=False
                    nclusters[i] = 1

                if not keep_going:
                    break

            if nclusters[i] != 1:
                if metric == 'silho':
                    if silho < min_sihlo:
                        nclusters[i] = 1
                        reduced_sst.append(sst[i])
                        reduced_amps.append(amps)
                        keep_id.append(range(len(sst[i])))
                    else:
                        highest_clust = np.argmin(kmeans.cluster_centers_)
                        highest_idx = np.where(labels==highest_clust)[0]
                        reduced_sst.append(sst[i][highest_idx])
                        reduced_amps.append(amps[highest_idx])
                        keep_id.append(highest_idx)
                elif metric == 'cal':
                    if cal_har < min_cal:
                        nclusters[i] = 1
                        reduced_sst.append(sst[i])
                        reduced_amps.append(amps)
                        keep_id.append(range(len(sst[i])))
                    else:
                        if alg=='kmeans':
                            highest_clust = np.argmin(kmeans.cluster_centers_)
                        elif alg == 'mog':
                            highest_clust = np.argmin(gmm.means_)
                        highest_idx = np.where(labels==highest_clust)[0]
                        red_spikes = sst[i][highest_idx]
                        if 'ica_amp' in red_spikes.annotations:
                            red_spikes.annotate(ica_amp=red_spikes.annotations['ica_amp'])
                        if 'ica_wf' in red_spikes.annotations:
                            red_spikes.annotate(ica_wf=red_spikes.annotations['ica_wf'])
                        reduced_sst.append(red_spikes)
                        reduced_amps.append(amps[highest_idx])
                        keep_id.append(highest_idx)

                silhos[i] = silho
                cal_hars[i] = cal_har
            else:
                reduced_sst.append(sst[i])
                reduced_amps.append(amps)
                keep_id.append(range(len(sst[i])))
        else:
            reduced_sst.append(sst[i])
            reduced_amps.append(amps)
            keep_id.append(range(len(sst[i])))

    if metric == 'silho':
        score = silhos
    elif metric == 'cal':
        score = cal_hars

    return reduced_sst, reduced_amps, nclusters, keep_id, score



def evaluate_spiketrains(gtst, sst, t_jitt = 1*pq.ms, overlapping=False):
    '''

    Parameters
    ----------
    gtst
    sst
    t_jitt
    overlapping_pairs

    Returns
    -------

    '''
    import neo
    from elephant.spike_train_correlation import cch, corrcoef

    t_stop = gtst[0].t_stop

    or_mat, original_st = bin_spiketimes(gtst, T=1*pq.ms, t_stop=t_stop)
    pr_mat, predicted_st = bin_spiketimes(sst, T=1*pq.ms, t_stop=t_stop)

    cc_matr = np.zeros((or_mat.shape[0], pr_mat.shape[0]))

    for o, o_st in enumerate(original_st):
        for p, p_st in enumerate(predicted_st):
            cc, bin_ids = cch(o_st, p_st, kernel=np.hamming(100))
            central_bin = len(cc) // 2
            # normalize by number of spikes
            cc_matr[o, p] = np.max(cc[central_bin-10:central_bin+10]) / (len(gtst[o]) + len(sst[p])) # (abs(len(gtst[o]) - len(sst[p])) + 1)
    cc_matr /= np.max(cc_matr)

    best_pairs = np.array([])
    assigned_sst = []
    assigned_gtst = []

    sorted_rows = []
    sorted_idxs = []

    # find best matching pairs (based in CCH)
    print 'Finding best ST matches'
    for i, row in enumerate(cc_matr):
        sorted_rows.append(np.sort(row)[::-1])
        sorted_idxs.append(np.argsort(row)[::-1])

    sorted_rows = np.array(sorted_rows)
    sorted_idxs = np.array(sorted_idxs)
    put_pairs = np.zeros((len(gtst), 2), dtype=int)

    for sp in range(len(gtst)):
        put_pairs[sp] = np.array([sp, sorted_idxs[sp, 0]], dtype=int)
    put_pairs = np.array(put_pairs)

    # reassign wrong assingments
    ass, count = np.unique(put_pairs[:, 1], return_counts=True)
    reassign_sst = ass[np.where(count > 1)]
    for st in reassign_sst:
        all_pairs = np.where(put_pairs[:, 1] == st)[0]
        sorted_pairs = all_pairs[np.argsort(sorted_rows[all_pairs, 0])[::-1][1:]]

        for pp in sorted_pairs:
            # print 'Row', pp
            assigned = False
            for i in range(1, sorted_rows.shape[1]):
                possible_st = sorted_idxs[pp, i]
                if possible_st not in ass:
                    assigned = True
                    # print 'Assign', pp, possible_st
                    put_pairs[pp] = np.array([pp, possible_st], dtype=int)
                    ass = np.append(ass, possible_st)
                    break
                if i == sorted_rows.shape[1] - 1:
                    # print 'Out of spiketrains'
                    put_pairs[pp] = np.array([-1, -1], dtype=int)

    # shift best match by max lag
    for gt_i, gt in enumerate(gtst):
        pair = put_pairs[gt_i]
        if pair[0] != -1:
            o_st = original_st[gt_i]
            p_st = predicted_st[pair[1]]

            cc, bin_ids = cch(o_st, p_st, kernel=np.hamming(50))
            central_bin = len(cc) // 2
            # normalize by number of spikes
            max_lag = np.argmax(cc[central_bin-5:central_bin+5])
            optimal_shift = (-5+max_lag)*pq.ms
            sst[pair[1]] -= optimal_shift
            idx_after = np.where(sst[pair[1]] > sst[pair[1]].t_stop)[0]
            idx_before = np.where(sst[pair[1]] < sst[pair[1]].t_start)[0]
            if len(idx_after) > 0:
                sst[pair[1]][idx_after] = sst[pair[1]].t_stop
            if len(idx_before) > 0:
                sst[pair[1]][idx_before] = sst[pair[1]].t_start

    # Evaluate
    t_jitt = 2*pq.ms
    print 'Finding TP'
    for gt_i, gt in enumerate(gtst):
        if put_pairs[gt_i, 0] != -1:
            lab_gt = np.array(['UNPAIRED'] * len(gt))
            st = sst[put_pairs[gt_i, 1]]
            lab_st = np.array(['UNPAIRED'] * len(st))
            # from gtst: TP, TPO, TPSO, FN, FNO, FNSO
            for sp_i, t_sp in enumerate(gt):
                id_sp = np.where((st > t_sp - t_jitt) & (st < t_sp + t_jitt))[0]
                if len(id_sp) == 1:
                    if 'overlap' in gt.annotations.keys():
                        if gt.annotations['overlap'][sp_i] == 'NO':
                            lab_gt[sp_i] = 'TP'
                            lab_st[id_sp] = 'TP'
                        elif gt.annotations['overlap'][sp_i] == 'O':
                            lab_gt[sp_i] = 'TPO'
                            lab_st[id_sp] = 'TPO'
                        elif gt.annotations['overlap'][sp_i] == 'SO':
                            lab_gt[sp_i] = 'TPSO'
                            lab_st[id_sp] = 'TPSO'
                    else:
                        lab_gt[sp_i] = 'TP'
                        lab_st[id_sp] = 'TP'
            st.annotate(labels=lab_st)
        else:
            lab_gt = np.array(['FN'] * len(gt))
        gt.annotate(labels=lab_gt)

    # if unpaired SST
    for st in sst:
        if 'labels' not in st.annotations:
            lab_st = np.array(['UNPAIRED'] * len(st))
            st.annotate(labels=lab_st)

    # find CL-CLO-CLSO
    print 'Finding CL'
    for gt_i, gt in enumerate(gtst):
        lab_gt = gt.annotations['labels']
        for lab_i, lab in enumerate(lab_gt):
            if lab == 'UNPAIRED':
                for st_i, st in enumerate(sst):
                    t_up = gt[lab_i]
                    id_sp = np.where((st > t_up - t_jitt) & (st < t_up + t_jitt))[0]
                    lab_st = st.annotations['labels']
                    if len(id_sp) == 1 and lab_st[id_sp] == 'UNPAIRED':
                        if 'overlap' in gt.annotations.keys():
                            if gt.annotations['overlap'][lab_i] == 'NO':
                                lab_gt[lab_i] = 'CL'
                                lab_st[id_sp] = 'CL'
                            elif gt.annotations['overlap'][lab_i] == 'O':
                                lab_gt[lab_i] = 'CLO'
                                lab_st[id_sp] = 'CLO'
                            elif gt.annotations['overlap'][lab_i] == 'SO':
                                lab_gt[lab_i] = 'CLSO'
                                lab_st[id_sp] = 'CLSO'
                        else:
                            lab_gt[lab_i] = 'CL'
                            lab_st[id_sp] = 'CL'
                    st.annotate(labels=lab_st)
        gt.annotate(labels=lab_gt)

    print 'Finding FP and FN'
    for gt_i, gt in enumerate(gtst):
        lab_gt = gt.annotations['labels']
        for lab_i, lab in enumerate(lab_gt):
            if lab == 'UNPAIRED':
                if 'overlap' in gt.annotations.keys():
                    if gt.annotations['overlap'][lab_i] == 'NO':
                        lab_gt[lab_i] = 'FN'
                    elif gt.annotations['overlap'][lab_i] == 'O':
                        lab_gt[lab_i] = 'FNO'
                    elif gt.annotations['overlap'][lab_i] == 'SO':
                        lab_gt[lab_i] = 'FNSO'
                else:
                    lab_gt[lab_i] = 'FN'
        gt.annotate(labels=lab_gt)

    for st_i, st in enumerate(sst):
        lab_st = st.annotations['labels']
        for lab_i, lab in enumerate(lab_st):
            if lab == 'UNPAIRED':
                    lab_st[lab_i] = 'FP'
        st.annotate(labels=lab_st)

    TOT_GT = sum([len(gt) for gt in gtst])
    TOT_ST = sum([len(st) for st in sst])
    total_spikes = TOT_GT + TOT_ST

    TP = sum([len(np.where('TP' == gt.annotations['labels'])[0]) for gt in gtst])
    TPO = sum([len(np.where('TPO' == gt.annotations['labels'])[0]) for gt in gtst])
    TPSO = sum([len(np.where('TPSO' == gt.annotations['labels'])[0]) for gt in gtst])

    print 'TP :', TP, TPO, TPSO, TP+TPO+TPSO

    CL = sum([len(np.where('CL' == gt.annotations['labels'])[0]) for gt in gtst]) + \
         sum([len(np.where('CL' == st.annotations['labels'])[0]) for st in sst])
    CLO = sum([len(np.where('CLO' == gt.annotations['labels'])[0]) for gt in gtst]) + \
          sum([len(np.where('CLO' == st.annotations['labels'])[0]) for st in sst])
    CLSO = sum([len(np.where('CLSO' == gt.annotations['labels'])[0]) for gt in gtst]) + \
           sum([len(np.where('CLSO' == st.annotations['labels'])[0]) for st in sst])

    print 'CL :', CL, CLO, CLSO, CL+CLO+CLSO

    FN = sum([len(np.where('FN' == gt.annotations['labels'])[0]) for gt in gtst])
    FNO = sum([len(np.where('FNO' == gt.annotations['labels'])[0]) for gt in gtst])
    FNSO = sum([len(np.where('FNSO' == gt.annotations['labels'])[0]) for gt in gtst])

    print 'FN :', FN, FNO, FNSO, FN+FNO+FNSO


    FP = sum([len(np.where('FP' == st.annotations['labels'])[0]) for st in sst])

    print 'FP :', FP

    print 'TOTAL: ', TOT_GT, TOT_ST, TP+TPO+TPSO+CL+CLO+CLSO+FN+FNO+FNSO+FP

    counts = {'TP': TP, 'TPO': TPO, 'TPSO': TPSO,
              'CL': CL, 'CLO': CLO, 'CLSO': CLSO,
              'FN': FN, 'FNO': FNO, 'FNSO': FNSO,
              'FP': FP, 'TOT': total_spikes, 'TOT_GT': TOT_GT, 'TOT_ST': TOT_ST}


    return counts, put_pairs, cc_matr


def annotate_overlapping(gtst, t_jitt = 1*pq.ms, overlapping_pairs=None, verbose=False):
    # find overlapping spikes
    for i, st_i in enumerate(gtst):
        if verbose:
            print 'SPIKETRAIN ', i
        over = np.array(['NO'] * len(st_i))
        for i_sp, t_i in enumerate(st_i):
            for j, st_j in enumerate(gtst):
                if i != j:
                    # find overlapping
                    id_over = np.where((st_j > t_i - t_jitt) & (st_j < t_i + t_jitt))[0]
                    if not np.any(overlapping_pairs):
                        if len(id_over) != 0:
                            over[i_sp] = 'O'
                            # if verbose:
                            #     print 'found overlap! spike 1: ', i, t_i, ' spike 2: ', j, st_j[id_over]
                    else:
                        pair = [i, j]
                        pair_i = [j, i]
                        if np.any([np.all(pair == p) for p in overlapping_pairs]) or \
                                np.any([np.all(pair_i == p) for p in overlapping_pairs]):
                            if len(id_over) != 0:
                                over[i_sp] = 'SO'
                                # if verbose:
                                #     print 'found spatial overlap! spike 1: ', i, t_i, ' spike 2: ', j, st_j[id_over]
                        else:
                            if len(id_over) != 0:
                                over[i_sp] = 'O'
                                # if verbose:
                                #     print 'found overlap! spike 1: ', i, t_i, ' spike 2: ', j, st_j[id_over]
        st_i.annotate(overlap=over)

def raster_plots(st, bintype=False, ax=None, overlap=None, color_st=None, fs=10):
    '''

    Parameters
    ----------
    st
    bintype
    ax

    Returns
    -------

    '''
    import matplotlib.pylab as plt
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    for i, spiketrain in enumerate(st):
        t = spiketrain.rescale(pq.s)
        if bintype:
            if spiketrain.annotations['bintype'] == 'EXCIT':
                ax.plot(t, i * np.ones_like(t), 'b.', markersize=5)
            elif spiketrain.annotations['bintype'] == 'INHIB':
                ax.plot(t, i * np.ones_like(t), 'r.', markersize=5)
        else:
            if overlap is None:
                if np.any(color_st):
                    import seaborn as sns
                    colors = sns.color_palette("Paired", len(color_st))
                    if i in color_st:
                        idx = np.where(color_st==i)[0][0]
                        ax.plot(t, i * np.ones_like(t), '.', color=colors[idx], markersize=5)
                    else:
                        ax.plot(t, i * np.ones_like(t), 'k.', markersize=5)
                else:
                    ax.plot(t, i * np.ones_like(t), 'k.', markersize=5)
            else:
                for j, t_sp in enumerate(spiketrain):
                    if spiketrain.annotations['overlap'][j] == 'SO':
                        ax.plot(t_sp, i, 'r.', markersize=5)
                    elif spiketrain.annotations['overlap'][j] == 'O':
                        ax.plot(t_sp, i, 'g.', markersize=5)
                    elif spiketrain.annotations['overlap'][j] == 'NO':
                        ax.plot(t_sp, i, 'k.', markersize=5)

    ax.axis('tight')
    ax.set_xlim([st[0].t_start.rescale(pq.s), st[0].t_stop.rescale(pq.s)])
    ax.set_xlabel('Time (ms)', fontsize=fs)
    ax.set_ylabel('Spike Train Index', fontsize=fs)
    plt.gca().tick_params(axis='both', which='major')

    return ax

###### KLUSTA #########
def save_binary_format(filename, signal, spikesorter='klusta', dtype='float32'):
    """Saves analog signals into klusta (time x chan) or spyking
    circus (chan x time) binary format (.dat)

    Parameters
    ----------
    filename : string
               absolute path (_klusta.dat or _spycircus.dat are appended)
    signal : np.array
             2d array of analog signals
    spikesorter : string
                  'klusta' or 'spykingcircus'

    Returns
    -------
    """
    if spikesorter is 'klusta':
        fdat = filename + '_klusta.dat'
        print('Saving ', fdat)
        with open(fdat, 'wb') as f:
            np.transpose(np.array(signal, dtype=dtype)).tofile(f)
    elif spikesorter is 'spykingcircus':
        fdat = filename + '_spycircus.dat'
        print('Saving ', fdat)
        with open(fdat, 'wb') as f:
            np.array(signal, dtype=dtype).tofile(f)
    elif spikesorter == 'kilosort' or spikesorter == 'none':
        if not filename.endswith('dat'):
            filename += '.dat'
        print('saving ', filename)
        with open(filename, 'wb') as f:
            np.transpose(np.array(signal, dtype=dtype)).tofile(f)


def create_klusta_prm(pathname, prb_path, nchan=32, fs=30000,
                      klusta_filter=True, filter_low=300, filter_high=6000):
    """Creates klusta .prm files, with spikesorting parameters

    Parameters
    ----------
    pathname : string
               absolute path (_klusta.dat or _spycircus.dat are appended)
    prbpath : np.array
              2d array of analog signals
    nchan : int
            number of channels
    fs: float
        sampling frequency
    klusta_filter : bool
        filter with klusta or not
    filter_low: float
                low cutoff frequency (if klusta_filter is True)
    filter_high : float
                  high cutoff frequency (if klusta_filter is True)
    Returns
    -------
    full_filename : absolute path of .prm file
    """
    assert pathname is not None
    abspath = os.path.abspath(pathname)
    assert prb_path is not None
    prb_path = os.path.abspath(prb_path)
    full_filename = abspath + '.prm'

    if isinstance(fs, Quantity):
        fs = fs.rescale('Hz').magnitude

    extract_s_before = int(5*1e-4*fs)
    extract_s_after = int(1*1e-3*fs)

    print full_filename
    print('Saving ', full_filename)
    with open(full_filename, 'w') as f:
        f.write('\n')
        f.write('experiment_name = ' + "r'" + str(abspath) + '_klusta' + "'" + '\n')
        f.write('prb_file = ' + "r'" + str(prb_path) + "'")
        f.write('\n')
        f.write('\n')
        f.write("traces = dict(\n\traw_data_files=[experiment_name + '.dat'],\n\tvoltage_gain=1.,"
                "\n\tsample_rate="+str(fs)+",\n\tn_channels="+str(nchan)+",\n\tdtype='float32',\n)")
        f.write('\n')
        f.write('\n')
        f.write("spikedetekt = dict(")
        if klusta_filter:
            f.write("\n\tfilter_low="+str(filter_low)+",\n\tfilter_high="+str(filter_high)+","
                    "\n\tfilter_butter_order=3,\n\tfilter_lfp_low=0,\n\tfilter_lfp_high=300,\n")
        f.write("\n\tchunk_size_seconds=1,\n\tchunk_overlap_seconds=.015,\n"
                "\n\tn_excerpts=50,\n\texcerpt_size_seconds=1,"
                "\n\tthreshold_strong_std_factor=4.5,\n\tthreshold_weak_std_factor=2,\n\tdetect_spikes='negative',"
                "\n\n\tconnected_component_join_size=1,\n"
                "\n\textract_s_before="+str(extract_s_before)+",\n\textract_s_after="+str(extract_s_after)+",\n"
                "\n\tn_features_per_channel=3,\n\tpca_n_waveforms_max=10000,\n)")
        f.write('\n')
        f.write('\n')
        f.write("klustakwik2 = dict(\n\tnum_starting_clusters=50,\n)")
                # "\n\tnum_cpus=4,)")
    return full_filename

def export_prb_file(n_elec, electrode_name, pathname,
                    pos=None, adj_dist=None, graph=True, geometry=True, separate_channels=False):

    assert pathname is not None
    abspath = os.path.abspath(pathname)
    full_filename = join(abspath, electrode_name + '.prb')

    # find adjacency graph
    if graph:
        if pos is not None and adj_dist is not None:
            adj_graph = []
            for el1, el_pos1 in enumerate(pos):
                for el2, el_pos2 in enumerate(pos):
                    if el1 != el2:
                        if np.linalg.norm(el_pos1 - el_pos2) < adj_dist:
                            adj_graph.append((el1, el2))

    print 'Saving ', full_filename
    with open(full_filename, 'w') as f:
        f.write('\n')
        f.write('channel_groups = {\n')
        if not separate_channels:
            f.write("    0: ")
            f.write("\n        {\n")
            f.write("           'channels': " + str(range(n_elec)) + ',\n')
            if graph:
                f.write("           'graph':  " + str(adj_graph) + ',\n')
            else:
                f.write("           'graph':  [],\n")
            if geometry:
                f.write("           'geometry':  {\n")
                for i, pos in enumerate(pos):
                    f.write('               ' + str(i) +': ' + str(tuple(pos[1:])) + ',\n')
                f.write('           }\n')
            f.write('       }\n}')
            f.write('\n')
        else:
            for elec in range(n_elec):
                f.write('    ' + str(elec) + ': ')
                f.write("\n        {\n")
                f.write("           'channels': [" + str(elec) + '],\n')
                f.write("           'graph':  [],\n")
                f.write('        },\n')
            f.write('}\n')

    return full_filename







