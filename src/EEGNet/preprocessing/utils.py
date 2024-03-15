import numpy as np
import scipy.io
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch


def change_annot_names(desc, enumerated=False):
    """ Change Annotations to Meaningful Names"""
    annot_name_changer = {'Stimulus/S  1': 'Switch',
                          'Stimulus/S200': 'EO',
                          'Stimulus/S210': 'EC',
                          'Stimulus/S208': 'EC',  # for sub-010126, this is the EC marker
                          }
    for idx, val in enumerate(desc):
        if val in annot_name_changer.keys():
            if enumerated:
                desc[idx] = annot_name_changer[val] + '_' + str(idx)
            else:
                desc[idx] = annot_name_changer[val]
    return desc


def segment_raw(raw, onsets, pattern, duration=60):
    # segment raw data into EC/EO
    raws = {'EC': [], 'EO': []}

    for k, v in zip(pattern, onsets):
        raws[k].append(raw.copy().crop(tmin=v, tmax=v+duration))  # crop 60s after switch

    # concatenate all EC segments
    raws['EC'][0].append(raws['EC'][1:])
    raws['EC'] = raws['EC'][0]

    # concatenate all EO segments
    raws['EO'][0].append(raws['EO'][1:])
    raws['EO'] = raws['EO'][0]

    return raws


def find_switch_onset_pattern(annot_dict):
    # only keep EC/EO markers and code them as 0/1
    new_annot = {}
    for k in annot_dict.keys():
        if 'EC' in k or 'EO' in k:
            new_annot[k] = annot_dict[k]

    # code EC/EO as 0/1
    keys = [0 if 'EC' in k else 1 for k in new_annot.keys()]

    # where switch between EC/EO happens
    switch = list(np.where(np.diff(np.array(keys)))[0] + 1)
    switch = [0] + switch  # append the beginning of the first segment

    # find the onsets and the pattern of the switches
    switch_onsets = [list(new_annot.values())[i] for i in switch]
    switch_keys = [keys[i] for i in switch]

    # change back the pattern to EC/EO
    switch_pattern = ['EC' if i == 0 else 'EO' for i in switch_keys]
    return switch_onsets, switch_pattern


def check_segments(raw):
    """ Make sure the segmentation is correct """
    uniq = np.unique(raw.annotations.description)
    # check 'Stimulus/S210' and 'Stimulus/S200' are not in the uniqes at the same time
    return not (('Stimulus/S210' in uniq) and ('Stimulus/S200' in uniq))


def get_channel_positions(data_dir):
    ch_positions = {}
    for path in Path(data_dir).glob('sub*'):
        sub = path.stem
        mat = scipy.io.loadmat(path / f'{sub}.mat')
        n_channels = len(mat['Channel'][0])
        ch_names = [mat['Channel'][0][i][0][0] for i in range(n_channels)]
        ch_names = [ch.split('_')[2] for ch in ch_names if ch != 'Reference']
        ch_pos = np.array([mat['Channel'][0][i][3].reshape(3, -1) for i in range(n_channels)]).squeeze()
        ch_positions[sub] = {ch: pos for ch, pos in zip(ch_names, ch_pos)}
    return ch_positions


def get_headpoints(data_dir):
    headpoints = {}
    for path in Path(data_dir).glob('sub*'):
        sub = path.stem
        mat = scipy.io.loadmat(path / f'{sub}.mat')
        x = mat['HeadPoints'][0][0][0][0]
        y = mat['HeadPoints'][0][0][0][1]
        z = mat['HeadPoints'][0][0][0][2]
        coord = np.vstack((x, y, z)).T
        labels = mat['HeadPoints'][0][0][1][0]
        first_set = {l[0]: i for l, i in zip(labels[:3], coord[:3])}
        second_set = {l[0]: i for l, i in zip(labels[3:], coord[3:])}
        headpoints[sub] = {'first': first_set, 'second': second_set}
    return headpoints


def __split_data(data, type, cut_point, shu_idx_train=None, shu_idx_test=None):
    if type == 'split_shuffle':
        assert shu_idx_train is not None and shu_idx_test is not None, 'shu_idx_train and shu_idx_test must be provided'
        train = data[:, :cut_point].flatten(0, 1)[shu_idx_train]
        test = data[:, cut_point:].flatten(0, 1)[shu_idx_test]
        return train, test
    elif type == 'shuffle_split':
        assert shu_idx_train is not None, 'shu_idx_train must be provided'
        data = data.flatten(0, 1)[shu_idx_train]
        train, test = data[:cut_point], data[cut_point:]
        return train, test
    elif type == 'no_shuffle':
        train = data[:, :cut_point].flatten(0, 1)
        test = data[:, cut_point:].flatten(0, 1)
        return train, test


def _split_sub(data, subject_ids, positions, y_cls, stratified=True, train_ratio=0.7):
    n_subjects = data.shape[0]
    if stratified:
        stratify_cls = y_cls[:, :, 1]
    else:
        stratify_cls = None
    train_ids, val_ids = train_test_split(torch.arange(n_subjects),
                                          train_size=train_ratio,
                                          stratify=stratify_cls,
                                          )
    train_idx = torch.where(torch.isin(subject_ids, train_ids))[0]
    val_idx = torch.where(torch.isin(subject_ids, val_ids))[0]
    X_train = data[train_idx].flatten(0, 1)
    X_test = data[val_idx].flatten(0, 1)
    subject_ids_train = subject_ids[train_idx].flatten(0, 1)
    subject_ids_test = subject_ids[val_idx].flatten(0, 1)
    positions_train = positions[train_idx].flatten(0, 1)
    positions_test = positions[val_idx].flatten(0, 1)
    y_cls_train = y_cls[train_idx].flatten(0, 1)
    y_cls_test = y_cls[val_idx].flatten(0, 1)

    return X_train, X_test, subject_ids_train, subject_ids_test, positions_train, positions_test, y_cls_train, y_cls_test


def _split_time(data,
                subject_ids,
                positions,
                y_cls,
                shuffling, train_ratio=0.7):
    if shuffling == 'split_shuffle':
        cut_point = int(data.shape[1] * train_ratio)  # cutoff
        X_train = data[:, :cut_point, :, :].flatten(0, 1)
        X_test = data[:, cut_point:, :, :].flatten(0, 1)
        sh_tr, sh_te = torch.randperm(X_train.shape[0]), torch.randperm(X_test.shape[0])
        X_train = X_train[sh_tr]
        X_test = X_test[sh_te]
        subject_ids_train, subject_ids_test = __split_data(subject_ids, 'split_shuffle', cut_point, sh_tr, sh_te)
        positions_train, positions_test = __split_data(positions, 'split_shuffle', cut_point, sh_tr, sh_te)
        y_cls_train, y_cls_test = __split_data(y_cls, 'split_shuffle', cut_point, sh_tr, sh_te)

    elif shuffling == 'shuffle_split':
        X_input = data.flatten(0, 1)
        sh = torch.randperm(X_input.shape[0])
        cut_point = int(X_input.shape[0] * train_ratio)
        X_input = X_input[sh]
        X_train, X_test = X_input[:cut_point], X_input[cut_point:]
        subject_ids_train, subject_ids_test = __split_data(subject_ids, 'shuffle_split', cut_point, sh)
        positions_train, positions_test = __split_data(positions, 'shuffle_split', cut_point, sh)
        y_cls_train, y_cls_test = __split_data(y_cls, 'shuffle_split', cut_point, sh)

    elif shuffling == 'no_shuffle':
        cut_point = int(data.shape[1] * train_ratio)
        X_train, X_test = __split_data(data, 'no_shuffle', cut_point)
        subject_ids_train, subject_ids_test = __split_data(subject_ids, 'no_shuffle', cut_point)
        positions_train, positions_test = __split_data(positions, 'no_shuffle', cut_point)
        y_cls_train, y_cls_test = __split_data(y_cls, 'no_shuffle', cut_point)

    return X_train, X_test, subject_ids_train, subject_ids_test, positions_train, positions_test, y_cls_train, y_cls_test


def split_data(data, subject_ids, positions, y_cls, shuffling, split_type, train_ratio=0.7, stratified=True):
    assert shuffling in ['split_shuffle', 'shuffle_split', 'no_shuffle'], 'shuffling must be either split_shuffle, shuffle_split or no_shuffle'
    assert split_type in ['time', 'subject'], 'split_type must be either time or subject'
    if split_type == 'time':
        return _split_time(data, subject_ids, positions, y_cls, shuffling, train_ratio)
    elif split_type == 'subject':
        return _split_sub(data, subject_ids, positions, y_cls, stratified, train_ratio)
