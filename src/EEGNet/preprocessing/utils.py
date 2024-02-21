import numpy as np
import scipy.io
from pathlib import Path


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
