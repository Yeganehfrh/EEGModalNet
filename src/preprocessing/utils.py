import numpy as np


def annotations_checker(desc):
    """ Check Annotations """
    # basic variables
    first_four_annots1 = ['New Segment/',
                          'Comment/no USB Connection to actiCAP',
                          'Stimulus/S  1',
                          'Stimulus/S  1']
    annot_points = {'first_four': 0,
                    'all_same_type': 0}
    # 1. check if the first four markers' names are:
    if (desc[:4] == first_four_annots1).all() or desc[1] == 'Comment/actiCAP Data On':
        annot_points['first_four'] += 1
    # 2. check if all the annotations within 'Switch', are the same type
    annot_points['all_same_type'] += within_switch_checker(desc)
    return annot_points


def within_switch_checker(desc):
    # check if the annotations behave as expected
    check = False  # default
    idxs = np.where(desc == 'Stimulus/S  1')[0]
    for i in range(len(idxs)-2):
        unqs = np.unique(desc[idxs[i]:idxs[i+1]])
        if 'Stimulus/S  1' in unqs:
            unqs = np.delete(unqs, np.where(unqs == 'Stimulus/S  1')[0])
            if len(unqs) == 1:
                check = True
        else:
            if len(unqs) == 1:
                check = True
    return check


def change_annot_names(desc, enumerated=False):
    """ Change Annotations to Meaningful Names"""
    annot_name_changer = {'Stimulus/S  1': 'Switch',
                          'Stimulus/S200': 'EO',
                          'Stimulus/S210': 'EC'}
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

    for i, v in enumerate(onsets.values()):
        raws[pattern[i]].append(raw.copy().crop(tmin=v, tmax=v+duration))  # crop 60s after switch

    # concatenate all EC segments
    raws['EC'][0].append(raws['EC'][1:])
    raws['EC'] = raws['EC'][0]

    # concatenate all EO segments
    raws['EO'][0].append(raws['EO'][1:])
    raws['EO'] = raws['EO'][0]

    return raws


def find_pattern(raw):
    desc = change_annot_names(raw.annotations.description.copy(),
                              enumerated=True)
    onsets = raw.annotations.onset.copy()
    annot_dict = {k: v for k, v in zip(desc[3:], onsets[3:])}  # drop first 3 annotations

    switch_onsets = {}
    pattern = []
    for i, k in enumerate(annot_dict.keys()):
        if 'Switch' in k:
            switch_onsets[k] = annot_dict[k]
            # procession of EC/EO based on the next item appearing after switch
            pattern.append(list(annot_dict.keys())[i+1].split('_')[0])
    return switch_onsets, pattern
