import numpy as np


def annotations_checker(desc):
    """ Check Annotations """
    # basic variables
    first_four_annots = ['New Segment/',
                         'Comment/no USB Connection to actiCAP',
                         'Switch',
                         'Switch']
    annot_points = {'first_four': 0,
                    'all_same_type': 0}
    # 1. check if the first four markers' names are:
    if (desc[:4] == first_four_annots).all():
        annot_points['first_four'] += 1
    # 2. check if all the annotations within 'Switch', are the same type
    annot_points['all_same_type'] += within_switch_checker(desc)
    return annot_points


def change_annot_names(desc):
    """ Change Annotations to Meaningful Names"""
    annot_name_changer = {'Stimulus/S  1': 'Switch',
                          'Stimulus/S200': 'Eyes Open',
                          'Stimulus/S210': 'Eyes Closed'}
    for idx, val in enumerate(desc):
        if val in annot_name_changer.keys():
            desc[idx] = annot_name_changer[val]
    return desc


def within_switch_checker(desc):
    # check if the annotations behave as expected
    idxs = np.where(desc == 'Switch')[0]
    for i in range(len(idxs)-2):
        unqs = np.unique(desc[idxs[i]:idxs[i+1]])
        if 'Switch' in unqs:
            unqs = np.delete(unqs, np.where(unqs == 'Switch')[0])
            if len(unqs) == 1:
                check = True
        else:
            if len(unqs) == 1:
                check = True
            else:
                check = False
    return check
