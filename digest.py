import sys
import importlib
from collections import defaultdict
import warnings


from brian2.only import *
import deepdish as dd

import numpy_ as np
from util import Tree


conds = ('std', 'msc', 'dev')
voltage_measures = ('Activity', 'Depression', 'Threshold')


def get_voltages(params, dynamics, overflow=None):
    if isinstance(dynamics['v']/volt, Quantity):
        unit, factor = volt, 1000
    else:
        unit, factor = 1, 1
    depression = dynamics['u'] - dynamics['v']
    threshold = dynamics['th_adapt']
    activity = dynamics['u'] - params['v_threshold']/unit
    if overflow == 'max':
        activity = np.maximum(activity, threshold+depression)
    elif overflow is not None:
        activity[activity >= threshold+depression] = overflow
    return {
        'Activity': activity*factor,
        'Depression': depression*factor,
        'Threshold': threshold*factor}


def get_voltage_histograms(params, rundata, overflow=None):
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', message='Mean of empty slice')
        
        hists, masked_hists = Tree(), Tree()
        for ipair, pair in enumerate(rundata['pairs']):
            for stim in (pair['S1'], pair['S2']):
                for cond in conds:
                    dynamics = rundata['dynamics'][ipair][stim][cond]
                    voltages = get_voltages(params, dynamics, overflow)
                    bmask = voltages['Activity'] >= 0
                    masked_hists['weight'] = np.mean(bmask, 1)
                    mask = np.where(bmask, 1, np.nan)
                    for measure, val in voltages.items():
                        hists[measure][ipair][stim][cond] = val.mean(1)
                        masked_hists[measure][ipair][stim][cond] = np.nanmean(val*mask, 1)
        return hists.asdict(), masked_hists.asdict()
