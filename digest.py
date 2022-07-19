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
                    masked_hists['weight'][ipair][stim][cond] = np.mean(bmask, 1)
                    mask = np.where(bmask, 1, np.nan)
                    for measure, val in voltages.items():
                        hists[measure][ipair][stim][cond] = val.mean(1)
                        masked_hists[measure][ipair][stim][cond] = np.nanmean(val*mask, 1)
        return hists.asdict(), masked_hists.asdict()


def iter_runs(cfg, dynamics_only=False):
    N_templates = min(cfg.N_templates, cfg.N_templates_with_dynamics) if dynamics_only else cfg.N_templates
    for templ in range(N_templates):
        for net in range(cfg.N_networks):
            for STD in cfg.STDs:
                for TA in cfg.TAs:
                    for iISI, isi in enumerate(cfg.ISIs):
                        yield templ, net, STD, TA, iISI, isi


def digest(cfg):
    spike_runs_shape = (cfg.N_templates, cfg.N_networks, len(cfg.STDs), len(cfg.TAs), len(cfg.ISIs), len(cfg.pairings), 2)
    dynamic_runs_shape = (min(cfg.N_templates, cfg.N_templates_with_dynamics),) + spike_runs_shape[1:] + (len(conds),)
    nspikes = {}
    histograms, masked_histograms = None, None
    for templ, net, STD, TA, iISI, isi in iter_runs(cfg):
        try:
            res = dd.io.load(cfg.fname.format(**locals()))
        except Exception as e:
            print(e)
        for ipair, pair in enumerate(res['pairs']):
            for istim, stim in enumerate((pair['S1'], pair['S2'])):
                for icond, cond in enumerate(conds):
                    idx = templ, net, STD, TA, iISI, ipair, istim, icond
                    spikes = res['spikes'][ipair][stim][cond]
                    if cond not in nspikes:
                        nspikes[cond] = np.empty(spike_runs_shape + spikes['nspikes'].shape)
                    nspikes[cond][idx[:-1]] = spikes['nspikes']

                    if 'voltage_histograms' in res:
                        for measure, hists in res['voltage_histograms'].items():
                            hist = hists[ipair][stim][cond]
                            if histograms is None:
                                histograms = {
                                    k: np.empty(dynamic_runs_shape + hist.shape)
                                    for k in list(res['voltage_histograms'].keys()) + ['p(spike)']}
                            histograms[measure][idx] = hist
                        histograms['p(spike)'][idx] = spikes['spike_hist']

                        for measure, hists in res['masked_voltage_histograms'].items():
                            hist = hists[ipair][stim][cond]
                            if masked_histograms is None:
                                masked_histograms = {
                                    k: np.empty(dynamic_runs_shape + hist.shape)
                                    for k in res['masked_voltage_histograms'].keys()}
                            masked_histograms[measure][idx] = hist
                        
    scrub_stimulated_overactivation(cfg, histograms)
    scrub_stimulated_overactivation(cfg, masked_histograms)

    try:
        dd.io.save(cfg.digestfile.format(kind='nspikes'), nspikes)
        dd.io.save(cfg.digestfile.format(kind='histograms'), histograms)
        dd.io.save(cfg.digestfile.format(kind='masked_histograms'), masked_histograms)
    except Exception as e:
        print(e)


def scrub_stimulated_overactivation(cfg, histograms):
    for net in range(cfg.N_networks):
        try:
            stimulated = dd.io.load(cfg.netfile.format(net=net))['stimulated_neurons']
        except Exception as e:
            print(e)
        for ipair, pair in enumerate(cfg.pairings):
            for istim, stim in enumerate(pair):
                stimulated_neurons = stimulated[cfg.stimuli[stim]]
                unstimulated_neurons = np.flatnonzero(~np.isin(np.arange(cfg.params['N']), stimulated_neurons))
                idx = slice(None), net, slice(None), slice(None), slice(None), ipair, istim, slice(None)
                idx_to_filter = *idx, stimulated_neurons
                idx_reference = *idx, unstimulated_neurons
                reference = np.nanmax(histograms['Activity'][idx_reference])
                histograms['Activity'][idx_to_filter] = np.minimum(histograms['Activity'][idx_to_filter], reference)


if __name__ == '__main__':
    cfg = importlib.import_module('.'.join(sys.argv[1].split('.')[0].split('/')))
    digest(cfg)