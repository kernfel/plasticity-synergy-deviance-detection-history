from distutils.log import warn
from itertools import count
from brian2.only import *

import numpy_ as np
import inputs

from util import concatenate
np.concatenate = concatenate
from spike_utils import iterspikes


def get_raw_spikes(Net, params, episodes):
    offset = 0
    npulses = params['sequence_length']*params['sequence_count']
    episodes = sorted(episodes)
    T, I = [{episode: [] for episode in episodes} for _ in range(2)]
    raw = {episode: {} for episode in episodes}
    for k in ['Exc'+Net.suffix, 'Inh'+Net.suffix]:
        all_t, all_i = Net[f'SpikeMon_{k}'].t, Net[f'SpikeMon_{k}'].i + offset
        offset += Net[k].N
        for episode, (episodic_i, episodic_t) in enumerate(iterspikes(all_i, all_t, max(episodes)+1, inputs.get_episode_duration(params))):
            if episode in episodes:
                T[episode].append(episodic_t)
                I[episode].append(episodic_i)
    
    for episode in episodes:
        episodic_t = np.concatenate(T[episode]) - params['settling_period']
        idx = np.argsort(episodic_t)
        episodic_t = episodic_t[idx]
        episodic_i = np.concatenate(I[episode])[idx]
        raw[episode]['pulsed_i'], raw[episode]['pulsed_t'] = zip(*list(iterspikes(episodic_i, episodic_t, npulses, params['ISI'])))
        raw[episode]['pulsed_nspikes'] = np.zeros((len(raw[episode]['pulsed_i']), params['N']), int)
        for j, i in enumerate(raw[episode]['pulsed_i']):
            np.add.at(raw[episode]['pulsed_nspikes'][j], i, 1)
    return raw


def get_raw_dynamics(Net, params, episodes, tmax):
    raw = {}
    npulses = params['sequence_length']*params['sequence_count']
    if 'StateMon_Exc'+Net.suffix in Net:
        t0_episode = (np.arange(max(episodes)+1) * inputs.get_episode_duration(params))[episodes]
        t0_pulse = np.arange(npulses)*params['ISI'] + params['settling_period']
        t_in_pulse = np.arange(stop=tmax, step=params['dt'])
        T = ((t0_episode[:, None, None] + t0_pulse[None, :, None] + t_in_pulse[None, None, :]) / params['dt'] + .5).astype(int)
        ones_inhibitory = np.ones((params['N_inh'],) + T.shape)

        dynamic_variables = {}
        for varname, init in zip(Net['Exc'+Net.suffix].dynamic_variables, Net['Exc'+Net.suffix].dynamic_variable_initial):
            if not hasattr(Net['StateMon_Exc'+Net.suffix], varname):
                continue
            var_exc = getattr(Net['StateMon_Exc'+Net.suffix], varname)[:, T]
            try:
                var_inh = getattr(Net['StateMon_Inh'+Net.suffix], varname)[:, T]
            except AttributeError:
                if type(init) == str:
                    var_inh = ones_inhibitory * eval(init, globals(), params)
                else:
                    var_inh = ones_inhibitory * init
            dynamic_variables[varname] = np.concatenate([var_exc, var_inh], axis=0)
        if 'synaptic_xr' in dynamic_variables:
            dynamic_variables['xr'] = dynamic_variables.pop('synaptic_xr')
        raw.update(**dynamic_variables, dynamic_variables=list(dynamic_variables.keys()))
    return raw


def get_infused_histogram(params, spike_results, infusion, norm=False, default=0, **kwargs):
    '''
    Computes the pulse-triggered spike histogram of each neuron as a (N, t) ndarray,
    but instead of adding 1 for each spike, it adds `infusion(spike_results, pulse, indices, ticks, **kwargs)`,
    where `indices` and `ticks` are the firing neurons and corresponding time points
    in the given `pulse`.
    Note that this operates on **typed results**, and pulse indices thus enumerate the pulses
    of the given type (e.g. only A:deviant pulses)
    If `norm=False` (default), results are normalised by number of pulses.
    Otherwise, if `norm=True`, results are normalised by the number of spikes in each bin.
    '''
    hist = np.zeros((params['N'], int(params['ISI']/params['dt'] +.5)))
    hist_1 = np.zeros_like(hist)
    for pulse, (i, t) in enumerate(zip(spike_results['pulsed_i'], spike_results['pulsed_t'])):
        t = (t/params['dt'] +.5).astype(int)
        hist[i, t] += infusion(spike_results, pulse, i, t, **kwargs)
        hist_1[i, t] += 1
    if norm:
        np.divide(hist, hist_1, where=hist_1>0, out=hist)
    else:
        hist /= len(spike_results['pulsed_i'])
    hist[hist_1==0] = default
    return hist


def quantify_presynaptic(W, params, hist, xr):
    '''
    Finds:
    - The cumulative presynaptic input over STATIC weights (static_exc, static_inh: two (N,) vectors).
        In other words, the summed pulse-triggered spike histogram, weighted by static synaptic weight.
    - The factor by which static_exc is reduced due to short-term depression (depression_factor: one (N,) vector)
    Note: "input" here refers to the increments applied to the target's synaptic conductance.
    '''
    exc = np.zeros(params['N'], bool)
    exc[:params['N_exc']] = True
    W0 = W.copy()
    W0[np.isnan(W)] = 0
    # Postsyn input =   total presyn output       * weights,    summed over presyn
    static_exc = np.sum(hist[exc].sum(1)[:, None] * W0[exc, :], axis=0)
    static_inh = np.sum(hist[~exc].sum(1)[:, None] * W0[~exc, :], axis=0)
    dynamic = np.sum(xr[exc, None] * W0[exc, :], axis=0)
    return static_exc, static_inh, dynamic/static_exc


def get_spike_results(Net, params, rundata, compress=False, tmax=None):
    spike_output = []
    itmax = 0
    rundata['msc_spikes'] = {}

    episodes = set()
    for pair in rundata['pairs']:
        for S in (pair['S1'], pair['S2']):
            for key in ('std', 'dev', 'msc'):
                episodes.add(pair[key][S])

    raw_spikes = get_raw_spikes(Net, params, list(episodes))    
    for pair in rundata['pairs']:
        out = {}
        spike_output.append(out)
        for S in (pair['S1'], pair['S2']):
            out[S] = {}
            for key in ('std', 'dev', 'msc'):
                episode = pair[key][S]
                if key == 'msc' and episode not in rundata['msc_spikes']:
                    rundata['msc_spikes'][episode] = raw_spikes[episode]

                raw = raw_spikes[episode]
                pulse_mask = rundata['sequences'][episode] == rundata['stimuli'][S]
                results = out[S][key] = {}
                
                results['nspikes'] = raw['pulsed_nspikes'][pulse_mask]
                results['pulsed_i'] = [i for i, j in zip(raw['pulsed_i'], pulse_mask) if j]
                results['pulsed_t'] = [i for i, j in zip(raw['pulsed_t'], pulse_mask) if j]
                results['spike_hist'] = get_infused_histogram(params, results, lambda *args: 1)

                nz = np.nonzero(results['spike_hist'])[1]
                if len(nz):
                    itmax = max(itmax, np.max(nz) + 1)
                else:
                    itmax = max(itmax, results['spike_hist'].shape[1])
    
    if tmax is not None:
        itmax = int(tmax/params['dt'] + 0.5)
        compress = True
    elif compress:
        tmax = itmax*params['dt']
    else:
        tmax = params['ISI']
    
    if compress:
        for out in spike_output:
            for S in out.keys():
                for key in out[S].keys():
                    out[S][key]['spike_hist'] = out[S][key]['spike_hist'][:, :itmax]
    
    rundata['spikes'] = spike_output
    return spike_output


def get_dynamics_results(Net, params, rundata, compress=False, tmax=None):
    if 'StateMon_Exc'+Net.suffix not in Net:
        rundata['dynamics'] = []
        rundata['dynamic_variables'] = []
        return rundata['dynamics']

    if tmax is not None:
        compress = True
    elif compress:
        try:
            for out in rundata['spikes']:
                for S in out.keys():
                    for key in out[S].keys():
                        itmax = out[S][key]['spike_hist'].shape[1]
                        raise StopIteration
        except StopIteration:
            tmax = itmax*params['dt']
    else:
        tmax = params['ISI']

    episodes = set()
    for pair in rundata['pairs']:
        for S in (pair['S1'], pair['S2']):
            for key in ('std', 'dev', 'msc'):
                episodes.add(pair[key][S])
    episodes = sorted(list(episodes))
    episode_mapping = dict(zip(episodes, count()))

    raw_dynamics = get_raw_dynamics(Net, params, episodes, tmax)
    dynamic_variables = dynamic_variables_out = raw_dynamics.get('dynamic_variables', [])
    dynamics_output = []

    for pair in rundata['pairs']:
        out = {}
        dynamics_output.append(out)
        for S in (pair['S1'], pair['S2']):
            out[S] = {}
            for key in ('std', 'dev', 'msc'):
                episode = pair[key][S]
                pulse_mask = rundata['sequences'][episode] == rundata['stimuli'][S]
                results = out[S][key] = {}
                for key, okey in zip(dynamic_variables, dynamic_variables_out):
                    results[okey] = raw_dynamics[key][:, episode_mapping[episode], pulse_mask]
    rundata['dynamics'] = dynamics_output
    rundata['dynamic_variables'] = dynamic_variables_out
    return dynamics_output


def get_results(Net, params, rundata, compress=False, tmax=None):
    spike_output = get_spike_results(Net, params, rundata, compress=compress, tmax=tmax)
    dynamics_output = get_dynamics_results(Net, params, rundata, compress=compress, tmax=tmax)
    return spike_output, dynamics_output


def setup_run(Net, params, rng, stimuli, pairings=None):
    d = inputs.get_episode_duration(params)
    if Net.reset_dt != d:
        warn(f'Net reset_dt ({Net.reset_dt}) does not match episode duration ({d}).')
    stim_names = list(stimuli.keys())
    if pairings is None:
        pairings = [(stim_names[2*i], stim_names[2*i+1]) for i in range(len(stim_names)//2)]
    assert len(pairings)
    
    MSC, T = inputs.create_MSC(Net, params, rng)
    episode = 1
    sequences = [MSC]
    pairs = []
    for S1, S2 in pairings:
        oddball1, T = inputs.create_oddball(
            Net, params, stimuli[S1], stimuli[S2], rng, offset=T)
        oddball2, T = inputs.create_oddball(
            Net, params, stimuli[S2], stimuli[S1], rng, offset=T)
        sequences.extend([oddball1, oddball2])
        pairs.append({
            'S1': S1, 'S2': S2,
            'msc': {S1: 0, S2: 0},
            'std': {S1: episode, S2: episode+1},
            'dev': {S1: episode+1, S2: episode}})
        episode += 2
    return {'sequences': sequences, 'pairs': pairs, 'runtime': T, 'stimuli': stimuli}


def repeat_run(Net, params, template):
    d = inputs.get_episode_duration(params)
    if Net.reset_dt != d:
        warn(f'Net reset_dt ({Net.reset_dt}) does not match episode duration ({d}).')
    T = 0*second
    for seq in template['sequences']:
        T = inputs.set_input_sequence(Net, seq, params, T)
    return {**{k: template[k] for k in ('sequences', 'pairs', 'stimuli')}, 'runtime': T}


def compress_results(rundata):
    tmax = 0
    for rpair in rundata['results']:
        for rstim in rpair.values():
            for rtype in rstim.values():
                tmax = max(tmax, np.max(np.nonzero(rtype['spike_hist'])[1]) + 1)
                tceil = rtype['spike_hist'].shape[1]
    if tmax < tceil:
        for rpair in rundata['results']:
            for rstim in rpair.values():
                for rtype in rstim.values():
                    rtype['spike_hist'] = rtype['spike_hist'][:, :tmax]
                    for varname in rundata['dynamic_variables']:
                        rtype[varname] = rtype[varname][:, :, :tmax]
