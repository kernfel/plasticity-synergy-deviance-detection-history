from distutils.log import warn
from brian2.only import *

import numpy_ as np
import inputs

from util import concatenate
np.concatenate = concatenate
from spike_utils import iterspikes



def get_episode_spikes(Net, params, episode=0, sorted=True, with_xr=False):
    offset = 0
    T, I = [], []
    full_episode_duration = inputs.get_episode_duration(params)
    data_duration = full_episode_duration - params['settling_period']
    t0 = episode * full_episode_duration + params['settling_period']
    for k in ['Exc', 'Inh']:
        t, i = Net[f'SpikeMon_{k}'].t, Net[f'SpikeMon_{k}'].i + offset
        offset += Net[k].N
        mask = (t >= t0 - 0.5*params['dt']) & (t < t0 + data_duration - 0.5*params['dt'])
        T.append(t[mask])
        I.append(i[mask])
    T = np.concatenate(T) - t0
    I = np.concatenate(I)
    if sorted:
        sorted = np.argsort(T)
        T, I = T[sorted], I[sorted]
    
    if with_xr and 'StateMon_Exc' in Net and hasattr(Net['StateMon_Exc'], 'synaptic_xr'):
        i0 = int(t0/params['dt'] + 0.5)
        try:
            import brian2genn
            if isinstance(get_device(), brian2genn.device.GeNNDevice):
                i0 -= 1
        except ModuleNotFoundError:
            pass
        iend = i0 + int(data_duration/params['dt'] + 0.5)
        iT = (T/params['dt'] + 0.5).astype(int)
        xr_rec = Net['StateMon_Exc'].synaptic_xr[:, i0:iend]
        xr = np.ones(I.shape)
        Imask = I < params['N_exc']
        xr[Imask] = xr_rec[I[Imask], iT[Imask]]
    else:
        xr = None
    return I, T, xr


def get_raw_results(Net, params, episode=0):
    raw = {}
    npulses = params['sequence_length']*params['sequence_count']
    spike_i, spike_t, _ = get_episode_spikes(Net, params, episode=episode)
    raw['pulsed_i'], raw['pulsed_t'] = zip(*list(iterspikes(spike_i, spike_t, npulses, params['ISI'])))
    raw['pulsed_nspikes'] = np.zeros((len(raw['pulsed_i']), params['N']), int)
    for j, i in enumerate(raw['pulsed_i']):
        np.add.at(raw['pulsed_nspikes'][j], i, 1)
    if 'StateMon_Exc' in Net:
        tpulse = np.arange(npulses)*params['ISI'] + episode*npulses*params['ISI'] + (episode+1)*params['settling_period']
        t_in_pulse = np.arange(stop=params['ISI'], step=params['dt'])
        tpulse_all = ((t_in_pulse[None, :] + tpulse[:, None]) / params['dt'] + .5).astype(int)
        ones_inhibitory = np.ones((params['N_inh'],) + tpulse_all.shape)

        dynamic_variables = {}
        for varname, init in zip(Net['Exc'].dynamic_variables, Net['Exc'].dynamic_variable_initial):
            if not hasattr(Net['StateMon_Exc'], varname):
                continue
            tpulse_all_ = tpulse_all + (1 if varname.endswith('_delayed') else 0)
            var_exc = getattr(Net['StateMon_Exc'], varname)[:, tpulse_all_]
            try:
                var_inh = getattr(Net['StateMon_Inh'], varname)[:, tpulse_all_]
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


def get_infused_histogram(params, typed_results, infusion, norm=False, default=0, **kwargs):
    '''
    Computes the pulse-triggered spike histogram of each neuron as a (N, t) ndarray,
    but instead of adding 1 for each spike, it adds `infusion(typed_results, pulse, indices, ticks, **kwargs)`,
    where `indices` and `ticks` are the firing neurons and corresponding time points
    in the given `pulse`.
    Note that this operates on **typed results**, and pulse indices thus enumerate the pulses
    of the given type (e.g. only A:deviant pulses)
    If `norm=False` (default), results are normalised by number of pulses.
    Otherwise, if `norm=True`, results are normalised by the number of spikes in each bin.
    '''
    hist = np.zeros((params['N'], int(params['ISI']/params['dt'] +.5)))
    hist_1 = np.zeros_like(hist)
    for pulse, (i, t) in enumerate(zip(typed_results['pulsed_i'], typed_results['pulsed_t'])):
        t = (t/params['dt'] +.5).astype(int)
        hist[i, t] += infusion(typed_results, pulse, i, t, **kwargs)
        hist_1[i, t] += 1
    if norm:
        np.divide(hist, hist_1, where=hist_1>0, out=hist)
    else:
        hist /= len(typed_results['pulsed_i'])
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


def get_results(Net, params, rundata, replace_delayed=True):
    raw_results = {}
    outputs = []
    dynamic_variables_out = []
    for pair in rundata['pairs']:
        out = {}
        outputs.append(out)
        for S in (pair['S1'], pair['S2']):
            out[S] = {}
            for key in ('std', 'dev', 'msc'):
                episode = pair[key][S]
                if episode not in raw_results:
                    raw_results[episode] = get_raw_results(Net, params, episode)
                    dynamic_variables = raw_results[episode].get('dynamic_variables', [])
                    if len(dynamic_variables_out) != len(dynamic_variables):  # only once
                        if replace_delayed:
                            dynamic_variables_out = [v[:-len('_delayed')] if v.endswith('_delayed') else v
                                                     for v in dynamic_variables]
                        else:
                            dynamic_variables_out = dynamic_variables

                raw = raw_results[episode]
                pulse_mask = rundata['sequences'][episode] == rundata['stimuli'][S]
                results = out[S][key] = {}
                
                results['nspikes'] = raw['pulsed_nspikes'][pulse_mask].mean(0)
                results['pulsed_i'] = [i for i, j in zip(raw['pulsed_i'], pulse_mask) if j]
                results['pulsed_t'] = [i for i, j in zip(raw['pulsed_t'], pulse_mask) if j]
                results['spike_hist'] = get_infused_histogram(params, results, lambda *args: 1)
                
                for key, okey in zip(dynamic_variables, dynamic_variables_out):
                    results[okey] = raw[key][:, pulse_mask]
    rundata['results'] = outputs
    rundata['dynamic_variables'] = dynamic_variables_out
    return outputs


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
    T += params['dt']
    return {'sequences': sequences, 'pairs': pairs, 'runtime': T, 'stimuli': stimuli}


def repeat_run(Net, params, template):
    d = inputs.get_episode_duration(params)
    if Net.reset_dt != d:
        warn(f'Net reset_dt ({Net.reset_dt}) does not match episode duration ({d}).')
    T = 0*second
    for seq in template['sequences']:
        T = inputs.set_input_sequence(Net, seq, params, T)
    T += params['dt']
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
