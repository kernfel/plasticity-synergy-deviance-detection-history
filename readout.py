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
    
    if with_xr and 'StateMon_Exc' in Net:
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


def get_neuron_spike_counts(N, pulsed_i, sequence, target_item):
    numspikes = np.zeros(N, dtype=int)
    for pulse_idx, pulse_item in enumerate(sequence):
        if pulse_item == target_item:
            np.add.at(numspikes, pulsed_i[pulse_idx], 1)
    return numspikes


def get_pulse_spike_counts_TMP(t, ISI):
    pulse_number = t // ISI
    steps = np.flatnonzero(np.diff(pulse_number) > 0)
    internal_num_spikes = np.diff(steps)
    num_spikes = np.concatenate([[steps[0]+1], internal_num_spikes, [0]])
    num_spikes[-1] = len(t) - np.sum(num_spikes)
    return num_spikes


def get_pulse_spike_counts(Net, params, episode=0):
    I, T, _ = get_episode_spikes(Net, params, episode=episode)
    return get_pulse_spike_counts_TMP(T, params['ISI'])


def populate_spike_results(Net, params, results, episode=0):
    npulses = params['sequence_length']*params['sequence_count']
    results['nspikes'] = get_pulse_spike_counts(Net, params, episode)
    results['spike_i'], results['spike_t'], results['spike_xr'] = get_episode_spikes(Net, params, episode=episode, with_xr=True)
    results['pulsed_i'], results['pulsed_t'] = zip(*list(iterspikes(
        results['spike_i'], results['spike_t'], npulses, params['ISI'])))
    results['pulsed_nspikes'] = np.zeros((len(results['pulsed_i']), params['N']), int)
    for j, i in enumerate(results['pulsed_i']):
        np.add.at(results['pulsed_nspikes'][j], i, 1)
    if 'StateMon_Exc' in Net:
        results['pulsed_xr'] = list(map(lambda tp: tp[0], iterspikes(
            results['spike_xr'], results['spike_t'], npulses, params['ISI'])))
        
        tpulse = np.arange(npulses)*params['ISI'] + episode*npulses*params['ISI'] + (episode+1)*params['settling_period']
        t_in_pulse = np.arange(stop=params['ISI'], step=params['dt'])
        tpulse_all = ((t_in_pulse[None, :] + tpulse[:, None]) / params['dt'] + .5).astype(int)
        ones_inhibitory = np.ones((params['N_inh'],) + tpulse_all.shape)

        dynamic_variables = {}
        for varname, init in zip(Net['Exc'].dynamic_variables, Net['Exc'].dynamic_variable_initial):
            var_exc = getattr(Net['StateMon_Exc'], varname)[:, tpulse_all]
            try:
                var_inh = getattr(Net['StateMon_Inh'], varname)[:, tpulse_all]
            except AttributeError:
                if type(init) == str:
                    var_inh = ones_inhibitory * eval(init, globals(), params)
                else:
                    var_inh = ones_inhibitory * init
            dynamic_variables[varname] = np.concatenate([var_exc, var_inh], axis=0)
        dynamic_variables['xr'] = dynamic_variables.pop('synaptic_xr')
        results.update(**dynamic_variables, dynamic_variables=list(dynamic_variables.keys()))


def get_infused_histogram(params, results, target_stim, infusion, norm=False):
    '''
    Computes the pulse-triggered spike histogram of each neuron as a (N, t) ndarray,
    but instead of adding 1 for each spike, it adds `infusion(results, pulse, indices, ticks)`,
    where `indices` and `ticks` are the firing neurons and corresponding time points
    in the given `pulse`.
    Note that this operates on **raw results**, and pulse indices thus correspond to the
    index in the raw sequence.
    If `norm=False` (default), results are normalised by number of pulses.
    Otherwise, if `norm=True`, results are normalised by the number of spikes in each bin.
    '''
    pulse_ids = np.flatnonzero(results['Seq'] == target_stim)
    npulses = len(pulse_ids)

    hist = np.zeros((params['N'], int(params['ISI']/params['dt'] +.5)))
    hist_1 = np.zeros_like(hist)
    for pulse in pulse_ids:
        i, t = results['pulsed_i'][pulse], results['pulsed_t'][pulse]
        t = (t/params['dt'] +.5).astype(int)
        hist[i, t] += infusion(results, pulse, i, t)
        hist_1[i, t] += 1
    if norm:
        np.divide(hist, hist_1, where=hist_1>0, out=hist)
    else:
        hist /= npulses
    return hist


def quantify_presynaptic(W, params, typed_results):
    '''
    Finds:
    - The cumulative presynaptic input over STATIC weights (static_exc, static_inh: two (N,) vectors).
        In other words, the summed pulse-triggered spike histogram, weighted by static synaptic weight.
    - The factor by which static_exc is reduced due to short-term depression (depression_factor: one (N,) vector)
    Note: "input" here refers to the increments applied to the target's synaptic conductance.
    '''
    hist, xr = typed_results['spike_hist'], typed_results['xr_sum']
    exc = np.zeros(params['N'], bool)
    exc[:params['N_exc']] = True
    W0 = W.copy()
    W0[np.isnan(W)] = 0
    # Postsyn input =   total presyn output       * weights,    summed over presyn
    static_exc = np.sum(hist[exc].sum(1)[:, None] * W0[exc, :], axis=0)
    static_inh = np.sum(hist[~exc].sum(1)[:, None] * W0[~exc, :], axis=0)
    dynamic = np.sum(xr[exc, None] * W0[exc, :], axis=0)
    return static_exc, static_inh, dynamic/static_exc


def get_results(Net, params, W, all_results):
    populated_episodes = []
    for results_dict in all_results.values():
        for rkey in ('Std', 'Dev', 'MSC'):
            results = results_dict[rkey]
            if results['episode'] not in populated_episodes:
                populate_spike_results(Net, params, results, results['episode'])

            out = results_dict[rkey.lower()] = {}
            pulse_mask = results['Seq'] == results_dict['stimulus']
            
            out['nspikes'] = results['pulsed_nspikes'][pulse_mask].mean(0)
            out['spike_hist'] = get_infused_histogram(params, results, results_dict['stimulus'], lambda *args: 1)
            
            if 'StateMon_Exc' in Net:
                out['xr_sum'] = get_infused_histogram(params, results, results_dict['stimulus'], lambda r,p,i,t: r['xr'][i,p,t]).sum(1)
                out['inputs_exc'], out['inputs_inh'], out['depression_factor'] = quantify_presynaptic(W, params, out)
                out['pulse_onset_th_adapt'] = results['th_adapt'][:, pulse_mask, 0].T
                out['pulse_onset_xr'] = results['xr'][:, pulse_mask, 0].T
                for key in results['dynamic_variables']:
                    out[key] = results[key][:, pulse_mask]


def setup_run(Net, params, rng, stimuli):
    d = inputs.get_episode_duration(params)
    if Net.reset_dt != d:
        warn(f'Net reset_dt ({Net.reset_dt}) does not match episode duration ({d})')
    stim_names = list(stimuli.keys())
    MSC, T = inputs.create_MSC(Net, params, rng)
    out = {name: {'stimulus': stim, 'MSC': {'Seq': MSC, 'episode': 0}} for name, stim in stimuli.items()}
    episode = 1
    for i, S1 in enumerate(stim_names):
        for S2 in stim_names[i+1:]:
            oddball1, T = inputs.create_oddball(
                Net, params, stimuli[S1],
                stimuli[S2], offset=T)
            oddball2, T = inputs.create_oddball(
                Net, params, stimuli[S2],
                stimuli[S1], offset=T)
            out[S1]['Std'] = {'Seq': oddball1, 'episode': episode}
            out[S1]['Dev'] = {'Seq': oddball2, 'episode': episode+1}
            out[S2]['Std'] = {'Seq': oddball2, 'episode': episode+1}
            out[S2]['Dev'] = {'Seq': oddball1, 'episode': episode}
            episode += 2
    return out, T
