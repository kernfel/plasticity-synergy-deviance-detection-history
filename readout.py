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
            var_exc = getattr(Net['StateMon_Exc'], varname)[:, tpulse_all]
            try:
                var_inh = getattr(Net['StateMon_Inh'], varname)[:, tpulse_all]
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


def get_results(Net, params, W, all_results):
    raw_results = {}
    for results_dict in all_results.values():
        for rkey in ('Std', 'Dev', 'MSC'):
            episode = results_dict[rkey]
            if episode['episode'] not in raw_results:
                raw_results[episode['episode']] = get_raw_results(Net, params, episode['episode'])
            raw = raw_results[episode['episode']]

            out = results_dict[rkey.lower()] = {}
            pulse_mask = episode['Seq'] == results_dict['stimulus']
            
            out['nspikes'] = raw['pulsed_nspikes'][pulse_mask].mean(0)
            out['pulsed_i'] = [j for i,j in enumerate(raw['pulsed_i']) if episode['Seq'][i] == results_dict['stimulus']]
            out['pulsed_t'] = [j for i,j in enumerate(raw['pulsed_t']) if episode['Seq'][i] == results_dict['stimulus']]
            out['spike_hist'] = get_infused_histogram(params, out, lambda *args: 1)
            
            if 'StateMon_Exc' in Net:
                for key in raw['dynamic_variables']:
                    out[key] = raw[key][:, pulse_mask]


def setup_run(Net, params, rng, stimuli, pairings=None):
    d = inputs.get_episode_duration(params)
    if Net.reset_dt != d:
        warn(f'Net reset_dt ({Net.reset_dt}) does not match episode duration ({d})')
    stim_names = list(stimuli.keys())
    MSC, T = inputs.create_MSC(Net, params, rng)
    out = {name: {'stimulus': stim, 'MSC': {'Seq': MSC, 'episode': 0}} for name, stim in stimuli.items()}
    episode = 1
    if pairings is None:
        for i, S1 in enumerate(stim_names):
            for S2 in stim_names[i+1:]:
                oddball1, T = inputs.create_oddball(
                    Net, params, stimuli[S1], stimuli[S2], rng, offset=T)
                oddball2, T = inputs.create_oddball(
                    Net, params, stimuli[S2], stimuli[S1], rng, offset=T)
                out[S1]['Std'] = {'Seq': oddball1, 'episode': episode}
                out[S1]['Dev'] = {'Seq': oddball2, 'episode': episode+1}
                out[S2]['Std'] = {'Seq': oddball2, 'episode': episode+1}
                out[S2]['Dev'] = {'Seq': oddball1, 'episode': episode}
                episode += 2
    else:
        for S1, S2 in pairings:
            oddball1, T = inputs.create_oddball(
                Net, params, stimuli[S1], stimuli[S2], rng, offset=T)
            oddball2, T = inputs.create_oddball(
                Net, params, stimuli[S2], stimuli[S1], rng, offset=T)
            out[S1]['Std'] = {'Seq': oddball1, 'episode': episode}
            out[S1]['Dev'] = {'Seq': oddball2, 'episode': episode+1}
            out[S2]['Std'] = {'Seq': oddball2, 'episode': episode+1}
            out[S2]['Dev'] = {'Seq': oddball1, 'episode': episode}
            episode += 2
        for name in stimuli.keys():
            if 'Std' not in out[name]:
                out.pop(name)
    return out, T


def compress_results(all_results, discard_source=True):
    tmax = 0
    for r in all_results.values():
        for key in ('std', 'dev', 'msc'):
            tmax = max(tmax, np.max(np.nonzero(r[key]['spike_hist'])[1]) + 1)
    for r in all_results.values():
        if 'dynamic_variables' in r['Std']:
            for varname in r['Std']['dynamic_variables']:
                for key in ('std', 'dev', 'msc'):
                    r[key][varname] = r[key][varname][:, :, :tmax]
    if discard_source:
        for r in all_results.values():
            for key in ('Std', 'Dev', 'MSC'):
                r.pop(key)
