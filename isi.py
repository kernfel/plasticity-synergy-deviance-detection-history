import time
import multiprocessing as mp
import matplotlib.pyplot as plt
import deepdish as dd
from brian2.only import *

# for the IDE:
import numpy_ as np
import spatial, model, inputs, readout

from util import concatenate
np.concatenate = concatenate
from spike_utils import iterspikes


rng = np.random.default_rng()
set_device('cpp_standalone')
prefs.devices.cpp_standalone.openmp_threads = mp.cpu_count() - 2


N = 1000
inhibitory_ratio = .2
params = {
    # Simulation
    'dt': 1*ms,
    
    # Population size
    'N': N,
    'N_exc': int((1-inhibitory_ratio) * N),
    'N_inh': int(inhibitory_ratio * N),

    # Membrane
    'v_rest': -60*mV,
    'v_reset': -74*mV,
    'v_threshold': -54*mV,
    'voltage_init': 'v_rest',
    
    'tau_mem': 30*ms,
    'refractory_exc': 3*ms,
    'refractory_inh': 2*ms,

    # Threshold adaptation - Exc
    'th_tau': 1*second,
    'th_ampl': 1*mV,

    # Short-term plasticity - Exc
    'tau_rec': 150*msecond,
    'U': 0.4,

    # Synapse dynamics
    'E_exc': 0*mV,
    'tau_ampa': 2*msecond,
    'E_inh': -100*mV,
    'tau_gaba': 4*msecond,
    
    # # Stochasticity
    # 'tau_noise': 10*msecond,
    # 'vnoise_std': 0.5*mV,

    # Layout
    'r_dish': 4*mm,
    'weight_distribution': 'singular',
    
    # Connectivity: Inh
    'r_inh': 1*mm,
    'outdeg_inh': 50,
    'w_inh_mean': 1,

    # Connectivity: Exc
    'r_exc': 2*mm,
    'outdeg_exc': 50,
    'w_exc_mean': 1,

    # Stimulus
    'N_stimuli': 5,
    'stim_distribution_radius': 2.5*mm,
    'neurons_per_stim': 10,
    'input_strength': 100,

    # Paradigm
    'settling_period': 1*second,
    'sequence_length': 5,
    'sequence_count': 100,
    'fully_random_msc': True,
    'fully_random_oddball': True
}


N_networks = 10
ISIs = (100, 200, 300, 500, 1000, 2000)
fbase = '/data/felix/culture/isi0_'
fname = fbase + 'net{net}_isi{isi}_STD{std}_TA{ta}.h5'
figfile = fbase + 'indices.png'
idxfile = fbase + 'idx.h5'
netfile = fbase + 'net{net}.h5'


Xstim, Ystim = spatial.create_stimulus_locations(params)

ddi, ai = np.empty((2, len(ISIs), 2, 2, N_networks))
for net in range(N_networks):
    X, Y, W, D = spatial.create_weights(params, rng)
    try:
        dd.io.save(netfile.format(net=net), dict(X=X, Y=Y, W=W, D=D))
    except Exception as e:
        print(e)
    for std, tau_rec_ in enumerate((0*ms, params['tau_rec'])):
        for ta, th_ampl_ in enumerate((0*mV, params['th_ampl'])):
            Tstart = time.time()
            for i, isi in enumerate(ISIs):
                mod_params = {**params, 'ISI': isi*ms, 'tau_rec': tau_rec_, 'th_ampl': th_ampl_}
                device.reinit()
                device.activate()
                Net = model.create_network(
                    X, Y, Xstim, Ystim, W, D, mod_params,
                    reset_dt=inputs.get_episode_duration(mod_params),
                    state_dt=params['dt'], when='before_resets', state_vars=['v', 'u'], extras=True)
                all_results, T = readout.setup_run(
                    Net, mod_params, rng, {key: j for j, key in enumerate('ABCDE')}, pairings=(('A','B'), ('C','E')))
                Net.run(T)
                readout.get_results(Net, mod_params, W, all_results)
                readout.compress_results(all_results)
                try:
                    dd.io.save(fname.format(**locals()), dict(all_results=all_results, params=mod_params))
                except Exception as e:
                    print(e)

                nspikes = {key: 0 for key in ('std', 'dev', 'msc')}
                for results in all_results.values():
                    for key in nspikes.keys():
                        nspikes[key] += results[key]['nspikes'].sum()
                ddi[i, std, ta, net] = (nspikes['dev'] - nspikes['msc']) / (nspikes['dev'] + nspikes['msc'])
                ai[i, std, ta, net] = (nspikes['msc'] - nspikes['std']) / (nspikes['msc'] + nspikes['std'])
            print(f'Completed ISI sweep {(net+1)*(std+1)*(ta+1)}/{N_networks*4} after {(time.time()-Tstart)/60:.1f} minutes.')

    fig, axs = plt.subplots(2, figsize=(9,12))
    for ax, idx, label in zip(axs, (ddi, ai), ('Deviance detection', 'Adaptation')):
        ax.plot(ISIs, idx[:, 0, 0, :net+1].mean(1), label='No plasticity')
        ax.plot(ISIs, idx[:, 1, 0, :net+1].mean(1), label='Short-term depression')
        ax.plot(ISIs, idx[:, 0, 1, :net+1].mean(1), label='Threshold adaptation')
        ax.plot(ISIs, idx[:, 1, 1, :net+1].mean(1), label='Both STD and TA')
        ax.plot(ISIs, idx[:, 0, 1, :net+1].mean(1) + idx[:, 1, 0, :net+1].mean(1), '--', label='Summation')
        ax.set_ylabel(f'{label} index')
        ax.set_xlabel('ISI (ms)')
        ax.legend()
        ax.axhline(0, color='lightgrey', lw=1)
    plt.savefig(figfile.format(**locals()))

    dd.io.save(idxfile, dict(ddi=ddi, ai=ai))