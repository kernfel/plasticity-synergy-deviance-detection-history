import time
import multiprocessing as mp
import matplotlib.pyplot as plt
import deepdish as dd
from brian2.only import *
import brian2genn

# for the IDE:
import numpy_ as np
import spatial, model, inputs, readout

from util import concatenate
np.concatenate = concatenate
from spike_utils import iterspikes


gpuid = 0
rng = np.random.default_rng()
device_args = dict(directory=f'GPU{gpuid}')
set_device('genn', **device_args)
prefs.devices.genn.cuda_backend.device_select='MANUAL'
prefs.devices.genn.cuda_backend.manual_device=gpuid


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
    'fully_random_oddball': True,
    'ISI': 100*ms
}


N_networks = 50
N_templates = 5
ISIs = (100, 200, 300, 500, 1000, 2000)
fbase = '/data/felix/culture/isi2_'
fname = fbase + 'net{net}_isi{isi}_STD{STD}_TA{TA}_templ{templ}.h5'
figfile = fbase + 'indices.png'
idxfile = fbase + 'idx.h5'
netfile = fbase + 'net{net}.h5'


Xstim, Ystim = spatial.create_stimulus_locations(params)
stimuli = {key: j for j, key in enumerate('ABCDE')}
pairings=(('A','B'), ('C','E'))

# Set up input template
X, Y, W, D = spatial.create_weights(params, rng)
Net = model.create_network(X, Y, Xstim, Ystim, W, D, params, reset_dt=inputs.get_episode_duration(params))
templates = [readout.setup_run(Net, params, rng, stimuli, pairings) for _ in range(N_templates)]

ddi, ai = np.empty((2, N_networks, 2, 2, len(ISIs), N_templates, len(pairings), 2))

for net in range(N_networks):
    X, Y, W, D = spatial.create_weights(params, rng)
    try:
        dd.io.save(netfile.format(net=net), dict(X=X, Y=Y, W=W, D=D))
    except Exception as e:
        print(e)
    for STD, tau_rec_ in enumerate((0*ms, params['tau_rec'])):
        for TA, th_ampl_ in enumerate((0*mV, params['th_ampl'])):
            Tstart = time.time()
            for iISI, isi in enumerate(ISIs):
                for templ, template in enumerate(templates):
                    device.reinit()
                    device.activate(**device_args)
                    
                    mod_params = {**params, 'ISI': isi*ms, 'tau_rec': tau_rec_, 'th_ampl': th_ampl_}
                    Net = model.create_network(
                        X, Y, Xstim, Ystim, W, D, mod_params,
                        reset_dt=inputs.get_episode_duration(mod_params),
                        state_dt=params['dt'], state_vars=['v', 'th_adapt'])
                    
                    rundata = readout.repeat_run(Net, mod_params, template)
                    rundata['params'] = mod_params
                    Net.run(rundata['runtime'])
                    readout.get_results(Net, mod_params, rundata)

                    if STD:
                        surrogate = {k: {'t': Net[f'SpikeMon_{k}'].t[:], 'i': Net[f'SpikeMon_{k}'].i[:]} for k in ('Exc', 'Inh')}

                        device.reinit()
                        device.activate(**device_args)
                        
                        mod_params_U = {**mod_params, 'tau_rec': 0*ms}
                        Net = model.create_network(
                            X, Y, Xstim, Ystim, W, D, mod_params_U,
                            reset_dt=inputs.get_episode_duration(mod_params_U),
                            state_dt=params['dt'], state_vars=['v'],
                            surrogate=surrogate, suffix='_surrogate')
                        
                        rundata_U = readout.repeat_run(Net, mod_params_U, template)
                        Net.run(rundata_U['runtime'])
                        readout.get_results(Net, mod_params_U, rundata_U)
                        for V_pair, U_pair in zip(rundata['results'], rundata_U['results']):
                            for S in V_pair.keys():
                                for tp in V_pair[S].keys():
                                    V_pair[S][tp]['u'] = U_pair[S][tp]['v']
                        rundata['dynamic_variables'].append('u')
                    
                    readout.compress_results(rundata)
                    try:
                        dd.io.save(fname.format(**locals()), rundata)
                    except Exception as e:
                        print(e)

                    for ipair, pairdata in enumerate(rundata['results']):
                        for istim, stimdata in enumerate(pairdata.values()):
                            std, dev, msc = [stimdata[key]['nspikes'].sum() for key in ('std', 'dev', 'msc')]
                            ddi[net, STD, TA, iISI, templ, ipair, istim] = (dev-msc)/(dev+msc)
                            ai[net, STD, TA, iISI, templ, ipair, istim] = (msc-std)/(msc+std)
                    try:
                        dd.io.save(idxfile, dict(ddi=ddi, ai=ai, net=net, STD=STD, TA=TA, iISI=iISI, templ=templ))
                    except Exception as e:
                        print(e)

            print(f'Completed GPU ISI sweep (net {net}, STD {STD}, TA {TA}) after {(time.time()-Tstart)/60:.1f} minutes.')
