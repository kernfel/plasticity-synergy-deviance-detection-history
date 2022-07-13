import time
import sys
import os
import importlib
import multiprocessing as mp
import deepdish as dd
from brian2.only import *
import brian2genn

# for the IDE:
import numpy_ as np
import spatial, model, inputs, readout

from util import brian_cleanup


def run_cpu(templ, STD, TA, mod_params, *net_args, **device_args):
    device.reinit()
    device.activate()

    if templ < cfg.N_templates_with_dynamics:
        Net = model.create_network(
            *net_args, params=mod_params,
            reset_dt=inputs.get_episode_duration(mod_params),
            state_dt=cfg.params['dt'],
            state_vars=['v'] + [k for k,v in (('th_adapt', TA), ('u', STD)) if v],
            extras=('u',) if STD else ())
    else:
        Net = model.create_network(
            *net_args, params=mod_params,
            reset_dt=inputs.get_episode_duration(mod_params))
    
    rundata = readout.repeat_run(Net, mod_params, templates[templ])
    rundata['params'] = mod_params
    Net.run(rundata['runtime'])
    readout.get_results(Net, mod_params, rundata, tmax=cfg.ISIs[0]*ms)
    return rundata


def run_genn(templ, STD, TA, mod_params, *net_args, **device_args):
    device.reinit()
    device.activate(**device_args)
    
    if templ < cfg.N_templates_with_dynamics:
        Net = model.create_network(
            *net_args, params=mod_params,
            reset_dt=inputs.get_episode_duration(mod_params),
            state_dt=cfg.params['dt'], state_vars=['v'] + (['th_adapt'] if TA else []))
    else:
        Net = model.create_network(
            *net_args, params=mod_params,
            reset_dt=inputs.get_episode_duration(mod_params))
    
    rundata = readout.repeat_run(Net, mod_params, templates[templ])
    rundata['params'] = mod_params
    Net.run(rundata['runtime'])
    readout.get_results(Net, mod_params, rundata, compress=True, tmax=cfg.ISIs[0]*ms)

    if STD and templ < cfg.N_templates_with_dynamics:
        surrogate = {k: {'t': Net[f'SpikeMon_{k}'].t[:], 'i': Net[f'SpikeMon_{k}'].i[:]} for k in ('Exc', 'Inh')}

        device.reinit()
        device.activate(**device_args)
        
        mod_params_U = {**mod_params, 'tau_rec': 0*ms}
        Net = model.create_network(
            *net_args, params=mod_params_U,
            reset_dt=inputs.get_episode_duration(mod_params_U),
            state_dt=cfg.params['dt'], state_vars=['v'],
            surrogate=surrogate, suffix='_surrogate')
        
        rundata_U = readout.repeat_run(Net, mod_params_U, templates[templ])
        Net.run(rundata_U['runtime'])
        readout.get_results(Net, mod_params_U, rundata_U, compress=True, tmax=cfg.ISIs[0]*ms)
        for V_pair, U_pair in zip(rundata['dynamics'], rundata_U['dynamics']):
            for S in V_pair.keys():
                for tp in V_pair[S].keys():
                    V_pair[S][tp]['u'] = U_pair[S][tp]['v']
        rundata['dynamic_variables'].append('u')
    return rundata


if __name__ == '__main__':
    cfg = importlib.import_module('.'.join(sys.argv[1].split('.')[0].split('/')))

    if 'gpuid' in dir(cfg):
        working_dir = f'tmp/GPU{cfg.gpuid}'
        device_args = dict(directory=working_dir)
        set_device('genn', **device_args)
        prefs.devices.genn.cuda_backend.device_select='MANUAL'
        prefs.devices.genn.cuda_backend.manual_device=cfg.gpuid
        runit = run_genn
    elif 'runtime' not in dir(cfg) or not cfg.runtime:
        working_dir = 'tmp/CPP'
        device_args = {}
        set_device('cpp_standalone')
        prefs.devices.cpp_standalone.openmp_threads = mp.cpu_count() - 2
        runit = run_cpu
    else:
        working_dir = 'tmp/Runtime'
        device_args = {}
        set_device('runtime')
        runit = run_cpu
    os.makedirs(working_dir, exist_ok=True)

    rng = np.random.default_rng()

    Xstim, Ystim = spatial.create_stimulus_locations(cfg.params)

    # Set up input templates
    X, Y, W, D = spatial.create_weights(cfg.params, rng)
    Net = model.create_network(X, Y, Xstim, Ystim, W, D, cfg.params, reset_dt=inputs.get_episode_duration(cfg.params))
    templates = [readout.setup_run(Net, cfg.params, rng, cfg.stimuli, cfg.pairings) for _ in range(cfg.N_templates)]

    for templ, template in enumerate(templates):
        if templ < cfg.start_at.get('templ', 0):
            continue
        else:
            cfg.start_at.pop('templ', 0)
        for net in range(cfg.N_networks):
            if net < cfg.start_at.get('net', 0):
                continue
            else:
                cfg.start_at.pop('net', 0)
            if templ == 0 and cfg.start_at.pop('newnet', True):
                X, Y, W, D = spatial.create_weights(cfg.params, rng)
                try:
                    dd.io.save(cfg.netfile.format(net=net), dict(X=X, Y=Y, W=W, D=D))
                except Exception as e:
                    print(e)
            else:
                res = dd.io.load(cfg.netfile.format(net=net))
                X, Y, W, D = res['X']*meter, res['Y']*meter, res['W'], res['D']
            for STD, tau_rec_ in enumerate((0*ms, cfg.params['tau_rec'])):
                for TA, th_ampl_ in enumerate((0*mV, cfg.params['th_ampl'])):
                    if STD < cfg.start_at.get('STD', 0) or TA < cfg.start_at.get('TA', 0):
                        continue
                    else:
                        cfg.start_at.pop('STD', 0)
                        cfg.start_at.pop('TA', 0)
                    Tstart = time.time()
                    for iISI, isi in enumerate(cfg.ISIs):
                        if isi < cfg.start_at.get('isi', cfg.ISIs[0]):
                            continue
                        else:
                            cfg.start_at.pop('isi', 0)
                        mod_params = {**cfg.params, 'ISI': isi*ms, 'tau_rec': tau_rec_, 'th_ampl': th_ampl_}
                        
                        rundata = runit(templ, STD, TA, mod_params, X, Y, Xstim, Ystim, W, D, **device_args)
                        
                        for r in rundata['dynamics']:
                            for rs in r.values():
                                for rt in rs.values():
                                    if not STD:
                                        rt['u'] = rt['v']
                                    if not TA:
                                        rt['th_adapt'] = zeros_like(rt['v'])*volt
                        
                        try:
                            dd.io.save(cfg.fname.format(**locals()), rundata)
                        except Exception as e:
                            print(e)
                        brian_cleanup(working_dir)

                    print(f'Completed GPU ISI sweep (templ {templ}, net {net}, STD {STD}, TA {TA}) after {(time.time()-Tstart)/60:.1f} minutes.')