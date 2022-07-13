import time
import sys
import importlib
import deepdish as dd
from brian2.only import *
import brian2genn

# for the IDE:
import numpy_ as np
import spatial, model, inputs, readout

from util import brian_cleanup

cfg = importlib.import_module('.'.join(sys.argv[1].split('.')[0].split('/')))

working_dir = f'GPU{cfg.gpuid}'
rng = np.random.default_rng()
device_args = dict(directory=working_dir)
set_device('genn', **device_args)
prefs.devices.genn.cuda_backend.device_select='MANUAL'
prefs.devices.genn.cuda_backend.manual_device=cfg.gpuid


Xstim, Ystim = spatial.create_stimulus_locations(cfg.params)

# Set up input templates
X, Y, W, D = spatial.create_weights(cfg.params, rng)
Net = model.create_network(X, Y, Xstim, Ystim, W, D, cfg.params, reset_dt=inputs.get_episode_duration(cfg.params))
templates = [readout.setup_run(Net, cfg.params, rng, cfg.stimuli, cfg.pairings) for _ in range(cfg.N_templates)]

for templ, template in enumerate(templates):
    if templ < cfg.start_at.get('templ', 0):
        continue
    else:
        cfg.start_at.pop('templ')
    for net in range(cfg.N_networks):
        if net < cfg.start_at.get('net', 0):
            continue
        else:
            cfg.start_at.pop('net')
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
                    cfg.start_at.pop('STD')
                    cfg.start_at.pop('TA')
                Tstart = time.time()
                for iISI, isi in enumerate(cfg.cfg.ISIs):
                    if isi < cfg.start_at.get('isi', cfg.ISIs[0]):
                        continue
                    else:
                        cfg.start_at.pop('isi')
                    device.reinit()
                    device.activate(**device_args)
                    
                    mod_params = {**cfg.params, 'ISI': isi*ms, 'tau_rec': tau_rec_, 'th_ampl': th_ampl_}
                    if templ == 0:
                        Net = model.create_network(
                            X, Y, Xstim, Ystim, W, D, mod_params,
                            reset_dt=inputs.get_episode_duration(mod_params),
                            state_dt=cfg.params['dt'], state_vars=['v'] + (['th_adapt'] if TA else []))
                    else:
                        Net = model.create_network(
                            X, Y, Xstim, Ystim, W, D, mod_params,
                            reset_dt=inputs.get_episode_duration(mod_params))
                    
                    rundata = readout.repeat_run(Net, mod_params, template)
                    rundata['cfg.params'] = mod_params
                    Net.run(rundata['runtime'])
                    readout.get_results(Net, mod_params, rundata, compress=True, tmax=cfg.ISIs[0]*ms)

                    if STD and templ==0:
                        surrogate = {k: {'t': Net[f'SpikeMon_{k}'].t[:], 'i': Net[f'SpikeMon_{k}'].i[:]} for k in ('Exc', 'Inh')}

                        device.reinit()
                        device.activate(**device_args)
                        
                        mod_params_U = {**mod_params, 'tau_rec': 0*ms}
                        Net = model.create_network(
                            X, Y, Xstim, Ystim, W, D, mod_params_U,
                            reset_dt=inputs.get_episode_duration(mod_params_U),
                            state_dt=cfg.params['dt'], state_vars=['v'],
                            surrogate=surrogate, suffix='_surrogate')
                        
                        rundata_U = readout.repeat_run(Net, mod_params_U, template)
                        Net.run(rundata_U['runtime'])
                        readout.get_results(Net, mod_params_U, rundata_U, compress=True, tmax=cfg.ISIs[0]*ms)
                        for V_pair, U_pair in zip(rundata['dynamics'], rundata_U['dynamics']):
                            for S in V_pair.keys():
                                for tp in V_pair[S].keys():
                                    V_pair[S][tp]['u'] = U_pair[S][tp]['v']
                        rundata['dynamic_variables'].append('u')
                    
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
