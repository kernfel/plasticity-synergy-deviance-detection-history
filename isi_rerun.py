import time
import sys
import os
import functools
import importlib
import multiprocessing as mp
import deepdish as dd
from brian2.only import *
import brian2genn

# for the IDE:
import numpy_ as np
import spatial, model, inputs, readout

from util import brian_cleanup
from digest import get_voltage_histograms

import isi as isipy

if __name__ == '__main__':
    cfg = importlib.import_module('.'.join(sys.argv[1].split('.')[0].split('/')))
    rng = np.random.default_rng()
        
    net = int(sys.argv[2])
    try:
        res = dd.io.load(cfg.netfile.format(net=net))
    except Exception as e:
        print("Error loading network:", e)
    X, Y, W, D = res['X']*meter, res['Y']*meter, res['W'], res['D']
    Xstim, Ystim = spatial.create_stimulus_locations(cfg.params)

    templ=0
    try:
        template = readout.load_results(cfg.fname.format(templ=templ, net=0, STD=cfg.STDs[0], TA=cfg.TAs[0], isi=cfg.ISIs[0]))
    except Exception as e:
        print("Error loading template:", e)
        Net = model.create_network(X, Y, Xstim, Ystim, W, D, cfg.params, reset_dt=inputs.get_episode_duration(cfg.params))
        template = readout.setup_run(Net, cfg.params, rng, cfg.stimuli, cfg.pairings)
        templ = 'R'

    iISI = int(sys.argv[3]) if len(sys.argv)>3 else 2
    isi = cfg.ISIs[iISI]

    runit, working_dir = isipy.set_run_func(cfg)
    raw_fbase = cfg.raw_fbase if hasattr(cfg, 'raw_fbase') else None

    STD, TA = 0, 1
    mod_params = {**cfg.params, 'ISI': isi*ms,
                                      'tau_rec': (0*ms, cfg.params['tau_rec'])[STD],
                                      'th_ampl': (0*mV, cfg.params['th_ampl'])[TA]}
    rundata = runit(template, True, STD, TA, mod_params, X, Y, Xstim, Ystim, W, D,
                    raw_fbase=None if raw_fbase is None else raw_fbase.format(**locals()))
    for r in rundata['dynamics']:
        for rs in r.values():
            for rt in rs.values():
                if not STD:
                    rt['u'] = rt['v']
                if not TA:
                    rt['th_adapt'] = zeros_like(rt['v'])*volt
    rundata['voltage_histograms'], rundata['masked_voltage_histograms'] = get_voltage_histograms(mod_params, rundata)

    try:
        readout.save_results(cfg.fname.format(**locals()), rundata)
    except Exception as e:
        print("Error saving:", e)
    brian_cleanup(working_dir)
