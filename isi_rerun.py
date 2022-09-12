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

    try:
        template = dd.io.load(cfg.fname.format(templ=0, net=0, STD=cfg.STDs[0], TA=cfg.TAs[0], isi=cfg.ISIs[0]))
        for key in preset.keys():
            if key not in templates[templ]:
                preset.pop(key)
    except Exception as e:
        print("Error loading template:", e)
    
    net = int(sys.argv[2])
    try:
        res = dd.io.load(cfg.netfile.format(net=net))
    except Exception as e:
        print("Error loading network:", e)
    X, Y, W, D = res['X']*meter, res['Y']*meter, res['W'], res['D']

    iISI = int(sys.argv[3]) if len(sys.argv)>3 else 2
    isi = cfg.ISIs[iISI]

    runit, working_dir = isipy.set_run_func(cfg)

    Xstim, Ystim = spatial.create_stimulus_locations(cfg.params)

    mod_params = {**cfg.params, 'ISI': isi*ms}
    rundata = runit(template, True, True, True, mod_params, X, Y, Xstim, Ystim, W, D)
    rundata['voltage_histograms'], rundata['masked_voltage_histograms'] = get_voltage_histograms(mod_params, rundata)

    try:
        dd.io.save(cfg.fname.format(**locals()), rundata)
    except Exception as e:
        print("Error saving:", e)
    brian_cleanup(working_dir)
