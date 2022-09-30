from .isi5 import *

del gpuid

fbase_500 = '/data/felix/culture/isi5-500/isi5_'
fname = fname.replace(fbase, fbase_500)
netfile = netfile.replace(fbase, fbase_500)
digestfile = digestfile.replace(fbase, fbase_500)
fbase = fbase_500

raw_fbase = fname[:-3]

ISIs = (500,)
N_templates = 1

start_at = {'newnet': False}
