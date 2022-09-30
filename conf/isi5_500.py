from .isi5 import *

del gpuid

fbase_500 = 'data/isi5_'
fname = fname.replace(fbase, fbase_500)
netfile = netfile.replace(fbase, fbase_500)
digestfile = digestfile.replace(fbase, fbase_500)
fbase = fbase_500

raw_fbase = fname[:-3]

ISIs = (500,)

start_at = {'newnet': False}
