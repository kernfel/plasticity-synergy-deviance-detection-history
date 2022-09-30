from .isi5 import *

fbase_mouse = 'data/isi5_'
fname = fname.replace(fbase, fbase_mouse)
netfile = netfile.replace(fbase, fbase_mouse)
digestfile = digestfile.replace(fbase, fbase_mouse)
fbase = fbase_mouse

raw_fbase = fname[:-3]
