from .params import params

N_networks = 30
N_templates = 1
N_templates_with_dynamics = 1
STDs = (0,1)
TAs = (0,1)
ISIs = (500,)
fbase = 'data/isi5-500/isi5_'
raw_fbase = fbase + 'net{net}_isi{isi}_STD{STD}_TA{TA}_templ{templ}'
fname = raw_fbase + '.h5'
netfile = fbase + 'net{net}.h5'
digestfile = fbase + '{kind}.h5'

stimuli = {key: j for j, key in enumerate('ABCDE')}
pairings=(('A','B'), ('C','E'))

contrib_cutoff = .8
minimum_active_fraction = .5
