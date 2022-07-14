from .params import params

N_networks = 30
N_templates = 5
N_templates_with_dynamics = 1
STDs = (0,1)
TAs = (0,1)
ISIs = (100, 500, 1000, 2000)
fbase = '/data/felix/culture/isi3_'
fname = fbase + 'net{net}_isi{isi}_STD{STD}_TA{TA}_templ{templ}.h5'
netfile = fbase + 'net{net}.h5'
digestfile = fbase + '{kind}.h5'

stimuli = {key: j for j, key in enumerate('ABCDE')}
pairings=(('A','B'), ('C','E'))

contrib_cutoff = .8
