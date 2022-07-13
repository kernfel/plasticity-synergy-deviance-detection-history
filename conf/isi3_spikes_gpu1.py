from .params import params

gpuid = 1
start_at = dict(net=0, isi=100, STD=0, TA=0, templ=3)

N_networks = 30
N_templates = 5
ISIs = (100, 500, 1000, 2000)
fbase = '/data/felix/culture/isi3_'
fname = fbase + 'net{net}_isi{isi}_STD{STD}_TA{TA}_templ{templ}.h5'
netfile = fbase + 'net{net}.h5'

stimuli = {key: j for j, key in enumerate('ABCDE')}
pairings=(('A','B'), ('C','E'))
