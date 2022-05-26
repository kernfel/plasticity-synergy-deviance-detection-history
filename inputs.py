import numpy_ as np
from brian2 import second


def set_input_sequence(Net, sequence, params, offset=0*second):
    t = np.arange(len(sequence)) * params['ISI'] + params['settling_period'] + offset
    if hasattr(Net, 'input_sequence'):
        sequence = np.concatenate([Net.input_sequence, sequence])
        t = np.concatenate([Net.input_sequence_t, t])
    Net.input_sequence = sequence
    Net.input_sequence_t = t
    Net['Input'].set_spikes(sequence, t)
    return t[-1] + params['ISI']


def create_oddball(Net, params, A, B):
    sequence = np.tile([A] * (params['sequence_length']-1) + [B], params['sequence_count'])
    return sequence, set_input_sequence(Net, sequence, params)


def create_MSC(Net, params, rng : np.random.Generator):
    base_sequence = np.arange(params['N_stimuli'])
    rng.shuffle(base_sequence)
    sequence = np.tile(base_sequence, params['sequence_count'])
    return sequence, set_input_sequence(Net, sequence, params)