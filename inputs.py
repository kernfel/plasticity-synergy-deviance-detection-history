import numpy as np


def set_input_sequence(Net, sequence, params):
    t = np.arange(len(sequence)) * params['ISI'] + params['settling_period']
    Net['Input'].set_spikes(sequence, t)
    return len(sequence) * params['ISI'] + params['settling_period']


def create_oddball(Net, params, A, B):
    sequence = np.tile([A] * (params['sequence_length']-1) + [B], params['sequence_count'])
    return sequence, set_input_sequence(Net, sequence, params)


def create_MSC(Net, params, rng : np.random.Generator):
    base_sequence = np.arange(params['N_stimuli'])
    rng.shuffle(base_sequence)
    sequence = np.tile(base_sequence, params['sequence_count'])
    return sequence, set_input_sequence(Net, sequence, params)