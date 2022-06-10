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


def create_oddball(Net, params, A, B, offset=0*second):
    sequence = np.tile([A] * (params['sequence_length']-1) + [B], params['sequence_count'])
    return sequence, set_input_sequence(Net, sequence, params, offset=offset)


def create_MSC(Net, params, rng : np.random.Generator, offset=0*second):
    if params.get('fully_random_msc', False):
        sequence = rng.choice(params['N_stimuli'], params['sequence_length']*params['sequence_count'])
    else:
        base_sequence = np.arange(params['N_stimuli'])
        rng.shuffle(base_sequence)
        sequence = np.tile(base_sequence, params['sequence_count'])
    return sequence, set_input_sequence(Net, sequence, params, offset=offset)


def get_episode_duration(params):
    data_duration = params['ISI']*params['sequence_length']*params['sequence_count']
    return params['settling_period'] + data_duration
