import numpy as np
from brian2.only import *
import spatial


def create_excitatory(Net, X, Y, params, rng : np.random.Generator):
    # Noisy dv/dt = ((v_rest-v) + (E_exc-v)*g_exc + (E_inh-v)*g_inh) / tau_mem + vnoise_std*sqrt(2/tau_noise)*xi : volt (unless refractory)
    excitatory_eqn = '''
        dv/dt = ((v_rest-v) + (E_exc-v)*g_exc + (E_inh-v)*g_inh) / tau_mem : volt (unless refractory)
        dg_exc/dt = -g_exc/tau_ampa : 1
        dg_inh/dt = -g_inh/tau_gaba : 1
        dth_adapt/dt = -th_adapt/th_tau : volt
        x : meter
        y : meter
    '''
    excitatory_threshold = 'v > v_threshold + th_adapt'
    excitatory_reset = '''
        v = v_reset
        th_adapt += th_ampl
    '''

    Exc = NeuronGroup(params['N_exc'], excitatory_eqn, threshold=excitatory_threshold, reset=excitatory_reset, refractory=params['refractory_exc'],
                    method='euler', namespace=params, name='Exc')
    Exc.x, Exc.y = X[:params['N_exc']], Y[:params['N_exc']]
    voltage_init = 'rand() * (v_threshold - v_reset) + v_reset'
    Exc.v = voltage_init
    Exc.th_adapt = 0
    Exc.g_exc, Exc.g_inh = 0, 0
    Exc.add_attribute('dynamic_variables')
    Exc.add_attribute('dynamic_variable_initial')
    Exc.dynamic_variables = ['v', 'th_adapt', 'g_exc', 'g_inh']
    Exc.dynamic_variable_initial = [voltage_init, '0 * volt', 0, 0]

    Net.add(Exc)
    return Exc


def create_inhibitory(Net, X, Y, params, rng : np.random.Generator):
    # Noisy dv/dt = ((v_rest-v) + (E_exc-v)*g_exc + (E_inh-v)*g_inh) / tau_mem + vnoise_std*sqrt(2/tau_noise)*xi : volt (unless refractory)
    inhibitory_eqn = '''
        dv/dt = ((v_rest-v) + (E_exc-v)*g_exc + (E_inh-v)*g_inh) / tau_mem : volt (unless refractory)
        dg_exc/dt = -g_exc/tau_ampa : 1
        dg_inh/dt = -g_inh/tau_gaba : 1
        x : meter
        y : meter
    '''
    inhibitory_threshold = 'v > v_threshold'
    inhibitory_reset = 'v = v_reset'

    Inh = NeuronGroup(params['N_inh'], inhibitory_eqn, threshold=inhibitory_threshold, reset=inhibitory_reset, refractory=params['refractory_inh'],
                    method='euler', namespace=params, name='Inh')
    Inh.x, Inh.y = X[params['N_exc']:], Y[params['N_exc']:]
    voltage_init = 'rand() * (v_threshold - v_reset) + v_reset'
    Inh.v = voltage_init
    Inh.g_exc, Inh.g_inh = 0, 0
    Inh.add_attribute('dynamic_variables')
    Inh.add_attribute('dynamic_variable_initial')
    Inh.dynamic_variables = ['v', 'g_exc', 'g_inh']
    Inh.dynamic_variable_initial = [voltage_init, 0, 0]

    Net.add(Inh)
    return Inh


def create_excitatory_synapses(Net, params, Exc, Inh, W, D):
    excitatory_synapse = '''
        dxr/dt = (1-xr)/tau_rec : 1 (event-driven)
        w : 1
    '''
    excitatory_on_pre = '''
        g_exc_post += U*xr*w
        xr -= U*xr
    '''
    iPre_ee, iPost_ee = np.nonzero(~np.isnan(W[:params['N_exc'], :params['N_exc']]))
    iPre_ei, iPost_ei = np.nonzero(~np.isnan(W[:params['N_exc'], params['N_exc']:]))

    Syn_EE = Synapses(Exc, Exc, excitatory_synapse, on_pre=excitatory_on_pre, method='exact', namespace=params, name='EE')
    Syn_EE.connect(i=iPre_ee, j=iPost_ee)
    Syn_EE.w = W[iPre_ee, iPost_ee].ravel()
    Syn_EE.xr = 1
    Syn_EE.add_attribute('dynamic_variables')
    Syn_EE.add_attribute('dynamic_variable_initial')
    Syn_EE.dynamic_variables = ['xr']
    Syn_EE.dynamic_variable_initial = [1]
    Syn_EE.add_attribute('num_synapses')
    Syn_EE.num_synapses = len(iPre_ee)

    Syn_EI = Synapses(Exc, Inh, excitatory_synapse, on_pre=excitatory_on_pre, method='exact', namespace=params, name='EI')
    Syn_EI.connect(i=iPre_ei, j=iPost_ei)
    Syn_EI.w = W[iPre_ei, iPost_ei + params['N_exc']].ravel()
    Syn_EI.xr = 1
    Syn_EI.add_attribute('dynamic_variables')
    Syn_EI.add_attribute('dynamic_variable_initial')
    Syn_EI.dynamic_variables = ['xr']
    Syn_EI.dynamic_variable_initial = [1]
    Syn_EI.add_attribute('num_synapses')
    Syn_EI.num_synapses = len(iPre_ei)

    Net.add(Syn_EE, Syn_EI)
    return Syn_EE, Syn_EI


def create_inhibitory_synapses(Net, params, Exc, Inh, W, D):
    inhibitory_synapse = 'w : 1'
    inhibitory_on_pre = '''
        g_inh_post += w
    '''
    iPre_ie, iPost_ie = np.nonzero(~np.isnan(W[params['N_exc']:, :params['N_exc']]))
    iPre_ii, iPost_ii = np.nonzero(~np.isnan(W[params['N_exc']:, params['N_exc']:]))

    Syn_IE = Synapses(Inh, Exc, inhibitory_synapse, on_pre=inhibitory_on_pre, method='exact', name='IE')
    Syn_IE.connect(i=iPre_ie, j=iPost_ie)
    Syn_IE.w = W[iPre_ie + params['N_exc'], iPost_ie].ravel()

    Syn_II = Synapses(Inh, Inh, inhibitory_synapse, on_pre=inhibitory_on_pre, method='exact', name='II')
    Syn_II.connect(i=iPre_ii, j=iPost_ii)
    Syn_II.w = W[iPre_ii + params['N_exc'], iPost_ii + params['N_exc']].ravel()

    Net.add(Syn_IE, Syn_II)
    return Syn_IE, Syn_II


def create_input(Net, X, Y, Xstim, Ystim, params, Exc, Inh):
    Input = SpikeGeneratorGroup(params['N_stimuli'], [], []*ms, name='Input')
    idx = spatial.get_stimulated(X, Y, Xstim, Ystim, params)
    
    Input_Exc = Synapses(Input, Exc, name='Input_Exc', method='exact',
                         on_pre=f'g_exc_post += {params["input_strength"]}')
    e = np.nonzero(idx < params['N_exc'])
    Input_Exc.connect(i=e[0], j=idx[e])
    
    Input_Inh = Synapses(Input, Inh, name='Input_Inh', method='exact',
                         on_pre=f'g_exc_post += {params["input_strength"]}')
    i = np.nonzero(idx >= params['N_exc'])
    Input_Inh.connect(i=i[0], j=idx[i] - params['N_exc'])

    Net.add(Input, Input_Exc, Input_Inh)
    return Input, Input_Exc, Input_Inh


def create_spikemonitors(Net, Exc, Inh):
    SpikeMon_Exc = SpikeMonitor(Exc, name='SpikeMon_Exc')
    SpikeMon_Inh = SpikeMonitor(Inh, name='SpikeMon_Inh')
    Net.add(SpikeMon_Exc, SpikeMon_Inh)
    return SpikeMon_Exc, SpikeMon_Inh


def create_statemonitors(Net, dt):
    monitors = []
    clock = Clock(dt)
    for obj in Net:
        if hasattr(obj, 'dynamic_variables'):
            monitor = StateMonitor(
                obj, obj.dynamic_variables, name=f'StateMon_{obj.name}', clock=clock,
                record=range(obj.num_synapses) if hasattr(obj, 'num_synapses') else True)
            monitors.append(monitor)
    Net.add(*monitors)
    return monitors


def create_network_reset(Net, dt):
    resets = []
    for obj in Net:
        if hasattr(obj, 'dynamic_variables') and hasattr(obj, 'dynamic_variable_initial'):
            reset = '\n'.join([f'{var} = {init}'
                              for var, init in zip(obj.dynamic_variables, obj.dynamic_variable_initial)
                              if init is not None])
            if len(reset):
                reg = obj.run_regularly(reset, dt=dt)
                resets.append(reg)
    Net.add(*resets)
    return resets


def create_network(X, Y, Xstim, Ystim, W, D, params, rng, reset_dt=None, state_dt=None):
    Net = Network()
    Exc = create_excitatory(Net, X, Y, params, rng)
    Inh = create_inhibitory(Net, X, Y, params, rng)
    Syn_EE, Syn_EI = create_excitatory_synapses(Net, params, Exc, Inh, W, D)
    Syn_IE, Syn_II = create_inhibitory_synapses(Net, params, Exc, Inh, W, D)
    Input, Input_Exc, Input_Inh = create_input(Net, X, Y, Xstim, Ystim, params, Exc, Inh)
    SpikeMon_Exc, SpikeMon_Inh = create_spikemonitors(Net, Exc, Inh)
    if state_dt is not None:
        state_monitors = create_statemonitors(Net, state_dt)
    if reset_dt is not None:
        resets = create_network_reset(Net, reset_dt)
    return Net