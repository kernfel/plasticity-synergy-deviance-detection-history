import numpy as np
from brian2.only import *
import spatial


def create_excitatory(Net, X, Y, params, clock, extras, enforced_spikes, suffix):
    # Noisy dv/dt = ((v_rest-v) + (E_exc-v)*g_exc + (E_exc-v)*g_input + (E_inh-v)*g_inh) / tau_mem + vnoise_std*sqrt(2/tau_noise)*xi : volt (unless refractory)
    vtmp = '(E_exc-{v})*g_exc + (E_exc-{v})*g_input + (E_inh-{v})*g_inh'
    dvdt = '((v_rest-{v}) + {tmp}) / tau_mem * int(not_refractory) + (v_reset-{v})/dt*(1-int(not_refractory))'
    dvsyndt = '((-{v}) + {tmp}) / tau_mem * int(not_refractory) + (-{v})/dt*(1-int(not_refractory))'
    eqn = f'''
        vtmp = {vtmp.format(v='v')} : volt
        dv/dt = {dvdt.format(v='v', tmp='vtmp')} : volt
        dg_exc/dt = -g_exc/tau_ampa : 1
        dg_inh/dt = -g_inh/tau_gaba : 1
        dg_input/dt = -g_input/tau_ampa : 1
        x : meter
        y : meter
    '''
    threshold = 'v > v_threshold'
    resets = {}
    if params['th_ampl'] != 0*mV and params['th_tau'] > 0*ms:
        extras = extras + ('th_adapt',)
    if 'th_adapt' in extras:
        eqn += '''
        dth_adapt/dt = -th_adapt/th_tau + int(t-1.5*dt < lastspike)*th_ampl/dt : volt
        '''
        threshold = 'v > v_threshold + th_adapt'
    if 'xr' in extras:
        eqn += '''
        dsynaptic_xr/dt = (1-synaptic_xr)/tau_rec - int(t-1.5*dt < lastspike)*U*synaptic_xr/dt : 1
        '''
    if 'u' in extras:
        eqn += f'''
        dg_exc_nox/dt = -g_exc_nox/tau_ampa : 1
        utmp = {vtmp.format(v='u').replace('g_exc', 'g_exc_nox')} : volt
        du/dt = {dvdt.format(v='u', tmp='utmp')} : volt
        '''
    if 'vsyn' in extras:
        eqn += f'''
        dvsyn/dt = {dvsyndt.format(v='vsyn', tmp='vtmp')} : volt
        '''
    if enforced_spikes:
        eqn += '''
        spike_enforcer : 1
        '''
        threshold = 'spike_enforcer > 0'
        resets['spike_enforcer'] = '= 0'
        

    reset = '\n'.join([f'{key} {value}' for key, value in resets.items()])
    Exc = NeuronGroup(params['N_exc'], eqn, threshold=threshold, reset=reset, refractory=params['refractory_exc'],
                    method='euler', namespace=params, name='Exc'+suffix, clock=clock)
    Exc.x, Exc.y = X[:params['N_exc']], Y[:params['N_exc']]
    Exc.add_attribute('dynamic_variables')
    Exc.add_attribute('dynamic_variable_initial')
    Exc.dynamic_variables = ['v', 'g_exc', 'g_inh', 'g_input']
    Exc.dynamic_variable_initial = [params['voltage_init'], 0, 0, 0]
    if 'th_adapt' in extras:
        Exc.dynamic_variables.append('th_adapt')
        Exc.dynamic_variable_initial.append('0 * volt')
    if 'xr' in extras:
        Exc.dynamic_variables.append('synaptic_xr')
        Exc.dynamic_variable_initial.append(1)
    if 'u' in extras:
        Exc.dynamic_variables.extend(('g_exc_nox', 'u'))
        Exc.dynamic_variable_initial.extend((0, params['voltage_init']))
    if 'vsyn' in extras:
        Exc.dynamic_variables.append('vsyn')
        Exc.dynamic_variable_initial.append('0*volt')
    if enforced_spikes:
        Exc.dynamic_variables.append('spike_enforcer')
        Exc.dynamic_variable_initial.append(0)
    
    for var, value in zip(Exc.dynamic_variables, Exc.dynamic_variable_initial):
        setattr(Exc, var, value)

    Net.add(Exc)
    return Exc


def create_inhibitory(Net, X, Y, params, clock, extras, enforced_spikes, suffix):
    # Noisy dv/dt = ((v_rest-v) + (E_exc-v)*g_exc + (E_exc-v)*g_input + (E_inh-v)*g_inh) / tau_mem + vnoise_std*sqrt(2/tau_noise)*xi : volt (unless refractory)
    dvdt = '((v_rest-V) + (E_exc-V)*g_exc + (E_exc-V)*g_input + (E_inh-V)*g_inh) / tau_mem'
    eqn = f'''
        dv/dt = {dvdt.replace('V','v')}*int(not_refractory) + (v_reset-v)/dt*(1-int(not_refractory)) : volt
        dg_exc/dt = -g_exc/tau_ampa : 1
        dg_inh/dt = -g_inh/tau_gaba : 1
        dg_input/dt = -g_input/tau_ampa : 1
        x : meter
        y : meter
    '''
    threshold = 'v > v_threshold'
    resets = {}
    if 'u' in extras:
        eqn += f'''
        dg_exc_nox/dt = -g_exc_nox/tau_ampa : 1
        du/dt = {dvdt.replace('V','u')}*int(not_refractory) + (v_reset-u)/dt*(1-int(not_refractory)) : volt
        '''
    if enforced_spikes:
        eqn += '''
        spike_enforcer : 1
        '''
        threshold = 'spike_enforcer > 0'
        resets['spike_enforcer'] = '= 0'
    

    reset = '\n'.join([f'{key} {value}' for key, value in resets.items()])
    Inh = NeuronGroup(params['N_inh'], eqn, threshold=threshold, reset=reset, refractory=params['refractory_inh'],
                    method='euler', namespace=params, name='Inh'+suffix, clock=clock)
    Inh.x, Inh.y = X[params['N_exc']:], Y[params['N_exc']:]
    Inh.add_attribute('dynamic_variables')
    Inh.add_attribute('dynamic_variable_initial')
    Inh.dynamic_variables = ['v', 'g_exc', 'g_inh', 'g_input']
    Inh.dynamic_variable_initial = [params['voltage_init'], 0, 0, 0]
    if 'u' in extras:
        Inh.dynamic_variables.extend(('g_exc_nox', 'u'))
        Inh.dynamic_variable_initial.extend((0, params['voltage_init']))
    if enforced_spikes:
        Inh.dynamic_variables.append('spike_enforcer')
        Inh.dynamic_variable_initial.append(0)
    
    for var, value in zip(Inh.dynamic_variables, Inh.dynamic_variable_initial):
        setattr(Inh, var, value)

    Net.add(Inh)
    return Inh


def create_surrogate(Net, Group, spikes, clock, suffix):
    Surrogate = SpikeGeneratorGroup(Group.N, spikes['i'], spikes['t'] - clock.dt, clock=clock, sorted=True, name=f'Surrogate_{Group.name}'+suffix)
    Enforcer = Synapses(Surrogate, Group, on_pre='spike_enforcer_post += 1', method='exact', name=f'Enforcer_{Group.name}'+suffix)
    Enforcer.connect(i='j')
    Net.add(Surrogate, Enforcer)
    return Surrogate, Enforcer


def make_exc_synapse(pre, post, iPre, iPost, w, params, with_u=False, **kwargs):
    plastic = params['tau_rec'] > 0*ms
    eqn = '''w : 1'''
    if plastic:
        eqn += '''
    dxr/dt = (1-xr)/tau_rec : 1 (event-driven)
        '''
        onpre = '''
    g_exc_post += U*xr*w
    xr -= U*xr
        '''
        if with_u:
            onpre += '''
    g_exc_nox_post += U*w
            '''
    elif not plastic:
        onpre = '''
    g_exc_post += U*w
        '''
    
    syn = Synapses(pre, post, eqn, on_pre=onpre, method='exact', namespace=params, **kwargs)
    syn.connect(i=iPre, j=iPost)
    syn.w = w
    syn.add_attribute('dynamic_variables')
    syn.add_attribute('dynamic_variable_initial')
    if plastic:
        syn.xr = 1
        syn.dynamic_variables = ['xr']
        syn.dynamic_variable_initial = [1]
    else:
        syn.dynamic_variables = []
        syn.dynamic_variable_initial = []

    return syn


def create_excitatory_synapses(Net, params, clock, presyn, Exc, Inh, W, D, extras, static_delay, suffix):
    iPre_ee, iPost_ee = np.nonzero(~np.isnan(W[:params['N_exc'], :params['N_exc']]))
    w = W[iPre_ee, iPost_ee].ravel()
    Syn_EE = make_exc_synapse(presyn, Exc, iPre_ee, iPost_ee, w, params, with_u='u' in extras,
                              name='EE'+suffix, clock=clock, delay=static_delay)
    
    iPre_ei, iPost_ei = np.nonzero(~np.isnan(W[:params['N_exc'], params['N_exc']:]))
    w = W[iPre_ei, iPost_ei + params['N_exc']].ravel()
    Syn_EI = make_exc_synapse(presyn, Inh, iPre_ei, iPost_ei, w, params, with_u='u' in extras,
                              name='EI'+suffix, clock=clock, delay=static_delay)

    Net.add(Syn_EE, Syn_EI)
    return Syn_EE, Syn_EI


def create_inhibitory_synapses(Net, params, clock, presyn, Exc, Inh, W, D, extras, static_delay, suffix):
    inhibitory_synapse = 'w : 1'
    inhibitory_on_pre = '''
        g_inh_post += w
    '''
    iPre_ie, iPost_ie = np.nonzero(~np.isnan(W[params['N_exc']:, :params['N_exc']]))
    iPre_ii, iPost_ii = np.nonzero(~np.isnan(W[params['N_exc']:, params['N_exc']:]))

    Syn_IE = Synapses(presyn, Exc, inhibitory_synapse, on_pre=inhibitory_on_pre, method='exact',
                      name='IE'+suffix, clock=clock, delay=static_delay)
    Syn_IE.connect(i=iPre_ie, j=iPost_ie)
    Syn_IE.w = W[iPre_ie + params['N_exc'], iPost_ie].ravel()

    Syn_II = Synapses(presyn, Inh, inhibitory_synapse, on_pre=inhibitory_on_pre, method='exact',
                      name='II'+suffix, clock=clock, delay=static_delay)
    Syn_II.connect(i=iPre_ii, j=iPost_ii)
    Syn_II.w = W[iPre_ii + params['N_exc'], iPost_ii + params['N_exc']].ravel()

    Net.add(Syn_IE, Syn_II)
    return Syn_IE, Syn_II


def create_input(Net, X, Y, Xstim, Ystim, params, clock, Exc, Inh, suffix):
    Input = SpikeGeneratorGroup(params['N_stimuli'], [], []*ms, name='Input'+suffix, clock=clock)
    idx = spatial.get_stimulated(X, Y, Xstim, Ystim, params)
    
    Input_Exc = Synapses(Input, Exc, name='Input_Exc'+suffix, method='exact',
                         on_pre=f'g_input_post += {params["input_strength"]}', clock=clock)
    e = np.nonzero(idx < params['N_exc'])
    Input_Exc.connect(i=e[0], j=idx[e])
    
    Input_Inh = Synapses(Input, Inh, name='Input_Inh'+suffix, method='exact',
                         on_pre=f'g_input_post += {params["input_strength"]}', clock=clock)
    i = np.nonzero(idx >= params['N_exc'])
    Input_Inh.connect(i=i[0], j=idx[i] - params['N_exc'])

    Net.add(Input, Input_Exc, Input_Inh)
    return Input, Input_Exc, Input_Inh


def create_spikemonitors(Net, Exc, Inh, suffix):
    SpikeMon_Exc = SpikeMonitor(Exc, name='SpikeMon_Exc'+suffix)
    SpikeMon_Inh = SpikeMonitor(Inh, name='SpikeMon_Inh'+suffix)
    Net.add(SpikeMon_Exc, SpikeMon_Inh)
    return SpikeMon_Exc, SpikeMon_Inh


def create_statemonitors(Net, dt, variables, when, suffix):
    monitors = []
    clock = Clock(dt)
    for obj in Net:
        if hasattr(obj, 'dynamic_variables'):
            varnames = [var for var in obj.dynamic_variables if variables is None or var in variables]
            if len(varnames):
                monitor = StateMonitor(
                    obj, varnames, name=f'StateMon_{obj.name}', clock=clock,
                    record=range(obj.num_synapses) if hasattr(obj, 'num_synapses') else True,
                    when=when)
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


def create_network(X, Y, Xstim, Ystim, W, D, params, reset_dt=None,
                   state_dt=None, state_vars=None, when='end',
                   extras=(),
                   surrogate={}, suffix=''):
    Net = Network()
    defaultclock.dt = params['dt']
    clock = defaultclock
    Exc = create_excitatory(Net, X, Y, params, clock, extras, bool(surrogate), suffix)
    Inh = create_inhibitory(Net, X, Y, params, clock, extras, bool(surrogate), suffix)
    if surrogate:
        assert params['settling_period'] >= params['dt'], 'Surrogacy requires a settling period of at least 1 dt.'
        presyn_Exc, enforcer_Exc = create_surrogate(Net, Exc, surrogate['Exc'], clock, suffix)
        presyn_Inh, enforcer_Inh = create_surrogate(Net, Inh, surrogate['Inh'], clock, suffix)
        static_delay = clock.dt
    else:
        presyn_Exc = Exc
        presyn_Inh = Inh
        static_delay = None
    Syn_EE, Syn_EI = create_excitatory_synapses(Net, params, clock, presyn_Exc, Exc, Inh, W, D, extras, static_delay, suffix)
    Syn_IE, Syn_II = create_inhibitory_synapses(Net, params, clock, presyn_Inh, Exc, Inh, W, D, extras, static_delay, suffix)
    Input, Input_Exc, Input_Inh = create_input(Net, X, Y, Xstim, Ystim, params, clock, Exc, Inh, suffix)
    SpikeMon_Exc, SpikeMon_Inh = create_spikemonitors(Net, Exc, Inh, suffix)
    if state_dt is not None:
        if state_vars is None:
            state_vars = Exc.dynamic_variables + Inh.dynamic_variables
        if len(state_vars):
            state_monitors = create_statemonitors(Net, state_dt, state_vars, when, suffix)
    if reset_dt is not None:
        resets = create_network_reset(Net, reset_dt)
    Net.reset_dt = reset_dt
    Net.suffix = suffix
    return Net
