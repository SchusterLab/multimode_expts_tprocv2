"""
calibration_helpers.py
All do_* and update_* functions for the calibration notebooks.
Import with: from calibration_helpers import *
"""

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from slab import AttrDict

# ── these will be set by calling init_helpers(station) ──────────────────────
station    = None
meas       = None
cfg_dict   = None
config_thisrun = None

def init_helpers(_station):
    """Call once at the top of each notebook after station is created."""
    global station, meas, cfg_dict, config_thisrun
    station        = _station
    meas           = _station.meas
    cfg_dict       = _station.cfg_dict
    config_thisrun = _station.config_thisrun


# ── experiment toggle dict ───────────────────────────────────────────────────
def default_expts_to_run():
    return {
        # readout
        'res_spec':           True,
        'single_shot':        True,
        # qubit ge
        'pulse_probe_ge':     True,
        't2_ge':              True,
        'amplitude_ge':       True,
        't1_ge':              True,
        # qubit ef
        'pulse_probe_ef':     True,
        't2_ef':              True,
        'amplitude_ef':       True,
        't1_ef':              True,
        # manipulate
        'man_modes':          [1],
        'pulse_probe_f0g1':   True,
        'length_rabi_sweep':  True,
        'length_rabi':        False,
        'chi_ge':             True,
        'chi_ef':             True,
        'RB':                 False,
        # storage
        'stor_modes':         [1, 2, 3, 4, 5, 6, 7],
        'stor_spectroscopy':  True,
        'sideband_freq_sweep':True,
        'sideband_length_rabi':True,
    }


# ════════════════════════════════════════════════════════════════════════════
# READOUT
# ════════════════════════════════════════════════════════════════════════════

def do_tof(
    config_thisrun,
    reps=1, rounds=100, check_e=False, final_delay=500,
    analyze_and_display=True, use_config_params_for_readout=True,
    gain=None, length=None, frequency=None,
):
    tof = meas.ToFExperiment(cfg_dict=cfg_dict, prefix='ToFExperiment', go=False)
    if use_config_params_for_readout:
        gain      = config_thisrun.device.readout.gain
        length    = config_thisrun.device.readout.length
        frequency = config_thisrun.device.readout.frequency
        print(f"Using readout params from config: gain={gain}, length={length}, frequency={frequency}")
    tof.cfg = AttrDict(deepcopy(config_thisrun))
    tof.cfg.expt = dict(reps=reps, rounds=rounds, check_e=check_e,
                        final_delay=final_delay, gain=gain, length=length,
                        frequency=frequency, sweep_other_param={})
    tof.acquire(progress=True)
    if analyze_and_display:
        tof.display(adc_trig_offset=config_thisrun.device.readout.trig_offset,
                    title_str='Time of Flight Calibration')
    return tof


def update_tof(tof, config_thisrun, new_trig_offset):
    config_thisrun.device.readout.trig_offset = new_trig_offset
    print(f"Updated trig_offset to {new_trig_offset:.3f} µs")
    tof.display(data=tof.data, adc_trig_offset=new_trig_offset,
                title_str='Time of Flight — updated trig_offset')


def do_res_spec(
    config_thisrun,
    start=None, step=None, expts=900, reps=500, rounds=1,
    pulse_e=False, prepulse={}, gain=0.11, length=3.0,
    frequency=7604.3, final_delay=250, span=10,
    analyze_and_display=True, use_config_params_for_readout=True,
):
    rspec = meas.ResonatorSpectroscopyExperiment(cfg_dict=cfg_dict,
                                                  prefix='ResonatorSpectroscopyExperiment')
    if use_config_params_for_readout:
        gain      = config_thisrun.device.readout.gain
        length    = config_thisrun.device.readout.length
        frequency = config_thisrun.device.readout.frequency
        print(f"Using readout params from config: gain={gain}, length={length}, frequency={frequency}")
    rspec.cfg = AttrDict(deepcopy(config_thisrun))
    if start is None:
        start = frequency - span / 2
    if step is None:
        step = span / expts
    rspec.cfg.expt = dict(start=start, step=step, expts=expts, reps=reps,
                          rounds=rounds, pulse_e=pulse_e, prepulse=prepulse,
                          gain=gain, length=length, frequency=frequency,
                          final_delay=final_delay)
    rspec.go(analyze=False, display=False, progress=True, save=True)
    if analyze_and_display:
        spec_analysis = meas.SpectroscopyFitting(title='Readout Spectroscopy', data=rspec.data)
        spec_analysis.analyze()
        spec_analysis.display()
    return rspec


def update_res_spec(rspec, config_thisrun):
    config_thisrun.device.readout.frequency = rspec.data['fit_amps'][2]
    print(f"Updated readout frequency to {config_thisrun.device.readout.frequency:.3f} MHz")


def do_single_shot(
    config_thisrun,
    readout_gain=None, readout_length=None, readout_frequency=None,
    shots=5000, pulse_e=True, prepulse={}, sweep_params={},
    final_delay=250, analyze_and_display=True,
):
    hstgrm = meas.SingleShotExperiment(cfg_dict=cfg_dict, prefix='SingleShot')
    hstgrm.cfg = AttrDict(deepcopy(config_thisrun))
    if readout_gain      is None: readout_gain      = station.yaml_cfg.device.readout.gain
    if readout_length    is None: readout_length    = station.yaml_cfg.device.readout.length
    if readout_frequency is None: readout_frequency = station.yaml_cfg.device.readout.frequency
    print(f"Using readout: gain={readout_gain}, length={readout_length}, frequency={readout_frequency}")
    hstgrm.cfg.expt = dict(shots=shots, reps=1, rounds=1, pulse_e=pulse_e,
                           prepulse=prepulse, sweep_param=sweep_params,
                           final_delay=final_delay, readout_gain=readout_gain,
                           readout_length=readout_length, readout_frequency=readout_frequency)
    hstgrm.go(analyze=False, display=False, progress=True, save=True)
    if analyze_and_display:
        hst_analysis = meas.Histogram(hstgrm.data, station=station)
        _ = hst_analysis.analyze()
        hst_analysis.display()
        return hst_analysis
    return hstgrm


def update_single_shot(hst_analysis, config_thisrun):
    fid      = float(hst_analysis.results['fidelity'])
    threshold = float(hst_analysis.results['threshold'])
    angle    = float(hst_analysis.results['rotation_angle'])
    config_thisrun.device.readout.phase     += angle
    config_thisrun.device.readout.threshold  = threshold
    print(f"Updated readout: fidelity={fid:.3f}, threshold={threshold:.3f}, angle={angle:.3f} deg")


# ════════════════════════════════════════════════════════════════════════════
# QUBIT
# ════════════════════════════════════════════════════════════════════════════

def do_pulse_probe_spec(
    config_thisrun, center=None, start=None, step=None, span=None,
    expts=300, reps=100, rounds=1, pulse_e=False,
    prepulse={}, postpulse={}, sweep_other_param=None, probe_pulse_param={},
    analyze_and_display=True,
):
    ppspec = meas.PulseProbeSpectroscopyExperiment(cfg_dict=cfg_dict,
                                                    prefix='PulseProbeSpectroscopyExperiment')
    ppspec.cfg = AttrDict(deepcopy(config_thisrun))
    if start is None: start = center - span / 2
    if step  is None: step  = span / expts
    ppspec.cfg.expt = dict(start=start, step=step, expts=expts, reps=reps,
                           rounds=rounds, pulse_e=pulse_e, prepulse=prepulse,
                           sweep_other_param=sweep_other_param,
                           probe_pulse_param=probe_pulse_param, postpulse=postpulse)
    ppspec.go(analyze=False, display=False, progress=True, save=True)
    if analyze_and_display:
        spec_analysis = meas.SpectroscopyFitting(data=ppspec.data)
        spec_analysis.analyze()
        spec_analysis.display()
    return ppspec


def update_pulse_probe_ge(qspec, config_thisrun):
    config_thisrun.device.qubit.f_ge = qspec.data['fit_avgi'][2]
    print(f"Updated f_ge to {config_thisrun.device.qubit.f_ge:.3f} MHz")


def update_pulse_probe_ef(qspec, config_thisrun):
    config_thisrun.device.qubit.f_ef = qspec.data['fit_avgi'][2]
    print(f"Updated f_ef to {config_thisrun.device.qubit.f_ef:.3f} MHz")


def do_t2_ramsey(
    config_thisrun, cfg_dict=None,
    prepulse={}, postpulse={}, start=0.01, step_size=0.02,
    expts=201, ramsey_freq=3, reps=200, rounds=1,
    sigma=0.05, sigma_inc=0, freq=None, gain=100,
    pulse_type='gauss', analyze_and_display=True,
    sweep_other_param={},
):
    t2ramsey = meas.RamseyExperiment(cfg_dict=cfg_dict, prefix='RamseyExperiment', go=False)
    t2ramsey.cfg = AttrDict(deepcopy(config_thisrun))
    t2ramsey.cfg.expt = dict(
        start=start, step=step_size, expts=expts, ramsey_freq=ramsey_freq,
        reps=reps, rounds=rounds, checkEF=False,
        prepulse=prepulse, postpulse=postpulse,
        sigma=sigma, sigma_inc=sigma_inc, freq=freq, gain=gain,
        phase=0, type=pulse_type, wait_time=0.0, num_pi=1,
        sweep_other_param=sweep_other_param,
    )
    t2ramsey.go(analyze=False, display=False, progress=True, save=True)
    if not analyze_and_display:
        return t2ramsey
    analysis = meas.RamseyFitting(t2ramsey.data, config=t2ramsey.cfg)
    analysis.display()
    return analysis


def update_t2_ramsey_ge(t2ramsey, config_thisrun):
    config_thisrun.device.qubit.f_ge += min(t2ramsey.data['f_adjust_ramsey_avgi'])
    print(f"Updated f_ge to {config_thisrun.device.qubit.f_ge:.3f} MHz")


def update_t2_ramsey_ef(t2ramsey, config_thisrun):
    config_thisrun.device.qubit.f_ef += min(t2ramsey.data['f_adjust_ramsey_avgi'])
    print(f"Updated f_ef to {config_thisrun.device.qubit.f_ef:.3f} MHz")


def do_rabi(
    config_thisrun, cfg_dict=None,
    start=0.01, expts=151, reps=200, rounds=1,
    pulse_type='gauss', sweep='amp', chan=0,
    freq=3500, gain=0.1, sigma=0.1, sigma_inc=4,
    length=1.0, ramp_sigma=0.02, ramp_sigma_inc=0.0,
    n_pulses=1, prepulse={}, postpulse={},
    max_gain=1.0, max_length=10.0,
    sweep_other_param={},
):
    step = (max_gain - start) / expts if sweep == 'amp' else (max_length - start) / expts
    rabi = meas.RabiExperiment(cfg_dict=cfg_dict, prefix='RabiExperiment', go=False)
    rabi.cfg = AttrDict(deepcopy(config_thisrun))
    rabi.cfg.expt = dict(
        start=start, max_gain=max_gain, max_length=max_length,
        expts=expts, reps=reps, rounds=rounds, sigma_test=None,
        sweep=sweep, chan=chan, freq=freq, type=pulse_type,
        gain=gain, sigma=sigma, step=step,
        sigma_inc=sigma_inc, length=length,
        ramp_sigma=ramp_sigma, ramp_sigma_inc=ramp_sigma_inc,
        prepulse=prepulse, postpulse=postpulse, n_pulses=n_pulses,
        final_delay=config_thisrun.device.readout.final_delay,
        sweep_other_param=sweep_other_param,
    )
    rabi.go(analyze=False, display=False, progress=True, save=True)
    return meas.AmplitudeRabiFitting(rabi.data, readout_per_round=4,
                                     config=rabi.cfg, station=station, sweep = sweep)


def update_amplitude_rabi_ge(amprabi, config_thisrun):
    config_thisrun.device.qubit.pulses.pi_ge.gain  = float(np.round(amprabi.data['pi_gain_avgi'],  3))
    config_thisrun.device.qubit.pulses.hpi_ge.gain = float(np.round(amprabi.data['hpi_gain_avgi'], 3))
    config_thisrun.device.qubit.pulses.pi_ge.sigma  = amprabi.cfg.expt.sigma
    config_thisrun.device.qubit.pulses.hpi_ge.sigma = amprabi.cfg.expt.sigma
    station.handle_config_update(True)
    print("Updated ge pi/hpi gain and sigma.")


def update_amplitude_rabi_ef(amprabi, config_thisrun):
    config_thisrun.device.qubit.pulses.pi_ef.gain  = float(np.round(amprabi.data['pi_gain_avgi'],  3))
    config_thisrun.device.qubit.pulses.hpi_ef.gain = float(np.round(amprabi.data['hpi_gain_avgi'], 3))
    config_thisrun.device.qubit.pulses.pi_ef.sigma  = amprabi.cfg.expt.sigma
    config_thisrun.device.qubit.pulses.hpi_ef.sigma = amprabi.cfg.expt.sigma
    station.handle_config_update(True)
    print("Updated ef pi/hpi gain and sigma.")


def do_t1(
    config_thisrun, cfg_dict=None,
    prepulse={}, start=0.01, step_size=0.5,
    expts=201, reps=200, rounds=1,
    sigma=None, sigma_inc=None, freq=None, gain=None,
    pulse_type=None, analyze_and_display=True,
    qubit_pulse = True,
):
    t1 = meas.T1Experiment(cfg_dict=cfg_dict, prefix='T1Experiment', go=False)
    t1.cfg = AttrDict(deepcopy(config_thisrun))
    if freq       is None: freq       = config_thisrun.device.qubit.f_ge
    if gain       is None: gain       = config_thisrun.device.qubit.pulses.pi_ge.gain
    if sigma      is None: sigma      = config_thisrun.device.qubit.pulses.pi_ge.sigma
    if pulse_type is None: pulse_type = 'gauss'
    if sigma_inc  is None: sigma_inc  = 4
    t1.cfg.expt = dict(start=start, step=step_size, expts=expts, reps=reps,
                       rounds=rounds, prepulse=prepulse,
                       sigma=sigma, sigma_inc=sigma_inc, freq=freq, gain=gain,
                       phase=0, type=pulse_type, wait_time=0.0, sweep_other_param={}, qubit_pulse=qubit_pulse)
    t1.go(analyze=False, display=False, progress=True, save=True)
    if not analyze_and_display:
        return t1
    analysis = meas.T1Fitting(t1.data, config=t1.cfg)
    analysis.display()
    return analysis


# ════════════════════════════════════════════════════════════════════════════
# MANIPULATE
# ════════════════════════════════════════════════════════════════════════════

def do_length_rabi_f0g1_sweep(config_thisrun, expt_path, config_path,
                               freq_start, freq_stop, freq_step, exp_param_file=None):
    from multimode_expts.sequential_experiment_classes import man_f0g1_class
    name = 'length_rabi_f0g1_sweep'
    cls  = man_f0g1_class(soccfg=None, path=expt_path, prefix=name,
                          config_file=config_path, exp_param_file=exp_param_file,
                          config_thisrun=config_thisrun)
    cls.yaml_cfg = AttrDict(deepcopy(config_thisrun))
    cls.loaded[name] = dict(freq_start=freq_start, freq_stop=freq_stop,
                            freq_step=freq_step, start=2, step=0.1,
                            qubits=[0], expts=25, reps=100, rounds=1,
                            gain=8000, ramp_sigma=0.005, use_arb_waveform=False,
                            pi_ge_before=True, pi_ef_before=True, pi_ge_after=False,
                            normalize=False, active_reset=False,
                            check_man_reset=[False, 0], check_man_reset_pi=[],
                            prepulse=False, pre_sweep_pulse=[], err_amp_reps=0)
    return cls.run_sweep(sweep_experiment_name=name)


def update_length_rabi_f0g1_sweep(expt_path, config_thisrun, ds_thisrun,
                                   man_mode_no=1, prev_data_fn=None):
    from multimode_expts.fit_display_classes import ChevronFitting
    from datetime import datetime
    temp_data, _, filename = prev_data_fn(expt_path, prefix='length_rabi_f0g1_sweep')
    print('File:', filename)
    ca = ChevronFitting(frequencies=temp_data['freq_sweep'],
                        time=temp_data['xpts'][0],
                        response_matrix=temp_data['avgi'],
                        config=config_thisrun)
    ca.analyze()
    ca.display_results(save_fig=True,
                       title=f'M{man_mode_no}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
    ds_thisrun.update_freq('M' + str(man_mode_no), ca.results['best_frequency_contrast'])
    print('Updated man freq to:', ca.results['best_frequency_contrast'])
    return ca


def do_error_amplification(
    config_thisrun, expt_path, config_path,
    reps=100, rounds=1, qubit=0, n_pulses=10,
    active_reset=False, man_reset=True, storage_reset=True,
    start=0, expts=10, step=100,
    parameter_to_test='gain',
    pulse_type=None,
):
    if pulse_type is None:
        pulse_type = ['qubit', 'ge', 'pi', 0]
    exp = meas.single_qubit.error_amplification.ErrorAmplificationExperiment(
        soccfg=None, path=expt_path, prefix='ErrorAmplificationExperiment',
        config_file=config_path)
    exp.cfg = AttrDict(deepcopy(config_thisrun))
    exp.cfg.expt = dict(reps=reps, qubit=qubit, qubits=[qubit],
                        active_reset=active_reset, man_reset=man_reset,
                        storage_reset=storage_reset, start=start,
                        expts=expts, step=step, n_pulses=n_pulses,
                        pulse_type=pulse_type, parameter_to_test=parameter_to_test,
                        rounds=rounds)
    exp.go(analyze=False, display=False, progress=True, save=True)
    return exp


def do_length_rabi_f0g1_general(config_thisrun, ds_thisrun, expt_path,
                                 config_path, man_mode_no=1):
    lr = meas.single_qubit.length_rabi_f0g1_general.LengthRabiGeneralF0g1Experiment(
        soccfg=None, path=expt_path, prefix='LengthRabiGeneralF0g1Experiment',
        config_file=config_path)
    lr.cfg = AttrDict(deepcopy(config_thisrun))
    lr.cfg.expt = dict(
        start=0.007, step=0.01, qubits=[0], expts=150, reps=100, rounds=1,
        gain=8000, freq=ds_thisrun.get_freq('M' + str(man_mode_no)),
        use_arb_waveform=False, pi_ge_before=True, pi_ef_before=True,
        pi_ge_after=True, normalize=False, active_reset=False,
        man_reset=True, stor_reset=True, check_man_reset=[False, 0],
        swap_lossy=False, check_man_reset_pi=[], prepulse=False,
        pre_sweep_pulse=[], err_amp_reps=0)
    lr.cfg.device.readout.relax_delay = [5000]
    lr.go(analyze=False, display=False, progress=True, save=True)
    from multimode_expts.fit_display_classes import LengthRabiFitting
    analysis = LengthRabiFitting(lr.data, config=lr.cfg)
    analysis.analyze()
    analysis.display(title_str='Length Rabi General F0g1')
    return analysis


def update_length_rabi_f0g1_general(analysis, ds_thisrun, man_mode_no=1):
    pi_len  = analysis.results['pi_length']
    pi2_len = analysis.results['pi2_length']
    gain    = analysis.cfg.expt['gain']
    freq    = analysis.cfg.expt['freq']
    ds_thisrun.update_all('M' + str(man_mode_no), freq, np.nan, pi_len, pi2_len, gain)
    print(f"Updated M{man_mode_no}: pi={pi_len:.4f} us, hpi={pi2_len:.4f} us, gain={gain}")


# ════════════════════════════════════════════════════════════════════════════
# STORAGE
# ════════════════════════════════════════════════════════════════════════════

def get_storage_mode_parameters(ds_thisrun, config_thisrun, man_mode_no, stor_mode_no):
    from MM_dual_rail_base import MM_dual_rail_base
    stor_name    = f'M{man_mode_no}-S{stor_mode_no}'
    freq         = ds_thisrun.get_freq(stor_name)
    gain         = ds_thisrun.get_gain(stor_name)
    pi_len       = ds_thisrun.get_pi(stor_name)
    h_pi_len     = ds_thisrun.get_h_pi(stor_name)
    flux_low_ch  = config_thisrun.hw.soc.dacs.flux_low.ch
    flux_high_ch = config_thisrun.hw.soc.dacs.flux_high.ch
    ch           = flux_low_ch if freq < 1000 else flux_high_ch
    mm           = MM_dual_rail_base(config_thisrun, soccfg=None)
    prep         = mm.prep_man_photon(man_mode_no)
    prepulse     = mm.get_prepulse_creator(prep).pulse.tolist()
    postpulse    = mm.get_prepulse_creator(prep[-1:-3:-1]).pulse.tolist()
    return freq, gain, pi_len, h_pi_len, ch, prepulse, postpulse


def do_stor_spectroscopy(config_thisrun, ds_thisrun, expt_path, config_path,
                         man_mode_no=1, stor_no=1):
    flux_spec = meas.single_qubit.rf_flux_spectroscopy_f0g1.FluxSpectroscopyF0g1Experiment(
        soccfg=None, path=expt_path, prefix='FluxSpectroscopyF0g1Experiment',
        config_file=config_path)
    flux_spec.cfg = AttrDict(deepcopy(config_thisrun))
    freq, gain, pi_len, h_pi_len, ch, prepulse, postpulse = \
        get_storage_mode_parameters(ds_thisrun, config_thisrun, man_mode_no, stor_no)
    flux_spec.cfg.expt = dict(
        start=freq - 30, step=0.3, expts=200, reps=100, qubit=[0],
        flux_drive=[ch, 1, 1000, 5], prepulse=True, postpulse=True,
        active_reset=True, pre_sweep_pulse=prepulse, post_sweep_pulse=postpulse)
    flux_spec.cfg.device.readout.relax_delay = [500]
    flux_spec.go(analyze=False, display=False, progress=True, save=True)
    return flux_spec


def update_stor_spectroscopy(flux_spec, ds_thisrun, man_mode_no=1, stor_no=1):
    freq = flux_spec.data['fit_avgi'][2]
    ds_thisrun.update_freq(f'M{man_mode_no}-S{stor_no}', freq)
    print(f"Updated M{man_mode_no}-S{stor_no} freq to {freq:.3f} MHz")


def do_sideband_general_sweep(
    config_thisrun, ds_thisrun, expt_path, config_path,
    freq_start, freq_stop, freq_step,
    reps=50, man_mode_no=1, stor_mode_no=1,
    start_time=0.007, liveplotting=True, exp_param_file=None,
):
    from multimode_expts.sequential_experiment_classes import sidebands_class
    name = 'sideband_general_sweep'
    cls  = sidebands_class(soccfg=None, path=expt_path, prefix=name,
                           config_file=config_path, exp_param_file=exp_param_file,
                           config_thisrun=config_thisrun, liveplotting=liveplotting)
    freq, gain, pi_len, h_pi_len, ch, prepulse, postpulse = \
        get_storage_mode_parameters(ds_thisrun, config_thisrun, man_mode_no, stor_mode_no)
    cls.loaded[name] = dict(
        start=start_time, step=pi_len / 5, qubits=[0], expts=15, reps=reps,
        rounds=1, freq_start=freq_start, freq_stop=freq_stop, freq_step=freq_step,
        flux_drive=[ch, freq, gain, 0.05], prepulse=True, postpulse=True,
        active_reset=False, man_reset=True, storage_reset=True,
        update_post_pulse_phase=[False, 1.07],
        pre_sweep_pulse=prepulse, post_sweep_pulse=postpulse)
    cls.yaml_cfg.device.readout.relax_delay = [8000]
    return cls.run_sweep(sweep_experiment_name=name)


def update_sideband_general_sweep(expt_path, config_thisrun, ds_thisrun,
                                   man_mode_no=1, stor_mode_no=1,
                                   update=True, prev_data_fn=None):
    from multimode_expts.fit_display_classes import ChevronFitting
    from datetime import datetime
    temp_data, _, filename = prev_data_fn(expt_path, prefix='sideband_general_sweep')
    print('File:', filename)
    ca = ChevronFitting(frequencies=temp_data['freq_sweep'],
                        time=temp_data['xpts'][0],
                        response_matrix=temp_data['avgi'],
                        config=config_thisrun)
    ca.analyze()
    ca.display_results(save_fig=True,
                       title=f'M{man_mode_no}-S{stor_mode_no}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
    if not update:
        return ca
    stor_name = f'M{man_mode_no}-S{stor_mode_no}'
    ds_thisrun.update_freq(stor_name, ca.results['best_frequency_period'])
    pi_len = abs(np.pi / ca.results['best_fit_params_period']['omega'])
    ds_thisrun.update_pi(stor_name, pi_len)
    ds_thisrun.update_h_pi(stor_name, pi_len / 2)
    print(f"Updated {stor_name}: freq={ca.results['best_frequency_period']:.4f}, pi={pi_len:.4f} us")
    return ca


def do_error_amp_storage(
    config_thisrun, ds_thisrun, expt_path, config_path,
    reps=100, rounds=1, qubit=0, n_start=1, n_step=1, n_pulses=10,
    active_reset=False, man_reset=True, storage_reset=True,
    span=1.0, expts=25, parameter_to_test='frequency',
    man_mode_no=1, stor_mode_no=1, stor_is_dump=False,
):
    label     = 'D' if stor_is_dump else 'S'
    pulse_type = ['storage', f'M{man_mode_no}-{label}{stor_mode_no}', 'pi', 0]
    freq, gain, pi_len, h_pi_len, ch, prepulse, postpulse = \
        get_storage_mode_parameters(ds_thisrun, config_thisrun, man_mode_no, stor_mode_no)
    if parameter_to_test == 'frequency':
        start = freq - span / 2
        step  = span / (expts - 1)
    elif parameter_to_test == 'gain':
        start = int(gain - span / 2)
        step  = int(span / (expts - 1))
    else:
        raise ValueError("parameter_to_test must be 'frequency' or 'gain'.")
    exp = meas.single_qubit.error_amplification.ErrorAmplificationExperiment(
        soccfg=None, path=expt_path, prefix='ErrorAmplificationExperiment',
        config_file=config_path)
    exp.cfg = AttrDict(deepcopy(config_thisrun))
    exp.cfg.expt = dict(reps=reps, qubit=qubit, qubits=[qubit],
                        active_reset=active_reset, man_reset=man_reset,
                        storage_reset=storage_reset, start=start,
                        expts=expts, step=step, n_start=n_start,
                        n_step=n_step, n_pulses=n_pulses,
                        pulse_type=pulse_type,
                        parameter_to_test=parameter_to_test, rounds=rounds)
    exp.go(analyze=False, display=False, progress=True, save=True)
    return exp


def do_sideband_general(config_thisrun, ds_thisrun, expt_path, config_path,
                        man_mode_no=1, stor_mode_no=1):
    sb = meas.single_qubit.sideband_general.SidebandGeneralExperiment(
        soccfg=None, path=expt_path, prefix='SidebandGeneralExperiment',
        config_file=config_path)
    sb.cfg = AttrDict(deepcopy(config_thisrun))
    freq, gain, pi_len, h_pi_len, ch, prepulse, postpulse = \
        get_storage_mode_parameters(ds_thisrun, config_thisrun, man_mode_no, stor_mode_no)
    sb.cfg.expt = dict(start=0.007, step=0.05, qubits=[0], expts=100, reps=200,
                       rounds=1, flux_drive=[ch, freq, gain], prepulse=True,
                       postpulse=True, active_reset=False, man_reset=True,
                       storage_reset=True, update_post_pulse_phase=[False, 1.07],
                       pre_sweep_pulse=prepulse, post_sweep_pulse=postpulse)
    sb.cfg.device.readout.relax_delay = [8000]
    sb.go(analyze=False, display=False, progress=True, save=True)
    from multimode_expts.fit_display_classes import LengthRabiFitting
    analysis = LengthRabiFitting(sb.data, config=sb.cfg)
    analysis.analyze()
    analysis.display(title_str='Sideband General', save_fig=True)
    return analysis


def update_sideband_general(analysis, ds_thisrun, man_mode_no=1, stor_mode_no=1):
    pi_len  = analysis.results['pi_length']
    pi2_len = analysis.results['pi2_length']
    gain    = analysis.cfg.expt['flux_drive'][2]
    freq    = analysis.cfg.expt['flux_drive'][1]
    ds_thisrun.update_all(f'M{man_mode_no}-S{stor_mode_no}',
                          freq, np.nan, pi_len, pi2_len, gain)
    print(f"Updated M{man_mode_no}-S{stor_mode_no}: pi={pi_len:.4f}, hpi={pi2_len:.4f}, gain={gain}")
