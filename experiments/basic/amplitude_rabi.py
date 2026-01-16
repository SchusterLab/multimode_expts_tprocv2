import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
import time

from qick import *
from qick.helpers import gauss
from slab import Experiment, dsfit, AttrDict

# import experiments.fitting.fitting as fitter
from ..general.MM_program import MMProgram
from qick.asm_v2 import QickSweep1D
from ..general.MM_experiment import MMExperiment

class RabiProgram(MMProgram):
    def __init__(self, soccfg, final_delay, cfg):
        self.cfg = AttrDict(cfg)
        # print("Resonator Spectroscopy Program Config:")
        
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg)

    def _initialize(self, cfg):
        self.cfg = AttrDict(cfg)
        self.initialize_readout()
        self.initialize_non_readout_channels()
        self.initialize_waveforms()
        
         # Add sweep loop for the experiment
        print(self.cfg.expt.expts)
        self.add_loop("sweep_loop", self.cfg.expt.expts)
    
        
        pulse = {
            "chan": self.cfg.expt.chan,
            "sigma": self.cfg.expt.sigma,
            "length": self.cfg.expt.length,
            "freq": self.cfg.expt.freq,
            "gain": self.cfg.expt.gain,
            "phase": 0,
            "type": self.cfg.expt.type,
            "sigma_inc": self.cfg.expt.sigma_inc,
            "ramp_sigma": self.cfg.expt.ramp_sigma,
            "ramp_sigma_inc": self.cfg.expt.ramp_sigma_inc,
        }
        print('RabiProgram pulse:', pulse)
    
                
        super().make_pulse(pulse, "rabi_pulse")

        
    def _body(self, cfg):
        """
        Define the main body of the experiment sequence.

        Args:
            cfg: Configuration dictionary containing experiment parameters
        """
        self.cfg = AttrDict(self.cfg)
        # If checking EF transition with ge pulse, apply first pi pulse
        # if self.cfg.expt.checkEF and self.cfg.expt.pulse_ge:
        #     self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
        #     self.delay_auto(t=0.01, tag="wait ef")
        #for prepulse
        if self.cfg.expt.get('prepulse', False):
            for pname in self.prepulse_names:
                self.pulse(ch=self.cfg.expt.prepulse[pname].chan, name=pname, t=0)
                print('Applied prepulse: ', pname)
                self.delay_auto(t=0.0133, tag="wait_prepulse")

        # Apply the main qubit pulse (variable amplitude or length)
        for i in range(cfg.expt.n_pulses):
            self.pulse(ch=self.qubit_ch, name="rabi_pulse", t=0)
            self.delay_auto(t=0.01)

        # If checking EF transition with ge pulse, apply second pi pulse
        # if cfg.expt.checkEF and cfg.expt.pulse_ge:
        #     self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
        #     self.delay_auto(t=0.01, tag="wait ef 2")
            # pass


        # Perform measurement
        self.measure_wrapper()


# ====================================================== #

class RabiExperiment(MMExperiment):
    """
    Amplitude Rabi Experiment
    Experimental Config:
    expt = dict(
        start: qubit gain [dac level]
        step: gain step [dac level]
        expts: number steps
        reps: number averages per expt
        rounds: number repetitions of experiment sweep
        sigma_test: gaussian sigma for pulse length [us] (default: from pi_ge in config)
        pulse_type: 'gauss' or 'flat_top' or 'drag' or 'const' (implementation not checked)
    )
    """

    def __init__(self,
        cfg_dict,
        prefix="Rabi Experiment",
        progress=True,
        display=True,
        save=True,
        analyze=True,
        go=True,

    ):
        """
        Initialize the Rabi experiment.
        
        Args:
            cfg_dict: Configuration dictionary
            prefix: Prefix for data files
            progress: Whether to show progress bar
            display: Whether to display results
            save: Whether to save data
            analyze: Whether to analyze data
            qi: Qubit index
            go: Whether to run the experiment immediately
            params: Additional parameters to override defaults
            style: Style of experiment ('coarse' or 'fine')
        """

        
        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress)


        if go:
            super().run(display=display, progress=progress, save=save, analyze=analyze)

    def acquire(self, progress=False, debug=False):
        """
        Acquire data for the Rabi experiment using AttrDict dot notation.
        """
        e = self.cfg.expt  # Short alias for cleaner code
        d = self.cfg.device
        
        print(e)
        d.readout.final_delay = e.final_delay

        # 1. Configure the sweep
        if e.sweep == "amp":
            param_pulse = "gain"
            print(f'Sweeping amps from {e.start} to {e.max_gain}')
            
            # Create QICK Sweep
            e.gain = QickSweep1D("sweep_loop", e.start, e.max_gain)
            
            # Set pulse length based on type
            if e.type == "gauss":
                e.length = e.sigma * e.sigma_inc
            elif e.type == "const" or e.type == "flat_top":
                # Default to sigma if length isn't explicitly set
                if "length" not in e:
                    e.length = e.sigma

        elif e.sweep == "length":
            param_pulse = "total_length"
            
            # Determine whether to sweep sigma (gauss) or length (others)
            par = "sigma" if e.type == "gauss" else "length"
            
            e[par] = QickSweep1D("sweep_loop", e.start, e.max_length)
        print(f'Confige after sweep setup: {e}')

        # 2. Set the parameter metadata
        self.param = {
            "label": "rabi_pulse",
            "param": param_pulse,
            "param_type": "pulse",
        }
        
        print('Outputting cfg keys:', self.cfg.keys())

        # 3. Acquire data using the RabiProgram
        super().acquire(RabiProgram, progress=progress)
        
        # 4. Cleanup: Reformat config to remove QickSweep objects (makes saving to JSON/HDF5 easier)
        if e.sweep == "amp":
            e.gain = 0
        elif e.sweep == "length":
            if e.type == "gauss":
                e.sigma = 0
            else:
                e.length = 0

        return self.data

    def analyze(self, data=None, fit=True, **kwargs):
        pass
    def display(self, data=None, fit=True, **kwargs):
        pass

# ====================================================== #

class AmplitudeRabiChevronExperiment(Experiment):
    """
    Amplitude Rabi Experiment
    Experimental Config:
    expt = dict(
        start_f: start qubit frequency (MHz),
        step_f: frequency step (MHz),
        expts_f: number of experiments in frequency,
        start_gain: qubit gain [dac level]
        step_gain: gain step [dac level]
        expts_gain: number steps
        reps: number averages per expt
        rounds: number repetitions of experiment sweep
        sigma_test: gaussian sigma for pulse length [us] (default: from pi_ge in config)
        flat_length: flat top length [us] (implementation not checked)
        pulse_type: 'gauss' or 'flat_top' or 'drag' or 'const' (implementation not checked)
        checkEF: bool
        checkZZ: bool
        prepulse: bool
        postpulse: bool
        pulse_ge_init: bool
        pulse_ge_after: bool
    )
    """

    def __init__(self, soccfg=None, path='', prefix='AmplitudeRabiChevron', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        # expand entries in config that are length 1 to fill all qubits
        num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items() :
                if isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if not(isinstance(value3, list)):
                                value2.update({key3: [value3]*num_qubits_sample})
                elif not(isinstance(value, list)):
                    subcfg.update({key: [value]*num_qubits_sample})

        if 'sigma_test' not in self.cfg.expt:
            self.cfg.expt.sigma_test = self.cfg.device.qubit.pulses.pi_ge.sigma

        freqpts = self.cfg.expt["start_f"] + self.cfg.expt["step_f"]*np.arange(self.cfg.expt["expts_f"])
        data={"xpts":[], "freqpts":[], "avgi":[], "avgq":[], "amps":[], "phases":[]}
        adc_ch = self.cfg.hw.soc.adcs.readout.ch[0]

        self.cfg.expt.start = self.cfg.expt.start_gain
        self.cfg.expt.step = self.cfg.expt.step_gain
        self.cfg.expt.expts = self.cfg.expt.expts_gain
        for freq in tqdm(freqpts):
            self.cfg.device.qubit.f_ge = [freq]
            amprabi = AmplitudeRabiProgram(soccfg=self.soccfg, cfg=self.cfg)

            xpts, avgi, avgq = amprabi.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=False, debug=debug)

            avgi = avgi[adc_ch][0]
            avgq = avgq[adc_ch][0]
            amps = np.abs(avgi+1j*avgq) # Calculating the magnitude
            phases = np.angle(avgi+1j*avgq) # Calculating the phase

            data["avgi"].append(avgi)
            data["avgq"].append(avgq)
            data["amps"].append(amps)
            data["phases"].append(phases)

        data['xpts'] = xpts
        data['freqpts'] = freqpts
        for k, a in data.items():
            data[k] = np.array(a)
        self.data=data
        return data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data
        pass

    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data

        x_sweep = data['xpts']
        y_sweep = data['freqpts']
        avgi = data['avgi']
        avgq = data['avgq']

        plt.figure(figsize=(10,8))
        plt.subplot(211, title="Amplitude Rabi", ylabel="Frequency [MHz]")
        plt.imshow(
            np.flip(avgi, 0),
            cmap='viridis',
            extent=[x_sweep[0], x_sweep[-1], y_sweep[0], y_sweep[-1]],
            aspect='auto')
        plt.colorbar(label='I [ADC level]')
        plt.clim(vmin=None, vmax=None)

        plt.subplot(212, xlabel="Gain [dac units]", ylabel="Frequency [MHz]")
        plt.imshow(
            np.flip(avgq, 0),
            cmap='viridis',
            extent=[x_sweep[0], x_sweep[-1], y_sweep[0], y_sweep[-1]],
            aspect='auto')
        plt.colorbar(label='Q [ADC level]')
        plt.clim(vmin=None, vmax=None)

        if fit: pass

        plt.tight_layout()
        plt.show()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname
