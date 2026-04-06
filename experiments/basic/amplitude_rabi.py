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
        self.initialize_multiple_loops()
        
         # Add sweep loop for the experiment
        # print(self.cfg.expt.expts)
        # self.add_loop("sweep_loop", self.cfg.expt.expts)
    
        
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
                self.delay_auto(t=0.0133, tag="wait_prepulse_" + str(pname))

        # Apply the main qubit pulse (variable amplitude or length)
        for i in range(cfg.expt.n_pulses):
            self.pulse(ch=self.cfg.expt.chan, name="rabi_pulse", t=0)
            self.delay_auto(t=0.01)

        # If checking EF transition with ge pulse, apply second pi pulse
        # if cfg.expt.checkEF and cfg.expt.pulse_ge:
        #     self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
        #     self.delay_auto(t=0.01, tag="wait ef 2")
            # pass
        if self.cfg.expt.get('postpulse', False):
            for pname in self.postpulse_names:
                self.pulse(ch=self.cfg.expt.postpulse[pname].chan, name=pname, t=0)
                print('Applied postpulse: ', pname)
                self.delay_auto(t=0.0133, tag="wait_postpulse_" + str(pname))

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
            primary_param_pulse = "gain"
            # Set pulse length based on type
            if e.type == "gauss":
                e.length = e.sigma * e.sigma_inc
            elif e.type == "const" or e.type == "flat_top":
                # Default to sigma if length isn't explicitly set
                if "length" not in e:
                    e.length = e.sigma
            par = "gain"

        elif e.sweep == "length":
            primary_param_pulse = "total_length"
            
            # Determine whether to sweep sigma (gauss) or length (others)
            par = "sigma" if e.type == "gauss" else "length"
        

        # 2. Set the parameter metadata
                
        primary_param = {"label": "rabi_pulse", "param": primary_param_pulse, "param_type": "pulse", 
                      "start": self.cfg.expt.start, "step": self.cfg.expt.step, "expts": self.cfg.expt.expts}
        self.sweep_param = {par: primary_param} 
        # Combine sweep_param and sweep_other_param dictionaries
        # Use the combine_sweep_params method from MMExperiment
        self.sweep_param = AttrDict(self.combine_sweep_params(self.sweep_param, getattr(self.cfg.expt, 'sweep_other_param', {})))
        print(self.sweep_param)
        self.initialize_sweep_variables()

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
