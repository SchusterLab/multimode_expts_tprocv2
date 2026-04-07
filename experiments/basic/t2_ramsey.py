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


class RamseyProgram(MMProgram):
    def __init__(self, soccfg, final_delay, cfg):
        self.cfg = AttrDict(cfg)
        # print("Resonator Spectroscopy Program Config:")
        
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg)

    def _initialize(self, cfg):
        self.cfg = AttrDict(cfg)
        self.initialize_readout()
        self.initialize_non_readout_channels()
        self.initialize_waveforms()
        
        pulse = {
            "sigma": self.cfg.expt.sigma,  
            "sigma_inc": self.cfg.expt.sigma_inc,
            "freq": self.cfg.expt.freq,
            "gain": self.cfg.expt.gain ,
            "phase": 0,  # First pulse has zero phase
            "type": self.cfg.expt.type,
        }
        print(pulse)

        # Create first π/2 pulse (preparation)
        super().make_pulse(pulse, "pi2_prep")

        # Create second π/2 pulse (readout) with phase that depends on wait time
        # Phase advances at rate of ramsey_freq (MHz) * wait_time (μs) * 360 (deg/cycle)
        pulse["phase"] = self.cfg.expt.wait_time * 360 * self.cfg.expt.ramsey_freq
        super().make_pulse(pulse, "pi2_read")

        # Create loop for sweeping wait time
        # self.add_loop("wait_loop", self.cfg.expt.expts)
        self.initialize_multiple_loops()


    def _body(self, cfg):
        cfg=AttrDict(self.cfg)
        
        # prepulse
        if self.cfg.expt.get('prepulse', False):
            for pname in self.prepulse_names:
                self.pulse(ch=self.cfg.expt.prepulse[pname].chan, name=pname, t=0)
                print('Applied prepulse: ', pname)
        
    

        # Configure readout
        # if self.adc_type == "dyn":
        #     self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)
        if self.adc_ch_type == 'dyn':
            self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)
        cfg = AttrDict(self.cfg)

        # For EF transition check in Ramsey: Apply π pulse to excite |g⟩ to |e⟩ first
        # if hasattr(cfg.expt, "checkEF") and cfg.expt.checkEF:
        #     self.pulse(ch=self.qubit_ch, name="pi_qubit_ge", t=0)
        #     self.delay_auto(t=0.01, tag="wait ef")  # Small buffer delay

        # First π/2 pulse (preparation)
        self.pulse(ch=self.qubit_ch, name="pi2_prep", t=0.0)
        # self.pulse(ch=self.qubit_ch, name="pi_qubit_ge", t=0.0)


        # if cfg.expt.num_pi > 0:
        #     self.delay_auto(t=cfg.expt.wait_time / cfg.expt.num_pi / 2, tag="wait")

        #     # Apply π pulses for Echo protocol (or multiple-pulse Echo)
        #     for i in range(cfg.expt.num_pi):
        #         self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)  # π pulse
        #         if i < cfg.expt.num_pi - 1:
        #             self.delay_auto(
        #                 t=cfg.expt.wait_time / cfg.expt.num_pi + 0.01,
        #                 tag=f"wait{i}",
        #             )  # Wait time
        #     self.delay_auto(
        #         t=cfg.expt.wait_time / cfg.expt.num_pi / 2 + 0.01, tag=f"wait{i+1}"
        #     )
        # else:
        self.delay_auto(t=cfg.expt.wait_time, tag="wait")

        # Second π/2 pulse (readout)
        self.pulse(ch=self.qubit_ch, name="pi2_read", t=0)
        self.delay_auto(t=0.01, tag="wait rd")  # Small buffer delay

        # For EF transition check in Ramsey: Apply π pulse to return to |g⟩ for readout
        # if hasattr(cfg.expt, "checkEF") and cfg.expt.checkEF:
        #     self.pulse(ch=self.qubit_ch, name="pi_qubit_ge", t=0)
        #     self.delay_auto(t=0.01, tag="wait ef 2")  # Small buffer delay

        

        # align channels and measure
        self.measure_wrapper()



class RamseyExperiment(MMExperiment):
    """
    Ramsey experiment
    Experimental Config:
    expt = dict(
        start: wait time start sweep [us]
        step: wait time step - make sure nyquist freq = 0.5 * (1/step) > ramsey (signal) freq!
        expts: number experiments stepping from start
        ramsey_freq: frequency by which to advance phase [MHz]
        reps: number averages per experiment
        rounds: number rounds to repeat experiment sweep
        checkEF: does ramsey on the EF transition instead of ge
    )
    """

    def __init__(
        self,
        cfg_dict,
        qi=0,
        go=True,
        params={},
        prefix=None,
        fname=None,
        progress=True,
        style="",
        disp_kwargs=None,
        min_r2=None,
        max_err=None,
        display=True,
        print_=False,
    ):
        """
        Initialize the T2 experiment.

        This experiment measures T2 using Ramsey, Echo, or CPMG protocols.
        Default `params` values:
        - 'reps': Number of repetitions, doubled from default (default: `2 * self.reps`)
        - 'rounds': Number of software averages, from `self.rounds`
        - 'expts': Number of wait time points (default: 100)
        - 'span': Total span of wait times in µs, set to ~3xT2 (default: `3 * self.cfg.device.qubit[par][qi]`)
        - 'start': Start time for wait sweep in µs (default: 0.01)
        - 'ramsey_freq': Ramsey frequency for phase advancement, 'smart' sets it to 1.5/T2 (default: 'smart')
        - 'active_reset': If True, uses active reset (default: from `cfg.device.readout.active_reset[qi]`)
        - 'experiment_type': 'ramsey', 'echo', or 'cpmg' (default: 'ramsey')
        - 'acStark': If True, applies an AC Stark pulse during the wait time (Ramsey only) (default: False)
        - 'checkEF': If True, measures the |e>-|f> transition (default: False)
        - 'num_pi': Number of π pulses for Echo/CPMG (default: 1 for 'echo', 0 for 'ramsey')

        Args:
            cfg_dict (dict): Configuration dictionary.
            qi (int): Qubit index to measure.
            go (bool): Whether to immediately run the experiment.
            params (dict): Additional parameters to override defaults.
            prefix (str): Filename prefix for saved data.
            fname (str): Full filename for saved data.
            progress (bool): Whether to show a progress bar.
            style (str): Measurement style ('fine' for more averages, 'fast' for fewer points).
            disp_kwargs (dict): Display options.
            min_r2 (float): Minimum R² value for acceptable fit.
            max_err (float): Maximum error for acceptable fit.
            display (bool): Whether to display results.
            print (bool): If True, prints the experiment config and exits.
        """
        # Determine experiment type and parameter name based on protocol
        if "experiment_type" in params and params["experiment_type"] == "echo":
            par = "T2e"  # Echo uses T2e parameter
            name = "echo"
        else:
            par = "T2r"  # Ramsey uses T2r parameter
            name = "ramsey"

        # Set appropriate filename prefix
        if prefix is None:
            ef = "ef_" if "checkEF" in params and params["checkEF"] else ""
            prefix = f"{name}_{ef}qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress)


        if go:
            super().run(display=display, progress=progress, save=True, analyze=False)

    
        # if go:
        #     super().qubit_run(
        #         display=display,
        #         progress=progress,
        #     )

    def acquire(self, progress=False):
        """
        Acquire T2 measurement data.

        This method:
        1. Sets up the wait time sweep parameters
        2. Runs the T2Program to collect data for each wait time
        3. Adjusts x-axis values to account for echo protocol

        Args:
            progress: Whether to show progress bar

        Returns:
            Measurement data dictionary
        """
        # Define parameter metadata for plotting
        primary_param = {"label": "wait", "param": "t", "param_type": "time", 
                      "start": self.cfg.expt.start, "step": self.cfg.expt.step, "expts": self.cfg.expt.expts}

        # Create a 1D sweep for the wait time from start to start+span
        primary_variable = "wait_time"

        self.sweep_param = {primary_variable: primary_param} 
        # Combine sweep_param and sweep_other_param dictionaries
        # Use the combine_sweep_params method from MMExperiment
        # self.sweep_param = self.combine_sweep_params(self.sweep_param, getattr(self.cfg.expt, 'sweep_other_param', {}))
        # self.initialize_sweep_variables()
        self.sweep_param = AttrDict(self.combine_sweep_params(self.sweep_param, getattr(self.cfg.expt, 'sweep_other_param', {})))
        print(self.sweep_param)
        self.initialize_sweep_variables()    

        # Run the T2Program to acquire data
        # print(RamseyProgram)
        super().acquire(RamseyProgram, progress=progress)

        # Adjust x-axis values to account for echo protocol
        # For echo, the effective wait time is longer due to the π pulses
        # if self.cfg.expt.num_pi == 0:
        #     coef = 1
        # else:
        #     coef = 2*self.cfg.expt.num_pi  # For echo, we have num_pi + 1 segments
        # self.data["xpts"] = coef * self.data["xpts"]
        
        # get rid of extra wait time info for saving purposes
        # self.cfg.expt.pop('wait_time', None)

        return self.data

    def analyze(self, data=None, fit=True, fitparams = None, **kwargs):
        pass

    def display(self, data=None, fit=True, **kwargs):
        pass

    # def save_data(self, data=None):
    #     """
    #     Save experiment data to an HDF5 file.

    #     Args:
    #         data: Dictionary containing experiment data.
    #     """
    #     import json

    #     # Helper function to check if an object is serializable
    #     def is_serializable(obj):
    #         try:
    #             json.dumps(obj)
    #             return True
    #         except (TypeError, OverflowError):
    #             return False

    #     # Ensure all NumPy arrays have a compatible dtype
    #     def sanitize_data(obj):
    #         if isinstance(obj, dict):  # Recursively process dictionaries
    #             return {k: sanitize_data(v) for k, v in obj.items()}
    #         elif isinstance(obj, np.ndarray):  # Ensure NumPy arrays have a compatible dtype
    #             return obj.astype(np.float64)  # Cast to float64 for compatibility
    #         elif isinstance(obj, list):  # Recursively process lists
    #             return [sanitize_data(v) for v in obj]
    #         else:
    #             return obj  # Return the object as-is if already compatible

    #     # Sanitize the data dictionary
    #     data = sanitize_data(data)

    #     # Save the sanitized data to the HDF5 file
    #     try:
    #         print(f"Saving {self.fname}")
    #         with self.datafile() as f:
    #             for k, d in data.items():
    #                 try:
    #                     f.add(k, np.array(d))
    #                 except TypeError as e:
    #                     print(f"Error saving key '{k}': {e}. Replacing with None.")
    #                     f.add(k, np.array(None))
    #     except Exception as e:
    #         print(f"Error in saving data: {e}")
    #         raise

    #     return self.fname

