import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
import time

from qick import *
from qick.helpers import gauss
from slab import Experiment, dsfit, AttrDict

from ..general.MM_program import MMProgram
from qick.asm_v2 import QickSweep1D
from ..general.MM_experiment import MMExperiment


class T1Program(MMProgram):
    def __init__(self, soccfg, final_delay, cfg):
        self.cfg = AttrDict(cfg)
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg)

    def _initialize(self, cfg):
        self.cfg = AttrDict(cfg)
        self.initialize_readout()
        self.initialize_non_readout_channels()
        self.initialize_waveforms()

        if self.cfg.expt.get("qubit_pulse", True):
            pulse = {
                "sigma": self.cfg.expt.sigma,
                "sigma_inc": self.cfg.expt.sigma_inc,
                "freq": self.cfg.expt.freq,
                "gain": self.cfg.expt.gain,
                "phase": 0,
                "type": self.cfg.expt.type,
            }
            print(pulse)

            # Create π pulse to excite qubit to |e⟩
            super().make_pulse(pulse, "pi_prep")

        self.initialize_multiple_loops()

    def _body(self, cfg):
        cfg = AttrDict(self.cfg)

        # prepulse
        if self.cfg.expt.get('prepulse', False):
            for pname in self.prepulse_names:
                self.pulse(ch=self.cfg.expt.prepulse[pname].chan, name=pname, t=0)
                print('Applied prepulse: ', pname)

        # Configure readout
        if self.adc_ch_type == 'dyn':
            self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)

        # π pulse to excite qubit from |g⟩ to |e⟩
        if self.cfg.expt.get("qubit_pulse", True):
            self.pulse(ch=self.qubit_ch, name="pi_prep", t=0.0)

        # Wait time — qubit decays from |e⟩ back toward |g⟩
        self.delay_auto(t=cfg.expt.wait_time, tag="wait")

        self.delay_auto(t=0.01, tag="wait rd")  # Small buffer before readout

        # Measure population remaining in |e⟩
        self.measure_wrapper()


class T1Experiment(MMExperiment):
    """
    T1 experiment — measures energy relaxation time.

    Pulse sequence:
        π pulse → wait time → measure

    Experimental Config:
    expt = dict(
        start:       wait time start sweep [us]
        step:        wait time step [us]
        expts:       number of wait time points
        reps:        number averages per experiment
        rounds:      number rounds to repeat experiment sweep
        span:        total span of wait times [us], set to ~3xT1
        sigma:       pulse sigma [us]
        sigma_inc:   pulse sigma increment
        freq:        qubit drive frequency [MHz]
        gain:        pulse gain
        type:        pulse type (e.g. 'gauss', 'flat_top')
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
        disp_kwargs=None,
        min_r2=None,
        max_err=None,
        display=True,
        print_=False,
    ):
        """
        Initialize the T1 experiment.

        Args:
            cfg_dict (dict): Configuration dictionary.
            qi (int): Qubit index to measure.
            go (bool): Whether to immediately run the experiment.
            params (dict): Additional parameters to override defaults.
            prefix (str): Filename prefix for saved data.
            fname (str): Full filename for saved data.
            progress (bool): Whether to show a progress bar.
            disp_kwargs (dict): Display options.
            min_r2 (float): Minimum R² value for acceptable fit.
            max_err (float): Maximum error for acceptable fit.
            display (bool): Whether to display results.
            print_ (bool): If True, prints the experiment config and exits.
        """
        if prefix is None:
            prefix = f"t1_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress)

        if go:
            super().run(display=display, progress=progress, save=True, analyze=False)

    def acquire(self, progress=False):
        """
        Acquire T1 measurement data.

        Sweeps wait time from start to start+span and measures
        qubit population decay from |e⟩ to |g⟩.

        Args:
            progress: Whether to show progress bar

        Returns:
            Measurement data dictionary
        """
        primary_param = {
            "label": "wait",
            "param": "t",
            "param_type": "time",
            "start": self.cfg.expt.start,
            "step": self.cfg.expt.step,
            "expts": self.cfg.expt.expts,
        }

        primary_variable = "wait_time"

        self.sweep_param = {primary_variable: primary_param}
        self.sweep_param = AttrDict(
            self.combine_sweep_params(
                self.sweep_param,
                getattr(self.cfg.expt, 'sweep_other_param', {})
            )
        )
        print(self.sweep_param)
        self.initialize_sweep_variables()

        super().acquire(T1Program, progress=progress)

        return self.data

    def analyze(self, data=None, fit=True, fitparams=None, **kwargs):
        pass

    def display(self, data=None, fit=True, **kwargs):
        pass