import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from qick import *
from slab import Experiment, AttrDict
from ..general.MM_program import MMProgram
from qick.asm_v2 import QickSweep1D
from ..general.MM_experiment import MMExperiment

class PulseProbeSpectroscopyProgram(MMProgram):
    def __init__(self, soccfg, final_delay, cfg):
        self.cfg = AttrDict(cfg)
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg)

    def _initialize(self, cfg):
        self.cfg = AttrDict(self.cfg)
        self.initialize_readout()
        self.initialize_non_readout_channels()
        self.initialize_waveforms()
        
        self.add_loop("freq_loop", self.cfg.expt.expts)
        self.initialize_multiple_loops()

        # Create the main pulse based on cfg.expt parameters
        pulse = {
            "chan": self.cfg.expt.chan,
            "freq": self.cfg.expt.freq,
            "gain": self.cfg.expt.gain,
            "phase": self.cfg.expt.phase,
            "length": self.cfg.expt.length,
            "type": self.cfg.expt.type,
            "sigma": self.cfg.expt.sigma,
            "sigma_inc": self.cfg.expt.sigma_inc,
            "ramp_sigma": self.cfg.expt.ramp_sigma,
            "ramp_sigma_inc": self.cfg.expt.ramp_sigma_inc,
        }
        super().make_pulse(pulse, "probe_pulse")

        

    def _body(self, cfg):
        # Apply prepulses if specified
        if self.cfg.expt.get("prepulse", False):
            for pname in self.prepulse_names:
                self.pulse(ch=self.cfg.expt.prepulse[pname].chan, name=pname, t=0)
                self.delay_auto(t=0.01, tag="wait_prepulse")

        # Apply the main pulse
        self.pulse(ch=self.cfg.expt.chan, name="probe_pulse", t=0)
        self.measure_wrapper()

class PulseProbeSpectroscopyExperiment(MMExperiment):
    def __init__(
        self,
        cfg_dict,
        prefix="",
        progress=True,
        display=True,
        save=True,
        analyze=True,
        go=False,
    ):
        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress)
        if go:
            super().run(display=display, progress=progress, save=save, analyze=analyze)

    def acquire(self, progress=False):
        self.cfg.device.readout.final_delay = self.cfg.expt.final_delay
        self.param = {"label": "pulse", "param": "freq", "param_type": "pulse"}
        self.cfg.expt.frequency = QickSweep1D(
            "freq_loop", self.cfg.expt.start, self.cfg.expt.start + self.cfg.expt.expts * self.cfg.expt.step
        )
        super().acquire(PulseProbeSpectroscopyProgram, progress=progress)
        self.cfg.expt.frequency = 0
        return self.data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data
        if fit:
            xdata = data["xpts"][1:-1]
            ydata = data["amps"][1:-1]
            data["fit"], data["fit_err"] = fitlor(xdata, ydata)
        return data

    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data
        xpts = data["xpts"][1:-1]
        plt.figure(figsize=(9, 11))
        plt.subplot(311, title="Pulse Probe Spectroscopy", ylabel="Amplitude [ADC units]")
        plt.plot(xpts, data["amps"][1:-1], "o-")
        if fit:
            plt.plot(xpts, lorfunc(data["xpts"][1:-1], *data["fit"]))
        plt.subplot(312, ylabel="I [ADC units]")
        plt.plot(xpts, data["avgi"][1:-1], "o-")
        plt.subplot(313, xlabel="Pulse Frequency (MHz)", ylabel="Q [ADC units]")
        plt.plot(xpts, data["avgq"][1:-1], "o-")
        plt.tight_layout()
        plt.show()

    def save_data(self, data=None):
        super().save_data(data=data)