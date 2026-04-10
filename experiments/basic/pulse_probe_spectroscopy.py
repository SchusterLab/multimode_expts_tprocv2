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
       
        
        # self.add_loop("freq_loop", self.cfg.expt.expts)
        self.initialize_multiple_loops()
        

        # Create the main pulse based on cfg.expt parameters
        # Define the main pulse parameters as a single dictionary
        # self.cfg.expt.probe_pulse_param should include the following keys:
        # - "chan": Channel for the pulse
        # - "freq": Frequency of the pulse
        # - "gain": Gain of the pulse
        # - "phase": Phase of the pulse
        # - "length": Length of the pulse
        # - "type": Type of the pulse (e.g., 'gauss', 'flat_top')
        # - "sigma": Sigma value for Gaussian pulses
        # - "sigma_inc": Increment for sigma
        # - "ramp_sigma": Ramp sigma for flat-top pulses
        # - "ramp_sigma_inc": Increment for ramp sigma
        pulse = self.cfg.expt.probe_pulse_param
        print('making probe pulse with parameters: ', pulse)

        super().make_pulse(pulse, "probe_pulse")
        # super().make_pulse(pulse, "readout_probe_pulse")

        

    def _body(self, cfg):
        # Apply prepulses if specified
        # print('enetered main body')
        # if self.adc_ch_type == 'dyn':
        #     self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)
        # cfg = AttrDict(self.cfg)
        if self.cfg.expt.get("prepulse", False):
            for pname in self.prepulse_names:
                print(pname)
                self.pulse(ch=self.cfg.expt.prepulse[pname].chan, name=pname, t=0)#self.cfg.expt.prepulse[pname].t)
                # Run delay by default. Only skip delay when delay_flag is explicitly False.
                delay_flag = self.cfg.expt.prepulse[pname].get("delay_flag", None)
                if delay_flag is False:
                    # explicit instruction to skip delay
                    pass
                else:
                    self.delay_auto(t=0.01, tag="wait_prepulse" + pname)
                    print('delayiong after prepulse', pname)

        # Apply the main pulse
        self.pulse(ch=self.cfg.expt.probe_pulse_param.chan, name="probe_pulse", t=0)
        # print('applied main pulse')
        if self.cfg.expt.get("postpulse", False):
            for pname in self.cfg.expt.postpulse:
                self.pulse(ch=self.cfg.expt.postpulse[pname].chan, name=pname, t=0)
                print('Applied postpulse: ', pname)
                self.delay_auto(t=0.01, tag="wait_postpulse" + pname)
        
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
        # if go:
        #     super().run(display=display, progress=progress, save=save, analyze=analyze)

    def acquire(self, progress=False):
        # self.cfg.device.readout.final_delay = self.cfg.expt.final_delay
        self.param = {"label": "probe_pulse", "param": "freq", "param_type": "pulse"}
        # Compute the frequency sweep separately
        # Convert AttrDict to a standard dictionary
        
        freq_sweep = QickSweep1D(
            "freq_loop", self.cfg.expt.start, self.cfg.expt.start + self.cfg.expt.expts * self.cfg.expt.step
        )
        primary_var = 'freq'
        primary_var_param_dict = {"label": "probe_pulse", "param": "freq", "param_type": "pulse", 
                      "start": self.cfg.expt.start, "step": self.cfg.expt.step, "expts": self.cfg.expt.expts, 
                      "parent_dict": 'probe_pulse_param'}

        # note if variable inside dict then the Attr Dict apllies beforehand will make it immutable.so have to do this correction
        self.sweep_param = {primary_var: primary_var_param_dict} 
        self.sweep_param = AttrDict(self.combine_sweep_params(self.sweep_param, getattr(self.cfg.expt, 'sweep_other_param', {})))
        # print(self.sweep_param)
        self.initialize_sweep_variables() # finished making qick objects 
        if "readout_probe" in self.cfg.expt.prepulse: 
            self.cfg.expt.prepulse.readout_probe.gain = self.cfg.expt.readout_probe_gain
            print(self.cfg.expt.prepulse.readout_probe)

        super().acquire(PulseProbeSpectroscopyProgram, progress=progress)
        # self.cfg.expt.frequency = 0
        self.cfg.expt.probe_pulse_param.freq = 0
    
    
        # if "readout_probe" in self.cfg.expt.prepulse: 
        #     self.cfg.expt.prepulse.readout_probe.gain = None
        return self.data

    def analyze(self, data=None, fit=True, **kwargs):
        pass

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