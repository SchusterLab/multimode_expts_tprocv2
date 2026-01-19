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

"""
Measures the resonant frequency of the readout resonator when the qubit is in its ground state: sweep readout pulse frequency and look for the frequency with the maximum measured amplitude.

The resonator frequency is stored in the parameter cfg.device.readouti.frequency.

Note that harmonics of the clock frequency (6144 MHz) will show up as "infinitely"  narrow peaks!
"""
class ResonatorSpectroscopyProgram(MMProgram):
    def __init__(self, soccfg, final_delay, cfg):
        self.cfg = AttrDict(cfg)
        print("Resonator Spectroscopy Program Config:")
        
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg)

    def _initialize(self, cfg):
        self.cfg = AttrDict(self.cfg)
        print(self.cfg.expt)
        self.readout_frequency = self.cfg.expt.frequency
        self.readout_length = self.cfg.expt.length 
        self.readout_gain = self.cfg.expt.gain
        # Initialize 
        # print('initializing readout')
        self.initialize_readout()
        self.initialize_non_readout_channels()
        self.initialize_waveforms()
        
        # print('setting up readout loop')
        # Add frequency sweep loop
        self.add_loop("freq_loop", self.cfg.expt.expts)
        
        
    def _body(self, cfg):
        # pass
        
        if self.cfg.expt.get('pulse_e', False):
            self.pulse(ch=self.qubit_ch, name="pi_qubit_ge", t=0)
            # self.delay_auto(t=0.02, tag="wait_qubit_pulse")
        #for prepulse
        if self.cfg.expt.get('prepulse', False):
            for pname in self.prepulse_names:
                self.pulse(ch=self.cfg.expt.prepulse[pname].chan, name=pname, t=0)
                print('Applied prepulse: ', pname)
                self.delay_auto(t=0.02, tag="wait_prepulse")
            
        self.measure_wrapper()

# ====================================================== #

class ResonatorSpectroscopyExperiment(MMExperiment):
    """
    Resonator Spectroscopy Experiment
    Experimental Config
    start = start, # resonator frequency to be mixed up [MHz]
        step = span / expts, # min step ~1 Hz
        expts = 250, # Number experiments stepping from start
        reps = 500, # Number averages per point
        pulse_e = False, # add ge pi pulse prior to measurement
        pulse_f = False, # add ef pi pulse prior to measurement
        pulse_cavity = False,  # prepulse on cavity prior to measurement (False also disables next line)
        cavity_pulse = [4984.373226159381, 800, 2, 0], # [frequency, gain, length, phase]  const pulse
        qubit = 0,
    """
    
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
        """
        Initialize the resonator spectroscopy experiment.
        
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

    def acquire(self, progress=False):
        """
        Acquire data for the resonator spectroscopy experiment.
        
        Args:
            progress: Whether to show progress bar
            
        Returns:
            Acquired data
        """
        # Get qubit index and set final delay
        self.cfg.device.readout.final_delay = self.cfg.expt.final_delay
        # print()
        
        # Set parameter to sweep
        self.param = {"label": "readout_pulse", "param": "freq", "param_type": "pulse"}
        
        # Choose acquisition method based on loop flag
        # Standard acquisition with frequency sweep
        self.cfg.expt.frequency = QickSweep1D(
            "freq_loop", self.cfg.expt.start, self.cfg.expt.start+ self.cfg.expt.expts*self.cfg.expt.step
        )
        # print loop parameters
        # print("Frequency sweep parameters:")
        # print(f"Start: {self.cfg.expt.start} MHz")
        # print(f"Stop: {self.cfg.expt.stop} MHz")
        
        super().acquire(ResonatorSpectroscopyProgram, progress=progress)
        # get rid of qick asm object from cfg to make saving easier 
        self.cfg.expt.frequency = 0 
        

        return self.data

   

    def analyze(self, data=None, fit=True, findpeaks=False, verbose=True, fitparams=None, **kwargs):
        if data is None:
            data=self.data
            
        if fit:
            # fitparams = [f0, Qi, Qe, phi, scale]
            xdata = data["xpts"][1:-1]
            # ydata = data["avgi"][1:-1] + 1j*data["avgq"][1:-1]
            ydata = data['amps'][1:-1]
            fitparams=fitparams
            data['fit'], data['fit_err'] = fitter.fithanger(xdata, ydata, fitparams=fitparams)
            if isinstance(data['fit'], (list, np.ndarray)):
                f0, Qi, Qe, phi, scale, a0, slope = data['fit']
                if verbose:
                    print(f'\nFreq with minimum transmission: {xdata[np.argmin(ydata)]}')
                    print(f'Freq with maximum transmission: {xdata[np.argmax(ydata)]}')
                    print('From fit:')
                    print(f'\tf0: {f0}')
                    print(f'\tQi: {Qi}')
                    print(f'\tQe: {Qe}')
                    print(f'\tQ0: {1/(1/Qi+1/Qe)}')
                    print(f'\tkappa [MHz]: {f0*(1/Qi+1/Qe)}')
                    print(f'\tphi [radians]: {phi}')
            
        if findpeaks:
            maxpeaks, minpeaks = dsfit.peakdetect(data['amps'][1:-1], x_axis=data['xpts'][1:-1], lookahead=30, delta=5*np.std(data['amps'][:5]))
            data['maxpeaks'] = maxpeaks
            data['minpeaks'] = minpeaks
            
        return data

    def display(self, data=None, fit=True, findpeaks=False, **kwargs):
        if data is None:
            data=self.data 

        if 'lo' in self.cfg.hw:
            xpts = float(self.cfg.hw.lo.readout.frequency)*1e-6 + self.cfg.device.readout.lo_sideband*(self.cfg.hw.soc.dacs.readout.mixer_freq + data['xpts'][1:-1])
        else:
            xpts = data['xpts'][1:-1]

        plt.figure(figsize=(16,16))
        plt.subplot(311, title=f"Resonator Spectroscopy at gain {self.cfg.device.readout.gain}",  ylabel="Amps [ADC units]")
        plt.plot(xpts, data['amps'][1:-1],'o-')
        if fit:
            plt.plot(xpts, fitter.hangerS21func_sloped(data["xpts"][1:-1], *data["fit"]))
        if findpeaks:
            # for peak in np.concatenate((data['maxpeaks'], data['minpeaks'])):
            for peak in data['minpeaks']:
                plt.axvline(peak[0], linestyle='--', color='0.2')
                print(f'Found peak [MHz]: {peak[0]}')
        # plt.axvline(float(self.cfg.hw.lo.readout.frequency)*1e-6 + self.cfg.device.readout.lo_sideband*(self.cfg.hw.soc.dacs.readout.mixer_freq + 812.37), c='k', ls='--')
        # plt.axvline(7687.5, c='k', ls='--')

        plt.subplot(312, xlabel="Readout Frequency [MHz]", ylabel="I [ADC units]")
        plt.plot(xpts, data["avgi"][1:-1],'o-')

        plt.subplot(313, xlabel="Readout Frequency [MHz]", ylabel="Q [ADC units]")
        plt.plot(xpts, data["avgq"][1:-1],'o-')
        plt.show()
        
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)