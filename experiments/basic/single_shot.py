from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from qick import *
from slab import AttrDict
from ..general.MM_program import MMProgram
from qick.asm_v2 import QickSweep1D
from ..general.MM_experiment import MMExperiment
from copy import deepcopy

class SingleShotProgram(MMProgram):
    def __init__(self, soccfg, final_delay, cfg):
        self.cfg = AttrDict(cfg)
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg)

    def _initialize(self, cfg):
        self.cfg = AttrDict(self.cfg)
        self.readout_frequency = self.cfg.expt.readout_frequency
        self.readout_length = self.cfg.expt.readout_length 
        self.readout_gain = self.cfg.expt.readout_gain
        self.initialize_multiple_loops()
        self.add_loop("shotloop", self.cfg.expt.shots)
        # print('Readout frequency in single shot program:', self.readout_frequency)
        self.initialize_readout()
        self.initialize_non_readout_channels()
        self.initialize_waveforms()
        

    def _body(self, cfg):
        # Apply prepulses if specified
        if self.cfg.expt.get("prepulse", False):
            
            print('inside body of single shot program')
            # print(self.cfg.expt.prepulse)
            # for pname in self.cfg.expt.prepulse:
            #     self.pulse(ch=self.cfg.expt.prepulse[pname].chan, name=pname, t=0)
            #     self.delay_auto(t=0.01, tag="wait_prepulse" + pname)
        
        if self.cfg.expt.pulse_e:
            print('pulsing e')
            self.pulse(ch = self.qubit_ch, name = 'pi_qubit_ge', t=0)

        self.measure_wrapper()


    def collect_shots(self, offset=0):
        """
        Collect and process the raw I/Q data from the experiment.

        Parameters
        ----------
        offset : float, optional
            Offset to subtract from the raw data

        Returns
        -------
        tuple
            (i_shots, q_shots) arrays containing I and Q values for each shot
        """
        for i, (ch, rocfg) in enumerate(self.ro_chs.items()):
            # Get raw IQ data
            iq_raw = self.get_raw()
            # Extract I values and flatten
            # print(iq_raw)
            i_shots = iq_raw[i][:, :, 0, 0]
            i_shots = i_shots.flatten()
            # Extract Q values and flatten
            q_shots = iq_raw[i][:, :, 0, 1]
            q_shots = q_shots.flatten()

        return i_shots, q_shots

class SingleShotExperiment(MMExperiment):
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

    def acquire_single_shot_data(self, pulse_e, progress, add_swept_param_to_data = [False, None]):
        """
        Helper function to acquire single-shot data for a given pulse state.
        
        the fir

        Args:
            pulse_e (bool): Whether to pulse the excited state.
            progress (bool): Whether to show progress during acquisition.

        Returns:
            dict: Dictionary containing I/Q data for the given pulse state.
            
        """
        cfg = deepcopy(self.cfg)
        cfg.expt.pulse_e = pulse_e
        final_delay = self.cfg.expt.final_delay
        # print(cfg.expt)

        # Create and configure single-shot program
        single_shot_prog = SingleShotProgram(
            soccfg=self.soccfg, cfg=cfg, final_delay=final_delay
        )

        # Acquire data
        iq_list = single_shot_prog.acquire(
            self.im[self.cfg.aliases.soc],
            threshold=None,
            progress=progress,
        )
        
        #for swept parameters 
        if add_swept_param_to_data[0]:
            self.add_swept_parameters_to_data(single_shot_prog, add_swept_param_to_data[1])

        # Return I/Q data (the first few incides are n_chs, n_reads which is 0,0 in our case)
        return iq_list[0][0]

    def acquire(self, progress=False, debug=False):
        """
        Acquire data for the single-shot experiment.

        This method collects single-shot data for the ground state and,
        if configured, the excited state. If sweep_other_params is non-empty,
        only raw I/Q lists are stored.

        Parameters
        ----------
        progress : bool, optional
            Whether to show progress during acquisition
        debug : bool, optional
            Whether to print debug information

        Returns
        -------
        dict
            Dictionary containing the acquired data
        """
        data = {}

        # Check if sweep_other_params is non-empty
        if self.cfg.expt.get("sweep_other_param", {}):
            self.initialize_sweep_variables(params = {})
            # Acquire ground state data
            data["iqlist_g"] = self.acquire_single_shot_data(pulse_e=False, progress=progress)

            # Acquire excited state data if configured
            if self.cfg.expt.pulse_e:
                data["iqlist_e"] = self.acquire_single_shot_data(pulse_e=True, progress=progress, 
                                                                 add_swept_param_to_data=[True, data])
            self.clean_config_after_sweep()
            
        else:
            # Acquire ground state data
            ground_state_data = self.acquire_single_shot_data(pulse_e=False, progress=progress)
            data["Ig"] = ground_state_data[:,0]
            data["Qg"] = ground_state_data[:,1]

            # Acquire excited state data if configured
            if self.cfg.expt.pulse_e:
                excited_state_data = self.acquire_single_shot_data(pulse_e=True, progress=progress)
                data["Ie"] = excited_state_data[:,0]
                data["Qe"] = excited_state_data[:,1]

        # Store data and return
        self.data = data
        return data

    def analyze(self, data=None, fit=True, **kwargs):
        pass

    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data
        xpts = data["xpts"][1:-1]
        plt.figure(figsize=(9, 11))
        plt.subplot(311, title="Single Shot", ylabel="Amplitude [ADC units]")
        plt.plot(xpts, data["amps"][1:-1], "o-")
        plt.subplot(312, ylabel="I [ADC units]")
        plt.plot(xpts, data["avgi"][1:-1], "o-")
        plt.subplot(313, xlabel="Pulse Frequency (MHz)", ylabel="Q [ADC units]")
        plt.plot(xpts, data["avgq"][1:-1], "o-")
        plt.tight_layout()
        plt.show()

    def save_data(self, data=None):
        super().save_data(data=data)