import logging
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from qick.asm_v2 import AveragerProgramV2
from slab import AttrDict

from dataset import floquet_storage_swap_dataset, storage_man_swap_dataset
from .MM_base import MMBase


logger = logging.getLogger("qick.asm_V2")
# logger.setLevel(logging.ERROR)


# def print_debug():
#     import inspect
#     print(inspect.getfile(QickProgram))
#     print(inspect.getfile(AveragerProgram))
#     print(inspect.getfile(RAveragerProgram))


# class MM_base:
#     """
#     Methods and handy properties that are useful for both averager and raverager programs
#     Prepares the commonly used pulses in multimode experiments
#     such as qubit ge, ef, f0g1, M1-Sx π and π/2 pulses,
#     such that child classes can directly use the waveforms (gaussians) added here.
#     Also provides a more generic way to create custom pulses and many convenience functions.
#     """
    # def __init__(self, cfg: AttrDict, soccfg: AttrDict):
    #     self.cfg = cfg
    #     self.soccfg = soccfg
    #     # self.parse_config()
    #     # self._initialize()
        
 
    #     # ------------ Multiphoton COnfig ----------
    #     f_path = self.cfg.device.multiphoton_config.file
    #     import yaml
    #     with open(f_path, 'r') as f:
    #         self.multiphoton_cfg = AttrDict(yaml.safe_load(f))
class MMProgram(AveragerProgramV2, MMBase):
    """
    Base class for single-qubit quantum experiments using the QICK framework.

    This class extends AveragerProgramV2 to provide a higher-level interface for
    creating and running quantum experiments. It handles channel configuration,
    pulse generation, measurement, and data collection for a single qubit.

    The class is designed to be extended by specific experiment implementations
    that override the _body method to define the experiment sequence.
    """

    def __init__(self, soccfg, final_delay=50, cfg={}):
        """
        Initialize the QickProgram with hardware configuration and experiment parameters.

        Args:
            soccfg: System-on-chip configuration object containing hardware details
            final_delay: Delay time (in mus) after each experiment repetition
            cfg: Configuration dictionary containing experiment parameters
        """
        self.cfg = AttrDict(cfg)  # Convert to attribute dictionary for easier access

        # Update configuration with experiment-specific parameters
        self.cfg.update(self.cfg.expt)
        self.parse_config()  # parse the cfg to get the parameters
        # print(" I have parsed the config in MMProgram init")
        super().__init__(soccfg, self.cfg.expt.reps, final_delay, cfg=cfg)
        
        
    
    def _initialize(self, cfg, readout="standard"):
        """
        Initialize hardware channels and configure pulses for the experiment.

        This method sets up the ADC (analog-to-digital converter) and DAC
        (digital-to-analog converter) channels for qubit control and readout.
        It also configures the readout pulse and qubit control channels.

        Args:
            cfg: Configuration dictionary
            readout: Readout configuration type (default: "standard")
        """
        cfg = AttrDict(self.cfg)

        # self.parse_config()  # parse the cfg to get the parameters
        self.initialize_readout()
        self.initialize_non_readout_channels()
        self.initialize_waveforms()
        

    def _body(self, cfg):
        """
        Default experiment sequence implementation.

        This method defines the basic pulse sequence for the experiment.
        It should be overridden by subclasses to implement specific experiments.

        Args:
            cfg: Configuration dictionary
        """ 
        self.delay_auto(t=0.02, tag="waiting")
        self.measure_wrapper()
        
            

    def initialize_readout(self):
        """
        Initialize hardware channels and configure pulses for the experiment.

        This method sets up the ADC (analog-to-digital converter) and DAC
        (digital-to-analog converter) channels for qubit control and readout.
        It also configures the readout pulse and qubit control channels.

        Args:
            cfg: Configuration dictionary
            readout: Readout configuration type (default: "standard")
        """
        
        # self.trig_offset = self.cfg.device.readout.trig_offset
        
        # Set up readout generator
        if self.res_ch_type == "full":

            self.declare_gen(
            ch=self.res_ch, nqz=self.res_nqz
            )  # Declare resonator signal generator
            # print parameters of res_ch and res_nqz
            # print(f"res_ch: {self.res_ch}, res_nqz: {self.res_nqz}")

            # Create readout pulse
            pulse_args = {
            "ch": self.res_ch,
            "name": "readout_pulse",
            "style": "const",
            "ro_ch": self.adc_ch,
            "freq": self.readout_frequency,
            "phase": self.readout_phase,
            "gain": self.readout_gain,
            }
        
            pulse_args["length"] = self.readout_length
            print('pulse_args in initialize_readout', pulse_args)
            self.add_pulse(**pulse_args)
        # elif statements for mux and other types can be added here  <--------  

        # Configure readout settings
        if self.adc_ch_type == "dyn":
            self.declare_readout(
            self.adc_ch, length=self.readout_length
            )  # Configure ADC for readout

            self.add_readoutconfig(
            ch=self.adc_ch, name="readout", freq=self.readout_frequency, gen_ch=self.res_ch
            )
            # if self.adc_ch_type == 'dyn':
            self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)#self.trig_offset)

        elif self.adc_ch_type == "std":
            self.declare_readout(
            ch=self.adc_ch,
            length=self.readout_length,
            freq=self.readout_frequency,
            phase=self.readout_phase,
            gen_ch=self.res_ch,
            )
    def initialize_non_readout_channels(self):
        """
        Initialize non-readout channels such as qubit control and manipulation channels."""
        

        if "qubit" in self.cfg.hw.soc.dacs:
            # Set up qubit control channel
            self.declare_gen(
                ch=self.qubit_ch, nqz=self.qubit_nqz
            )  # Declare qubit signal generator
            
        if "manipulate_in" in self.cfg.hw.soc.dacs:
            # Set up manipulation channel
            self.declare_gen(
                ch=self.manipulate_ch, nqz=self.manipulate_nqz
            )  # Declare manipulation signal generator  
            
        # if "storage_in" in self.cfg.hw.soc.dacs:
        #     # Set up storage channel
        #     self.declare_gen(
        #         ch=self.storage_ch, nqz=self.storage_nqz
        #     )  # Declare storage signal generator
        
        # declare flux ch 
        if "flux" in self.cfg.hw.soc.dacs:
            # Set up storage channel
            self.declare_gen(
                ch=self.flux_ch, nqz=self.flux_nqz
            )  # Declare storage signal generator

    def initialize_waveforms(self):
        """
        Initialize waveforms with custom sigma, length, gain, and pulse type.
        """
        # 1. Define base pulses
        pulses_to_load = {
            "pi_qubit_ge": {
                "freq": self.f_ge, 
                "chan": self.qubit_ch, 
                "sigma": self.pi_ge_sigma, 
                "length": self.pi_ge_sigma, # will be multiplified by n_sigmas later
                "n_sigmas": self.pi_ge_sigma_inc, 
                "gain": self.pi_ge_gain,
                "type": "gauss" # Explicitly set type
            },
            "hpi_qubit_ge": {
                "freq": self.f_ge, 
                "chan": self.qubit_ch, 
                "sigma": self.hpi_ge_sigma, 
                "length": self.hpi_ge_sigma, # will be multiplified by n_sigmas later
                "n_sigmas": self.hpi_ge_sigma_inc, 
                "gain": self.hpi_ge_gain,
                "type": "gauss" # Explicitly set type
            },
            "pi_qubit_ef": {
                "freq": self.f_ef, 
                "chan": self.qubit_ch, 
                "sigma": self.pi_ef_sigma, 
                "length": self.pi_ef_sigma, # will be multiplified by n_sigmas later
                "n_sigmas": self.pi_ef_sigma_inc, 
                "gain": self.pi_ef_gain,
                "type": "gauss" # Explicitly set type
            },
            "hpi_qubit_ef": {
                "freq": self.f_ef, 
                "chan": self.qubit_ch, 
                "sigma": self.hpi_ef_sigma, 
                "length": self.hpi_ef_sigma, # will be multiplified by n_sigmas later
                "n_sigmas": self.hpi_ef_sigma_inc, 
                "gain": self.hpi_ef_gain,
                "type": "gauss" # Explicitly set type
            },
            # "slow_pi_ge": {
            #     "freq": self.f_ge,
            #     "chan": self.qubit_ch,
        }

        # 2. Add prepulses from config if they exist
        self.add_prepulse_postpulse(pulses_to_load= pulses_to_load, tag = 'prepulse' )
        self.add_prepulse_postpulse(pulses_to_load= pulses_to_load, tag = 'postpulse' )

        # 3. Process and initialize
        for name, p_info in pulses_to_load.items():
            channel = p_info["chan"][0] if isinstance(p_info["chan"], (list, np.ndarray)) else p_info["chan"]
            
            pulse_dict = {
                "freq": p_info["freq"],
                "chan": channel,
                "sigma": p_info["sigma"],
                "length": p_info["length"],
                "sigma_inc":  p_info["n_sigmas"],
                "gain": p_info.get("gain", 0),
                "type": p_info.get("type", "gauss"),   # Passed to make_pulse
                "phase": 0,
                "t":0
            }
            # print('pulse_dict in initialize_waveforms', pulse_dict)
            # print('trying to make the pulse')
            self.make_pulse(pulse_dict, name)
            
    def add_prepulse_postpulse(self, pulses_to_load, tag):
        """
        Initialize waveforms for the section named by `tag` under `self.cfg.expt`.
        `tag` should be a string like 'prepulse' or 'postpulse'. This function will
        merge entries into `pulses_to_load` and, when a pulse already exists in
        `pulses_to_load`, write back the chosen channel into the configuration entry.
        """
        pulses_conf = self.cfg.expt.get(tag, None)
        if pulses_conf is None:
            return

        # store names for reference
        try:
            names = list(pulses_conf.keys())
        except Exception:
            return

        if tag == 'prepulse':
            self.prepulse_names = names
        elif tag == 'postpulse':
            self.postpulse_names = names

        for pname, pp in pulses_conf.items():
            # if pulse already defined in pulses_to_load, attach channel info back to cfg
            if pname in pulses_to_load:
                try:
                    pulses_conf[pname]['chan'] = pulses_to_load[pname]['chan']
                except Exception:
                    pass
                continue

            # add the pulse to pulses_to_load (support both AttrDict-like objects and plain dicts)
            def get_attr(obj, key, default=None):
                if obj is None:
                    return default
                if hasattr(obj, 'get'):
                    return obj.get(key, default)
                return getattr(obj, key, default)

            pulses_to_load[pname] = {
                "freq": get_attr(pp, 'freq', None),
                "chan": get_attr(pp, 'chan', None),
                "sigma": get_attr(pp, 'sigma', None),
                "length": get_attr(pp, 'length', None),
                "n_sigmas": get_attr(pp, 'n_sigmas', 4),
                "gain": get_attr(pp, 'gain', None),
                "type": get_attr(pp, 'type', 'gauss'),
                "ramp_sigma": get_attr(pp, 'ramp_sigma', 0.02),
            }
        
    def initialize_multiple_loops(self): 
        if self.cfg.expt.sweep_param: # not empty 
            for param_name, param_values in self.cfg.expt.sweep_param.items():
                # print('initializing loop')
                # print(param_name)
                # print(param_values)
                self.add_loop(param_name, param_values.expts)
     
    def measure_wrapper(self):
        """
        Perform qubit measurement.

        This method implements the standard measurement sequence:
        1. Apply readout pulse to the resonator
        2. Apply LO pulse if available   # removed
        3. Trigger data acquisition
        4. Perform active reset if enabled

        Args:
            cfg: Configuration dictionary
        """
        # print('sendingreadoutconfig')
        if self.adc_ch_type == 'dyn':
            self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)
        cfg = AttrDict(self.cfg)
        # just to wait before reading out ...make sure no other pulse is running -- eesh 
        self.delay_auto(t=0.4, tag="wait_read")
        # print('added long wait')
        # Apply readout pulse to resonator]
        print("pulsing resonator")
        self.pulse(ch=self.res_ch, name="readout_pulse", t=0.0)
        # print('finished pulsing the resonator')
        # Trigger data acquisition
        print('triggering readout at time', self.trig_offset)
        self.trigger(
            ros=[self.adc_ch],
            pins=[0],
            t=self.trig_offset,
        )
        # print('finished triggering ')

    def make_pulse(self, pulse, name):
        """
        Create a pulse with specified parameters.

        This method creates different types of pulses (Gaussian, flat-top, or constant)
        based on the pulse parameters provided. The pulse is then added to the program
        for later execution.

        Args:
            pulse: Dictionary containing pulse parameters
            name: Name to assign to the created pulse
        
        pulse = {
                "freq": cfg.expt.prepulse_frequency,
                "gain": cfg.expt.prepulse_gain,
                "type": cfg.expt.prepulse_pulse_type,
                "sigma": cfg.expt.prepulse_sigma,
                "phase": 0,
                "chan": cfg.expt.prepulse_ch
            }

        Supported pulse types:
            - "gauss": Gaussian pulse with specified sigma
            - "flat_top": Flat-top pulse with Gaussian rise/fall
            - Other (default): Constant amplitude pulse
        """
        pulse = AttrDict(pulse)  # Convert to attribute dictionary
        # print('pulse in make_pulse', pulse)

        # Common pulse parameters
        if "chan" not in pulse:
            # print('qubit_ch', self.qubit_ch)
            pulse.chan = self.qubit_ch
        pulse_args = {
            "ch": pulse.chan,
            "name": name,
            "freq": pulse.freq,  # Pulse frequency
            "phase": pulse.phase,  # Pulse phase
            "gain": pulse.gain,  # Pulse amplitude
        }
        # print('pulse_args', pulse_args)

        # Create different pulse types based on pulse.type
        if pulse.type == "gauss":
            # Gaussian pulse, with sigma = sigma, total length = sigma * sigma_inc or length
            style = "arb"  # Arbitrary waveform

            # Determine pulse length
            if "sigma_inc" in pulse:
                length = pulse.sigma * pulse.sigma_inc  # Calculate from sigma
            else:
                length = pulse.sigma * 4  # Use provided length

            # Create Gaussian envelope
            # print('cre')
            self.add_gauss(
                ch=pulse.chan,
                name="ramp_" + str(name),
                sigma=pulse.sigma,  # Width of Gaussian
                length=length,
                even_length=False,
            )
            pulse_args["envelope"] = "ramp_" + str(name)  # Use Gaussian envelope

        elif pulse.type == "flat_top":
            # Flat-top pulse with Gaussian rise/fall

            # Determine pulse length
            print('creating flat top pulse')    
            if "length" in pulse:
                length = pulse.length
            else:
                length = pulse.sigma

            style = "flat_top"

            # Create Gaussian ramp for rise/fall
            if "ramp_sigma" not in pulse:
                pulse.ramp_sigma = 0.02
            if "ramp_sigma_inc" not in pulse:
                pulse.ramp_sigma_inc = 4
            ramp_length = pulse.ramp_sigma * pulse.ramp_sigma_inc
            print('adding gaussian ramp with sigma', pulse.ramp_sigma, 'and length', ramp_length)

            self.add_gauss(
                ch=pulse.chan,
                name="ramp",
                sigma=pulse.ramp_sigma,  # Width of rise/fall
                length=pulse.ramp_sigma*4,  # Length of rise/fall
                even_length=True,
            )
            pulse_args["envelope"] = "ramp"  # Use Gaussian envelope for edges
            pulse_args["length"] = length  # Total pulse length (this is of the flat part?)

        else:
            # Default: Constant amplitude pulse

            # Determine pulse length
            if "length" in pulse:
                length = pulse.length
                # print('using length parameter')
            else:
                length = pulse.sigma
                # print('not using length parameter')

            style = "const"  # Constant amplitude
            pulse_args["length"] = length

        # Set pulse style and add to program
        pulse_args["style"] = style
        # print('pulse_args', pulse_args)
        self.add_pulse(**pulse_args)

