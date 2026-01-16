# define a class MM_base 
class MMBase:
    def __init__(self, cfg):
        self.cfg = cfg
        # self.parse_config()
        
    def parse_config(self):
        """
        "Software" initialization: parses the cfg and stores parameters in self for easy access.
        Merges channel info, hardware filter parameters, and pulse calibrations.
        """
        cfg = self.cfg

        # ----------- Channel Info -----------
        # Readout
        self.adc_ch, self.adc_ch_type = cfg.hw.soc.adcs.readout.ch, cfg.hw.soc.adcs.readout.type
        self.res_ch, self.res_ch_type = cfg.hw.soc.dacs.readout.ch, cfg.hw.soc.dacs.readout.type
        
        # Qubit
        self.qubit_ch, self.qubit_ch_type = cfg.hw.soc.dacs.qubit.ch, cfg.hw.soc.dacs.qubit.type
        self.qubit_chs = self.qubit_ch # Syncing alias for compatibility

        # Manipulation & Storage
        self.manipulate_ch, self.manipulate_ch_type = cfg.hw.soc.dacs.manipulate_in.ch, cfg.hw.soc.dacs.manipulate_in.type
        self.storage_ch, self.storage_ch_type = cfg.hw.soc.dacs.storage_in.ch, cfg.hw.soc.dacs.storage_in.type
        self.f0g1_ch, self.f0g1_ch_type = cfg.hw.soc.dacs.qubit_sideband.ch, cfg.hw.soc.dacs.qubit_sideband.type
        
        # Flux
        self.flux_low_ch, self.flux_low_ch_type = cfg.hw.soc.dacs.flux_low.ch, cfg.hw.soc.dacs.flux_low.type
        self.flux_high_ch, self.flux_high_ch_type = cfg.hw.soc.dacs.flux_high.ch, cfg.hw.soc.dacs.flux_high.type

        # ----------- Hardware Filter/Attentuation -----------
        # Added from second version of parse_config
        for name in ['res', 'qubit', 'adc', 'manipulate']:
            src = cfg.hw.soc.dacs.readout if name == 'res' else \
                cfg.hw.soc.dacs.qubit if name == 'qubit' else \
                cfg.hw.soc.adcs.readout if name == 'adc' else \
                cfg.hw.soc.dacs.manipulate_in
            
            setattr(self, f"{name}_ftype", src.ftype)
            setattr(self, f"{name}_fc", src.fc)
            setattr(self, f"{name}_bw", src.bw)
            setattr(self, f"{name}_att", src.att)

        # ----------- Nyquist Zones -----------
        self.res_nqz = cfg.hw.soc.dacs.readout.nyquist
        self.qubit_nqz = cfg.hw.soc.dacs.qubit.nyquist
        self.manipulate_nqz = cfg.hw.soc.dacs.manipulate_in.nyquist
        self.f0g1_nqz = cfg.hw.soc.dacs.qubit_sideband.nyquist
        self.storage_nqz = cfg.hw.soc.dacs.storage_in.nyquist

        # ----------- Frequencies & Readout -----------
        self.f_ge, self.f_ef = cfg.device.qubit.f_ge, cfg.device.qubit.f_ef
        
        self.readout_length = cfg.device.readout.length
        self.readout_frequency = cfg.device.readout.frequency
        self.readout_gain = cfg.device.readout.gain
        self.readout_phase = cfg.device.readout.phase
        self.trig_offset = cfg.device.readout.trig_offset

        # ----------- Qubit Pulse Parameters (Sigma, Gain, Sigma_inc) -----------
        # Mapping pulses to simplify assignment
        pulse_list = ['pi_ge', 'hpi_ge', 'pi_ef', 'hpi_ef']
        for p in pulse_list:
            p_cfg = cfg.device.qubit.pulses[p]
            setattr(self, f"{p}_sigma", p_cfg.sigma)
            setattr(self, f"{p}_gain", p_cfg.gain)
            setattr(self, f"{p}_sigma_inc", p_cfg.sigma_inc)

        # ----------- Optional/Multiphoton Config -----------
        if 'multiphoton_config' in cfg.device:
            import yaml
            from box import Box # or AttrDict
            with open(cfg.device.multiphoton_config.file, 'r') as f:
                self.multiphoton_cfg = Box(yaml.safe_load(f))