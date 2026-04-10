import numpy as np
import matplotlib.pyplot as plt

from qick import *
from qick.helpers import gauss
from slab import Experiment, dsfit, AttrDict

from ..general.MM_program import MMProgram
from ..general.MM_experiment import MMExperiment
from ..fitting_folder import fitting as fitter


class ToFProgram(MMProgram):
    def __init__(self, soccfg, final_delay, cfg):
        self.cfg = AttrDict(cfg)
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg)

    def _initialize(self, cfg):
        self.cfg = AttrDict(cfg)

        freq   = self.cfg.expt.frequency
        gain   = self.cfg.expt.gain
        length = self.cfg.expt.length

        # Manually declare and configure readout — do NOT call initialize_readout()
        # because that pulls frequency/gain/length from cfg.device.readout
        self.declare_gen(ch=self.res_ch, nqz=self.res_nqz)

        self.add_pulse(
            ch=self.res_ch,
            name="readout_pulse",
            style="const",
            ro_ch=self.adc_ch,
            freq=freq,
            phase=0,
            gain=gain,
            length=length,
        )

        if self.adc_ch_type == "dyn":
            self.declare_readout(ch=self.adc_ch, length=length)
            self.add_readoutconfig(
                ch=self.adc_ch, name="readout", freq=freq, gen_ch=self.res_ch
            )
        elif self.adc_ch_type == "std":
            self.declare_readout(
                ch=self.adc_ch,
                length=length,
                freq=freq,
                phase=0,
                gen_ch=self.res_ch,
            )

    def _body(self, cfg):
        cfg = AttrDict(self.cfg)

        if self.adc_ch_type == 'dyn':
            self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)

        # Optional: excite qubit to |e⟩ before readout
        if cfg.expt.get('check_e', False):
            self.pulse(ch=self.qubit_ch, name="pi_qubit_ge", t=0)
            self.delay_auto(t=0.01, tag="wait_e")

        # Fire readout pulse
        self.pulse(ch=self.res_ch, name="readout_pulse", t=0.0)

        # Trigger ADC — trig_offset = 0 so we capture the full waveform
        self.trigger(
            ros=[self.adc_ch],
            pins=[0],
            t=0,
        )


class ToFExperiment(MMExperiment):
    """
    Time of Flight Calibration Experiment.

    Run this when wiring changes. Measures the delay between sending the
    readout pulse and receiving the signal, so we know where to set
    cfg.device.readout.trig_offset.

    Experimental Config:
    expt = dict(
        reps:           number of averages (use high value, e.g. 10000)
        rounds:         software averages (default 1)
        check_e:        if True, apply ge pi pulse before readout (default False)
        final_delay:    delay after sequence [us]
    )
    Readout parameters (frequency, gain, length) are taken from cfg.device.readout.
    """

    def __init__(
        self,
        cfg_dict,
        qi=0,
        go=True,
        params={},
        prefix=None,
        progress=True,
        display=True,
        print_=False,
    ):
        """
        Args:
            cfg_dict:   QICK config dictionary
            qi:         Qubit index
            go:         Whether to run immediately
            params:     Override defaults
            prefix:     Filename prefix
            progress:   Show progress bar
            display:    Show plot after acquisition
            print_:     Print config and exit
        """
        if prefix is None:
            prefix = f"tof_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress)

        if go:
            self.acquire(progress=progress)
            if display:
                self.display()

    def acquire(self, progress=False):
        """
        Acquire decimated I/Q data for ToF calibration.

        Returns:
            dict with keys: xpts (time axis [us]), avgi, avgq, amps, phases
        """
        # Use trig_offset=0 so we see the full waveform and can identify the signal arrival
        final_delay = self.cfg.device.readout.final_delay

        prog = ToFProgram(
            soccfg=self.soccfg,
            final_delay=final_delay,
            cfg=self.cfg,
        )

        # Decimated acquisition — returns raw time-domain I/Q trace
        iq_list = prog.acquire_decimated(
            self.im[self.cfg.aliases.soc],
            rounds=self.cfg.expt.get('rounds', 1),
            progress=progress,
        )

        # Time axis
        t = prog.get_time_axis(ro_index=0)

        i = iq_list[0][:, 0]
        q = iq_list[0][:, 1]
        amps = np.abs(i + 1j * q)
        phases = np.angle(i + 1j * q)

        data = {
            "xpts": t,
            "avgi": i,
            "avgq": q,
            "amps": amps,
            "phases": phases,
        }

        for k in data:
            data[k] = np.array(data[k])

        self.data = data
        return data

    def analyze(self, data=None, fit=True, **kwargs):
        """
        Fit the ring-up of the amplitude to extract the decay constant τ.
        Ring-up model: A * (1 - exp(-(t - t0) / τ)) + offset
        which is equivalent to expfunc with negative yscale.

        Fit is done on amplitude only, and only on the rising portion
        (from trig_offset onward).
        """
        if data is None:
            data = self.data

        if not fit:
            return data

        trig_offset = self.cfg.device.readout.trig_offset

        # Only fit from trig_offset onward (the ring-up region)
        mask = data['xpts'] >= trig_offset
        xfit = data['xpts'][mask]
        yfit = data['amps'][mask]

        if len(xfit) < 4:
            print('Warning: not enough points after trig_offset to fit ring-up')
            return data

        # expfunc: y0 + yscale * exp(-(x - x0) / decay)
        # For ring-up: y0 = steady state, yscale = negative (starts low, rises)
        #              x0 = trig_offset, decay = τ
        fitparams = [
            yfit[-1],           # y0: steady state amplitude
            yfit[0] - yfit[-1], # yscale: negative for ring-up
            xfit[0],            # x0: start of ring-up
            (xfit[-1] - xfit[0]) / 5,  # decay: initial guess for τ
        ]

        pOpt, pCov = fitter.fitexp(xfit, yfit, fitparams=fitparams)
        data['fit_ringup'] = pOpt
        data['fit_ringup_err'] = pCov

        if isinstance(pOpt, (list, np.ndarray)):
            tau = pOpt[3]
            tau_err = np.sqrt(pCov[3][3]) if not np.isinf(pCov[3][3]) else np.nan
            kappa = 1 / tau  # [MHz] since t is in µs
            print(f"Ring-up τ  = {tau:.4f} µs  ±  {tau_err:.4f} µs")
            print(f"κ from τ   = {kappa:.4f} MHz")
            data['tau_ringup'] = tau
            data['kappa_ringup'] = kappa

        return data

    def display(self, data=None, adc_trig_offset=None, save_fig=False, title_str='Time of Flight', **kwargs):
        """
        Plot I, Q, and amplitude vs time.
        Draw a vertical line at adc_trig_offset so you can see where
        the trigger is currently set relative to the signal arrival.

        Parameters:
        - adc_trig_offset: current trig_offset value to mark on plot [us].
                           If None, uses cfg.device.readout.trig_offset.
        - save_fig:        whether to save the figure.
        - title_str:       plot title.
        """
        if data is None:
            data = self.data

        # Run fit first
        data = self.analyze(data=data, fit=True)

        if adc_trig_offset is None:
            adc_trig_offset = self.cfg.device.readout.trig_offset

        fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True, constrained_layout=True)

        plot_params = [
            ("avgi",   axes[0], "I",         "blue"),
            ("avgq",   axes[1], "Q",         "red"),
            ("amps",   axes[2], "Amplitude", "green"),
        ]

        for key, ax, label, color in plot_params:
            ax.plot(data["xpts"], data[key], color=color, alpha=0.8, label=label)
            # Overlay ring-up fit on amplitude panel
            if key == "amps" and 'fit_ringup' in data and isinstance(data['fit_ringup'], (list, np.ndarray)):
                p = data['fit_ringup']
                tau = data.get('tau_ringup', p[3])
                kappa = data.get('kappa_ringup', 1/p[3])
                xfit = data['xpts'][data['xpts'] >= adc_trig_offset]
                ax.plot(xfit, fitter.expfunc(xfit, *p), color='black', linestyle='--', linewidth=1.5,
                        label=f"Fit: τ={tau:.3f} µs, κ={kappa:.3f} MHz")
            ax.axvline(x=adc_trig_offset, color='k', linestyle='--', linewidth=1.2,
                       label=f"trig_offset = {adc_trig_offset:.3f} µs")
            ax.set_ylabel(f"{label} (ADC units)")
            ax.legend(fontsize=8)

        axes[2].set_xlabel("Time [µs]")
        axes[0].set_title(f"{title_str} — set trig_offset to where signal rises")

        if save_fig:
            self.save_plot(fig, filename=f"{title_str.replace(' ', '_')}.png")

        plt.show()