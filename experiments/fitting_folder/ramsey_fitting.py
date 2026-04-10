from .general_fitting import GeneralFitting
from .fitting import *
import numpy as np
import matplotlib.pyplot as plt


class RamseyFitting(GeneralFitting):
    def __init__(self, data, readout_per_round=None, threshold=None, config=None, fitparams=None, station=None):
        super().__init__(data, readout_per_round, threshold, config, station)
        self.fitparams = fitparams
        self.results = {}

    def analyze(self, data=None, fit=True, fitparams=None, **kwargs):
        '''
        Fit Ramsey fringes and compute frequency correction.

        Fit params: [yscale, freq, phase_deg, decay, y0, x0]

        Frequency correction:
            f_correction = ramsey_freq - f_fit
            new_qubit_freq = current_qubit_freq + f_correction

        The two possible corrections (ramsey_freq ± f_fit) are sorted by
        absolute value — the smallest is most likely correct.
        '''
        if data is None:
            data = self.data

        try:
            if self.cfg.expt.num_pi > 0:
                print('Echoes in the data')
                data['xpts'] *= (1 + self.cfg.expt['echoes'][1])
            else:
                print('No echoes in the data')
        except KeyError:
            print('No echoes in the data')

        start_idx = None
        end_idx = None
        if 'avgi' not in data.keys():
            Ilist, Qlist = self.post_select_raverager_data(data)
            data['avgi'] = Ilist[start_idx:end_idx]
            data['avgq'] = Qlist[start_idx:end_idx]
        xpts = data['xpts'][start_idx:end_idx]

        if fit:
            if fitparams is None:
                fitparams = [None, self.cfg.expt.ramsey_freq, None, None, None, None]

            p_avgi, pCov_avgi = fitdecaysin(xpts, data["avgi"], fitparams=fitparams)
            p_avgq, pCov_avgq = fitdecaysin(xpts, data["avgq"], fitparams=fitparams)

            data['fit_avgi'] = p_avgi
            data['fit_avgq'] = p_avgq
            data['fit_err_avgi'] = pCov_avgi
            data['fit_err_avgq'] = pCov_avgq

            ramsey_freq = self.cfg.expt.ramsey_freq
            current_freq = self.cfg.expt.freq  # actual drive frequency used in experiment

            # --- Frequency correction ---
            # Two candidate corrections: ramsey_freq ± f_fit
            # Sorted by abs value — smallest is most likely the right one
            if isinstance(p_avgi, (list, np.ndarray)):
                f_fit_i = p_avgi[1]
                # corrections_i = sorted(
                #     (current_freq - f_fit_i, current_freq + f_fit_i), key=abs
                # )
                # data['f_adjust_ramsey_avgi'] = corrections_i
                # data['f_corrected_avgi'] = corrections_i[0]
                print(f"[I]  f_fit={f_fit_i:.4f} MHz  |  "
                      f"candidates: {current_freq + ramsey_freq - f_fit_i:.4f}, {current_freq + ramsey_freq + f_fit_i:.4f} MHz  |  "
                    #   f"suggested new f_ge={:.4f} MHz  |  "
                      f"correction={ramsey_freq - f_fit_i:+.4f}, {ramsey_freq + f_fit_i:+.4f}  MHz")

            if isinstance(p_avgq, (list, np.ndarray)):
                f_fit_q = p_avgq[1]
                # corrections_q = sorted(
                #     (current_freq - f_fit_q, current_freq + f_fit_q), key=abs
                # )
                # data['f_adjust_ramsey_avgq'] = corrections_q
                # data['f_corrected_avgq'] = corrections_q[0]
                print(f"[Q]  f_fit={f_fit_q:.4f} MHz  |  "
                      f"candidates: {current_freq + ramsey_freq - f_fit_q:.4f}, {current_freq + ramsey_freq + f_fit_q:.4f} MHz  |  "
                    #   f"suggested new f_ge={data['f_corrected_avgq']:.4f} MHz  |  "
                        f"correction={ramsey_freq - f_fit_q:+.4f}, {ramsey_freq + f_fit_q:+.4f}  MHz")

            self.results = {
                'fit_avgi': p_avgi,
                'fit_avgq': p_avgq,
                'fit_err_avgi': pCov_avgi,
                'fit_err_avgq': pCov_avgq,
                'f_corrected_avgi': data.get('f_corrected_avgi'),
                'f_corrected_avgq': data.get('f_corrected_avgq'),
            }

        return data

    def display(self, data=None, fit=True, ylim=None, vlines=None, save_fig=False, title_str='Ramsey', **kwargs):
        """
        Display Ramsey fringes with fit and frequency correction annotation.
        """
        data = data or self.data
        fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True, constrained_layout=True)

        plot_params = [
            ("avgi", "fit_avgi", "f_corrected_avgi", axes[0], "I", "blue", "cyan"),
            ("avgq", "fit_avgq", "f_corrected_avgq", axes[1], "Q", "red",  "orange"),
        ]

        for avg_key, fit_key, fcorr_key, ax, label, color, fit_color in plot_params:
            ax.plot(data["xpts"], data[avg_key], 'o-', label=label, color=color, alpha=0.6)

            if fit and fit_key in data and isinstance(data[fit_key], (list, np.ndarray)):
                p = data[fit_key]
                f_corr = data.get(fcorr_key, None)
                fit_label = (
                    f"Fit {label} | f_fit={p[1]:.3f} MHz | T2={p[3]:.3f} µs"
                    + (f" | new f_ge={f_corr:.4f} MHz" if f_corr is not None else "")
                )
                ax.plot(data["xpts"], decaysin(data["xpts"], *p),
                        label=fit_label, color=fit_color, alpha=0.8)

            ax.set(ylabel=f"Amplitude ({label})")
            if ylim:
                ax.set_ylim(ylim)
            if vlines:
                for vline in vlines:
                    ax.axvline(x=vline, color='gray', linestyle='--', linewidth=1)
            ax.legend(fontsize=8)

        axes[1].set(xlabel="Wait Time [µs]")
        axes[0].set(title=title_str)

        if save_fig:
            self.save_plot(fig, filename=f"{title_str.replace(' ', '_')}.png")

        plt.show()