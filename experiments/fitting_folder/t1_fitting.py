from .general_fitting import GeneralFitting
from .fitting import *
import numpy as np
import matplotlib.pyplot as plt


class T1Fitting(GeneralFitting):
    def __init__(self, data, readout_per_round=None, threshold=None, config=None, fitparams=None, station=None):
        super().__init__(data, readout_per_round, threshold, config, station)
        self.fitparams = fitparams
        self.results = {}

    def analyze(self, data=None, fit=True, fitparams=None, **kwargs):
        """
        Fit T1 decay: y = yscale * exp(-x / T1) + y0

        fitparams order: [yscale, T1, y0]
        """
        if data is None:
            data = self.data

        start_idx = None
        end_idx = None

        if 'avgi' not in data.keys():
            Ilist, Qlist = self.post_select_raverager_data(data)
            data['avgi'] = Ilist[start_idx:end_idx]
            data['avgq'] = Qlist[start_idx:end_idx]

        xpts = data['xpts'][start_idx:end_idx]

        if fit:
            if fitparams is None:
                fitparams = [None, None, None, None]  # [y0, yscale, x0, decay]

            p_avgi, pCov_avgi = fitexp(xpts, data['avgi'], fitparams=fitparams)
            p_avgq, pCov_avgq = fitexp(xpts, data['avgq'], fitparams=fitparams)

            data['fit_avgi'] = p_avgi
            data['fit_avgq'] = p_avgq
            data['fit_err_avgi'] = pCov_avgi
            data['fit_err_avgq'] = pCov_avgq

            self.results = {
                'fit_avgi': p_avgi,
                'fit_avgq': p_avgq,
                'fit_err_avgi': pCov_avgi,
                'fit_err_avgq': pCov_avgq,
            }

            # Print T1 values
            # decay is index 3 → [y0, yscale, x0, decay]
            if isinstance(p_avgi, (list, np.ndarray)):
                print(f"T1 (I): {p_avgi[3]:.3f} us  ±  {np.sqrt(pCov_avgi[3][3]):.3f} us")
            if isinstance(p_avgq, (list, np.ndarray)):
                print(f"T1 (Q): {p_avgq[3]:.3f} us  ±  {np.sqrt(pCov_avgq[3][3]):.3f} us")

        return data

    def display(self, data=None, fit=True, ylim=None, vlines=None, save_fig=False, title_str='T1', **kwargs):
        """
        Display T1 decay data with optional exponential fit.

        Parameters:
        - data:       Data to display (default: self.data).
        - fit:        Whether to include fit results (default: True).
        - ylim:       Tuple specifying y-axis limits (default: None).
        - vlines:     List of x-coordinates for vertical lines (default: None).
        - save_fig:   Whether to save the figure (default: False).
        - title_str:  Title of the plot (default: 'T1').
        """
        data = self.analyze(data=data, fit=fit)

        fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True, constrained_layout=True)

        plot_params = [
            ("avgi", "fit_avgi", axes[0], "I", "blue", "cyan"),
            ("avgq", "fit_avgq", axes[1], "Q", "red",  "orange"),
        ]

        for avg_key, fit_key, ax, label, color, fit_color in plot_params:
            ax.plot(data["xpts"], data[avg_key], 'o-', label=label, color=color, alpha=0.6)

            if fit and fit_key in data and isinstance(data[fit_key], (list, np.ndarray)):
                p = data[fit_key]
                t1_val = p[3]  # decay is index 3 → [y0, yscale, x0, decay]
                ax.plot(
                    data["xpts"],
                    expfunc(data["xpts"], *p),
                    label=f"Fit {label} — T1: {t1_val:.3f} µs",
                    color=fit_color,
                    alpha=0.8,
                )

            ax.set(ylabel=f"Amplitude ({label})")
            if ylim:
                ax.set_ylim(ylim)
            if vlines:
                for vline in vlines:
                    ax.axvline(x=vline, color='gray', linestyle='--', linewidth=1)
            ax.legend()

        axes[1].set(xlabel="Wait Time [µs]")
        axes[0].set(title=title_str)

        if save_fig:
            self.save_plot(fig, filename=f"{title_str.replace(' ', '_')}.png")

        plt.show()