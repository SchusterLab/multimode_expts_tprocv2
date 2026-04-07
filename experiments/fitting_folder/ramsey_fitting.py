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
        yscale, freq, phase_deg, decay, y0, x0 
        '''
        if data is None:
            data = self.data
        try: 
            if self.cfg.expt.num_pi>0: # if there are echoes
                print('Echoes in the data')
                print(data['xpts'][:5])
                data['xpts'] *= (1 + self.cfg.expt['echoes'][1]) # multiply by the number of echoes
                print(data['xpts'][:5])
            else:
                print('No echoes in the data')
        except KeyError:
            print('No echoes in the data')
            pass

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

            if isinstance(p_avgi, (list, np.ndarray)):
                data['f_adjust_ramsey_avgi'] = sorted(
                    (self.cfg.expt.ramsey_freq - p_avgi[1], self.cfg.expt.ramsey_freq + p_avgi[1]), key=abs)
            if isinstance(p_avgq, (list, np.ndarray)):
                data['f_adjust_ramsey_avgq'] = sorted(
                    (self.cfg.expt.ramsey_freq - p_avgq[1], self.cfg.expt.ramsey_freq + p_avgq[1]), key=abs)

            self.results = {
                'fit_avgi': p_avgi,
                'fit_avgq': p_avgq,
                'fit_err_avgi': pCov_avgi,
                'fit_err_avgq': pCov_avgq,
            }
        return data

    def display(self, data=None, fit=True, ylim=None, vlines=None, save_fig=False, title_str='Ramsey', **kwargs):
        """
        Display the data with optional fitting, vertical lines, and y-axis limits.

        Parameters:
        - data: Data to display (default: self.data).
        - fit: Whether to include fit results (default: True).
        - ylim: Tuple specifying y-axis limits (default: None).
        - vlines: List of x-coordinates for vertical lines (default: None).
        - save_fig: Whether to save the figure (default: False).
        - title_str: Title of the plot (default: 'Ramsey').
        """
        data = data or self.data
        fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True, constrained_layout=True)

        # Define colors and labels for compactness
        plot_params = [
            ("avgi", "fit_avgi", axes[0], "I", "blue", "cyan"),
            ("avgq", "fit_avgq", axes[1], "Q", "red", "orange")
        ]
        
        for avg_key, fit_key, ax, label, color, fit_color in plot_params:
            # Label the T2 value as well ; 4th parameter is the decay parameter which is related to T2
            ax.plot(data["xpts"], data[avg_key], 'o-', label=label, color=color, alpha=0.6)
            if fit and fit_key in data:
                ax.plot(data["xpts"], decaysin(data["xpts"], *data[fit_key]), 
                        label=f"Fit {label} (Freq: {data[fit_key][1]:.2f}) and T2: {(data[fit_key][3]):.2f}", color=fit_color, alpha=0.6)
            ax.set(ylabel=f"Amplitude ({label})")
            if ylim:
                ax.set_ylim(ylim)
            if vlines:
                for vline in vlines:
                    ax.axvline(x=vline, color='gray', linestyle='--', linewidth=1)
            ax.legend()

        axes[1].set(xlabel="Wait Time [us]")
        axes[0].set(title=title_str)

        # Save figure if required
        if save_fig:
            self.save_plot(fig, filename=f"{title_str.replace(' ', '_')}.png")

        plt.show()