from .general_fitting import GeneralFitting
from .fitting import *
import numpy as np
import matplotlib.pyplot as plt

class AmplitudeRabiFitting(GeneralFitting):
    def __init__(self, data, readout_per_round=2, threshold=-4.0, config=None, fitparams=None, station=None):
        super().__init__(data, readout_per_round, threshold, config, station)
        self.fitparams = fitparams
        self.results = {}

    def analyze(self, data=None, fit=True, fitparams=None, **kwargs):
        if data is None:
            data = self.data

        def get_pi_hpi_gain_from_fit(p):
            if p[2] > 180:
                p[2] = p[2] - 360
            elif p[2] < -180:
                p[2] = p[2] + 360
            if np.abs(p[2] - 90) > np.abs(p[2] + 90):
                pi_gain = (1 / 4 - p[2] / 360) / p[1]
                hpi_gain = (0 - p[2] / 360) / p[1]
            else:
                pi_gain = (3 / 4 - p[2] / 360) / p[1]
                hpi_gain = (1 / 2 - p[2] / 360) / p[1]
            return pi_gain, hpi_gain

        if fit:
            xdata = data['xpts']
            p_avgi, pCov_avgi = fitdecaysin(data['xpts'][:-1], data["avgi"][:-1], fitparams=fitparams)
            p_avgq, pCov_avgq = fitdecaysin(data['xpts'][:-1], data["avgq"][:-1], fitparams=fitparams)
            p_amps, pCov_amps = fitdecaysin(data['xpts'][:-1], data["amps"][:-1], fitparams=fitparams)
            data['fit_avgi'] = p_avgi
            data['fit_avgq'] = p_avgq
            data['fit_amps'] = p_amps
            data['fit_err_avgi'] = pCov_avgi
            data['fit_err_avgq'] = pCov_avgq
            data['fit_err_amps'] = pCov_amps

            data['pi_gain_avgi'], data['hpi_gain_avgi'] = get_pi_hpi_gain_from_fit(p_avgi)
            data['pi_gain_avgq'], data['hpi_gain_avgq'] = get_pi_hpi_gain_from_fit(p_avgq)

            self.results = {
                'fit_avgi': p_avgi,
                'fit_avgq': p_avgq,
                'fit_amps': p_amps,
                'fit_err_avgi': pCov_avgi,
                'fit_err_avgq': pCov_avgq,
                'fit_err_amps': pCov_amps,
                'pi_gain_avgi': data['pi_gain_avgi'],
                'hpi_gain_avgi': data['hpi_gain_avgi'],
                'pi_gain_avgq': data['pi_gain_avgq'],
                'hpi_gain_avgq': data['hpi_gain_avgq'],
            }
        return data

    def display(self, data=None, fit=True, fitparams=None, vline=None, save_fig=False, title_str='AmpRabi', ylim=None, **kwargs):
        if data is None:
            data = self.data

        xpts = data["xpts"][1:-1]
        xpts_fit = data["xpts"][0:-1]

        fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

        components = [('avgi', 'I [ADC units]'), ('avgq', 'Q [ADC units]')]

        for i, (key, ylabel) in enumerate(components):
            ax = axes[i]
            ax.plot(xpts, data[key][1:-1], 'o-', label="Data", alpha=0.7)
            ax.set_ylabel(ylabel)

            if ylim is not None:
                ax.set_ylim(ylim)

            if i == 0:
                ax.set_title(f"{title_str} (Pulse Length {self.cfg.expt.sigma_test})")

            if fit and f'fit_{key}' in data:
                p = data[f'fit_{key}']
                ax.plot(xpts_fit, decaysin(xpts_fit, *p), label="Fit", lw=2)

                pi_gain = data[f'pi_gain_{key}']
                hpi_gain = data[f'hpi_gain_{key}']

                ax.axvline(pi_gain, color='r', ls='--', alpha=0.6, label=rf'$\pi$ gain: {pi_gain:.3f}')
                ax.axvline(hpi_gain, color='g', ls='--', alpha=0.6, label=rf'$\pi/2$ gain: {hpi_gain:.3f}')

            if vline is not None:
                ax.axvline(vline, color='0.2', linestyle=':', label=f'Vline: {vline}')

            ax.legend(loc='best', fontsize='small')

        axes[-1].set_xlabel("Gain [DAC units]")
        plt.tight_layout()
        plt.show()

        if save_fig:
            filename = title_str.replace(' ', '_').replace(':', '') + '.png'
            self.save_plot(fig, filename=filename)