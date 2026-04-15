from .general_fitting import GeneralFitting
from .fitting import *
import numpy as np
import matplotlib.pyplot as plt


class AmplitudeRabiFitting(GeneralFitting):
    def __init__(self, data, readout_per_round=2, threshold=-4.0, config=None,
                 fitparams=None, station=None, sweep='amp'):
        """
        Parameters
        ----------
        sweep : str — 'amp' (default) or 'length'
            Controls x-axis label, title, and which result keys are populated.
        """
        super().__init__(data, readout_per_round, threshold, config, station)
        self.fitparams = fitparams
        self.results   = {}
        self.sweep     = sweep   # 'amp' or 'length'

    # ── internal helpers ──────────────────────────────────────────────────────

    def _get_pi_hpi_from_fit(self, p):
        """Extract pi and pi/2 point from a decaysin fit parameter vector."""
        phase = p[2]
        if phase > 180:
            phase -= 360
        elif phase < -180:
            phase += 360
        if np.abs(phase - 90) > np.abs(phase + 90):
            pi_val  = (1 / 4 - phase / 360) / p[1]
            hpi_val = (0     - phase / 360) / p[1]
        else:
            pi_val  = (3 / 4 - phase / 360) / p[1]
            hpi_val = (1 / 2 - phase / 360) / p[1]
        return pi_val, hpi_val

    # ── public API ────────────────────────────────────────────────────────────

    def analyze(self, data=None, fit=True, fitparams=None, **kwargs):
        """
        Fit a decaying sinusoid to amps, avgi, avgq.
        Fit params: [yscale, freq, phase_deg, decay, y0, x0]

        Populates data with:
            fit_avgi, fit_avgq, fit_amps          — fit parameter vectors
            fit_err_avgi, fit_err_avgq, fit_err_amps — covariance matrices
            pi_{amp|length}_avgi/avgq/amps        — pi point
            hpi_{amp|length}_avgi/avgq/amps       — pi/2 point
        """
        if data is None:
            data = self.data

        suffix = 'amp' if self.sweep == 'amp' else 'length'

        if fit:
            for key in ('avgi', 'avgq', 'amps'):
                p, pCov = fitdecaysin(
                    data['xpts'][:-1],
                    data[key][:-1],
                    fitparams=fitparams,
                )
                data[f'fit_{key}']     = p
                data[f'fit_err_{key}'] = pCov

                pi_val, hpi_val = self._get_pi_hpi_from_fit(p)
                data[f'pi_{suffix}_{key}']  = pi_val
                data[f'hpi_{suffix}_{key}'] = hpi_val

            self.results = {
                'fit_avgi'              : data['fit_avgi'],
                'fit_avgq'              : data['fit_avgq'],
                'fit_amps'              : data['fit_amps'],
                'fit_err_avgi'          : data['fit_err_avgi'],
                'fit_err_avgq'          : data['fit_err_avgq'],
                'fit_err_amps'          : data['fit_err_amps'],
                f'pi_{suffix}_avgi'     : data[f'pi_{suffix}_avgi'],
                f'hpi_{suffix}_avgi'    : data[f'hpi_{suffix}_avgi'],
                f'pi_{suffix}_avgq'     : data[f'pi_{suffix}_avgq'],
                f'hpi_{suffix}_avgq'    : data[f'hpi_{suffix}_avgq'],
                f'pi_{suffix}_amps'     : data[f'pi_{suffix}_amps'],
                f'hpi_{suffix}_amps'    : data[f'hpi_{suffix}_amps'],
            }
        return data

    def display(self, data=None, fit=True, fitparams=None, vlines=None,
                hlines=None, save_fig=False, title_str=None, ylim=None, **kwargs):
        if data is None:
            data = self.data

        suffix    = 'amp' if self.sweep == 'amp' else 'length'
        xlabel    = 'Gain [DAC units]' if self.sweep == 'amp' else 'Length [us]'
        pi_label  = 'π gain'   if self.sweep == 'amp' else 'π length'
        hpi_label = 'π/2 gain' if self.sweep == 'amp' else 'π/2 length'

        if title_str is None:
            if self.sweep == 'amp':
                title_str = f'AmpRabi (sigma={getattr(self.cfg.expt, "sigma_test", "?")})'
            else:
                title_str = f'LenRabi (gain={getattr(self.cfg.expt, "gain", "?")})'

        xpts     = data['xpts'][1:-1]
        xpts_fit = data['xpts'][0:-1]

        fig, axes = plt.subplots(3, 1, figsize=(6, 6), sharex=True)
        components = [
            ('amps', 'Amplitude [ADC units]'),
            ('avgi', 'I [ADC units]'),
            ('avgq', 'Q [ADC units]'),
        ]

        for i, (key, ylabel) in enumerate(components):
            ax = axes[i]
            ax.plot(xpts, data[key][1:-1], 'o-', label='Data', alpha=0.7)
            ax.set_ylabel(ylabel)

            if ylim is not None:
                ax.set_ylim(ylim)

            if i == 0:
                ax.set_title(title_str)

            if fit and f'fit_{key}' in data:
                p = data[f'fit_{key}']
                ax.plot(xpts_fit, decaysin(xpts_fit, *p), label='Fit', lw=2)

                pi_val  = data.get(f'pi_{suffix}_{key}')
                hpi_val = data.get(f'hpi_{suffix}_{key}')

                if pi_val is not None:
                    ax.axvline(pi_val,  color='r', ls='--', alpha=0.6,
                               label=rf'${pi_label}$: {pi_val:.4f}')
                if hpi_val is not None:
                    ax.axvline(hpi_val, color='g', ls='--', alpha=0.6,
                               label=rf'${hpi_label}$: {hpi_val:.4f}')

            # vlines
            if vlines is not None:
                cmap  = plt.cm.tab10
                items = list(vlines.items()) if isinstance(vlines, dict) else list(vlines)
                for vidx, item in enumerate(items):
                    val, col = item if (isinstance(item, (list, tuple)) and len(item) == 2) \
                                    else (item, cmap(vidx % 10))
                    ax.axvline(val, color=col, ls=':', alpha=0.7, lw=1.5,
                               label=f'Vline: {val}')

            # hlines
            if hlines is not None:
                cmap  = plt.cm.tab10
                items = list(hlines.items()) if isinstance(hlines, dict) else list(hlines)
                for hidx, item in enumerate(items):
                    val, col = item if (isinstance(item, (list, tuple)) and len(item) == 2) \
                                    else (item, cmap(hidx % 10))
                    ax.axhline(val, color=col, ls=':', alpha=0.7, lw=1.5,
                               label=f'Hline: {val}')

            ax.legend(loc='best', fontsize='small')

        axes[-1].set_xlabel(xlabel)
        plt.tight_layout()
        plt.show()

        if save_fig:
            filename = title_str.replace(' ', '_').replace(':', '') + '.png'
            self.save_plot(fig, filename=filename)