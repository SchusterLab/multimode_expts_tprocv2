# from experiments.fitting.ramsey_fitting import RamseyFitting
from .ramsey_fitting import RamseyFitting
from .general_fitting import GeneralFitting
from .fitting import *
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

class CavityRamseyGainSweepFitting(RamseyFitting):
    def __init__(self, data, readout_per_round=None, threshold=None, config=None, fitparams=None):
        super().__init__(data, readout_per_round, threshold, config)
        self.fitparams = fitparams
        self.results = {}

    def analyze(self, data=None, fit=True, fitparams=None, **kwargs):
        def linear_model(n, T, t0):
            return T * n + t0

        if data is None:
            data = self.data

        if not fit:
            return

        debug = kwargs.get('debug', False)
        track_peaks = kwargs.get('track_peaks', True)
        guide_window_frac = kwargs.get('guide_window_frac', 0.35)
        if debug:
            print(f"[analyze] debug=True, track_peaks={track_peaks}, guide_window_frac={guide_window_frac}")

        gain_to_alpha = self.cfg.device.manipulate.gain_to_alpha[0]
        x = data['xpts'][0]
        y = data['gain_list']
        alpha_list = gain_to_alpha * y

        g_z = data['g_avgi']
        e_z = data['e_avgi']
        g_norm = np.zeros_like(g_z)
        e_norm = np.zeros_like(e_z)
        pop_norms = [g_norm, e_norm]
        omega_vec = np.zeros((len(g_z), 2))
        t0_vec = np.zeros((len(g_z), 2))
        gain_to_plot_e = np.array([])
        gain_to_plot_g = np.array([])
        gain_to_plot = [gain_to_plot_e, gain_to_plot_g]

        x_fit = alpha_list**2

        time_peak_g = []
        time_peak_e = []
        time_peaks = [time_peak_g, time_peak_e]

        last_peaks_idx = [None, None]
        last_peak_distance_idx = [None, None]
        last_T_fit = [np.nan, np.nan]
        last_t0_fit = [np.nan, np.nan]

        for i_gain in range(len(y)):
            for i_pop, data_set in enumerate([g_z, e_z]):
                _pop = data_set[i_gain]
                if i_pop == 0:
                    _pop_norm = (_pop - np.min(_pop)) / (np.max(_pop) - np.min(_pop))
                else:
                    _pop_norm = (_pop - np.max(_pop)) / (np.min(_pop) - np.max(_pop))
                pop_norm = pop_norms[i_pop]
                pop_norm[i_gain] = _pop_norm

                signal_smooth = gaussian_filter1d(pop_norm[i_gain], sigma=1.5)
                peak_height = (np.max(signal_smooth) - np.min(signal_smooth)) * 0.4 + np.min(signal_smooth)
                peak_prominence = (np.max(signal_smooth) - np.min(signal_smooth)) * 0.2

                if i_gain == 0 or last_peak_distance_idx[i_pop] is None:
                    peak_distance = None
                else:
                    peak_distance = max(1, int(np.round(0.8 * last_peak_distance_idx[i_pop])))

                peaks, props = find_peaks(
                    signal_smooth,
                    height=peak_height,
                    prominence=peak_prominence,
                    distance=peak_distance
                )

                if len(peaks) >= 2:
                    last_peak_distance_candidate = np.mean(np.diff(peaks))
                else:
                    last_peak_distance_candidate = last_peak_distance_idx[i_pop]

                guided_peaks = None
                if track_peaks and last_peaks_idx[i_pop] is not None and last_peak_distance_idx[i_pop] is not None:
                    w = int(max(2, np.round(guide_window_frac * last_peak_distance_idx[i_pop])))
                    candidates = []
                    for pidx in last_peaks_idx[i_pop]:
                        lo = max(0, pidx - w)
                        hi = min(signal_smooth.size, pidx + w + 1)
                        if hi > lo:
                            local = signal_smooth[lo:hi]
                            off = int(np.argmax(local))
                            candidates.append(lo + off)
                    if len(candidates) >= 2:
                        guided_peaks = np.array(sorted(np.unique(candidates)))

                use_peaks = peaks
                if len(use_peaks) < 2 and guided_peaks is not None and len(guided_peaks) >= 2:
                    use_peaks = guided_peaks

                if len(use_peaks) >= 2:
                    last_peaks_idx[i_pop] = use_peaks
                    last_peak_distance_idx[i_pop] = last_peak_distance_candidate if last_peak_distance_candidate is not None else np.mean(np.diff(use_peaks))

                if len(use_peaks) >= 2:
                    n = np.arange(len(use_peaks))
                    popt, _ = curve_fit(linear_model, n, x[use_peaks])
                    T_fit, t0_fit = popt
                    omega_fit = 2 * np.pi / T_fit
                    last_T_fit[i_pop] = T_fit
                    last_t0_fit[i_pop] = t0_fit
                    time_peaks[i_pop].append(x[use_peaks])
                else:
                    popt = [np.nan, np.nan]
                    omega_fit = np.nan
                    t0_fit = np.nan

                omega_vec[i_gain, i_pop] = omega_fit
                t0_vec[i_gain, i_pop] = t0_fit

                if not np.isnan(omega_fit):
                    gain_to_plot[i_pop] = np.append(gain_to_plot[i_pop], y[i_gain])

        deltaf_e = omega_vec[:, 1] / (2 * np.pi) - self.cfg.expt.ramsey_freq
        deltaf_g = omega_vec[:, 0] / (2 * np.pi) - self.cfg.expt.ramsey_freq

        valid_g = np.isfinite(deltaf_g) & np.isfinite(x_fit)
        valid_e = np.isfinite(deltaf_e) & np.isfinite(x_fit)

        N = x_fit.size
        sel_mask_g = np.ones(N, dtype=bool)
        sel_mask_e = np.ones(N, dtype=bool)

        if 'fit_mask_g' in kwargs and kwargs['fit_mask_g'] is not None:
            m = np.asarray(kwargs['fit_mask_g'])
            if m.dtype == bool and m.size == N:
                sel_mask_g = m.copy()
        if 'fit_mask_e' in kwargs and kwargs['fit_mask_e'] is not None:
            m = np.asarray(kwargs['fit_mask_e'])
            if m.dtype == bool and m.size == N:
                sel_mask_e = m.copy()

        if 'fit_indices_g' in kwargs and kwargs['fit_indices_g'] is not None:
            idx = np.asarray(kwargs['fit_indices_g'], dtype=int)
            if idx.size > 0:
                mask = np.zeros(N, dtype=bool)
                mask[np.clip(idx, 0, N-1)] = True
                sel_mask_g = mask
        if 'fit_indices_e' in kwargs and kwargs['fit_indices_e'] is not None:
            idx = np.asarray(kwargs['fit_indices_e'], dtype=int)
            if idx.size > 0:
                mask = np.zeros(N, dtype=bool)
                mask[np.clip(idx, 0, N-1)] = True
                sel_mask_e = mask

        if 'fit_alpha2_range_g' in kwargs and kwargs['fit_alpha2_range_g'] is not None:
            lo, hi = kwargs['fit_alpha2_range_g']
            sel_mask_g &= (x_fit >= lo) & (x_fit <= hi)
        if 'fit_alpha2_range_e' in kwargs and kwargs['fit_alpha2_range_e'] is not None:
            lo, hi = kwargs['fit_alpha2_range_e']
            sel_mask_e &= (x_fit >= lo) & (x_fit <= hi)

        popt_g = [np.nan, np.nan]
        popt_e = [np.nan, np.nan]
        Kerr = chi = chi2 = detuning_g = np.nan
        Kerr_err = chi_err = chi2_err = detuning_g_err = np.nan

        mask_g = valid_g & sel_mask_g
        if np.sum(mask_g) >= 2:
            popt_g, pcov_g = curve_fit(lambda n, T, t0: linear_model(n, T, t0), x_fit[mask_g], deltaf_g[mask_g])
            detuning_g = popt_g[1]
            Kerr = -popt_g[0]
            perr_g = np.sqrt(np.diag(pcov_g))
            Kerr_err = perr_g[0]
            detuning_g_err = perr_g[1]

            if self.cfg.expt.do_g_and_e:
                mask_e = valid_e & sel_mask_e
                if np.sum(mask_e) >= 2:
                    popt_e, pcov_e = curve_fit(lambda n, T, t0: linear_model(n, T, t0), x_fit[mask_e], deltaf_e[mask_e])
                else:
                    popt_e = [np.nan, np.nan]
                    pcov_e = np.array([[np.nan, np.nan],[np.nan, np.nan]])
                perr_e = np.sqrt(np.diag(pcov_e))
                chi = -(popt_e[1] - detuning_g)
                chi2 = -0.5 * (popt_e[0] + Kerr)
                chi_err = np.sqrt(perr_e[1]**2 + detuning_g_err**2)
                chi2_err = 0.5 * np.sqrt(perr_e[0]**2 + perr_g[0]**2)

        self.results['omega'] = omega_vec
        self.results['t0'] = t0_vec

        data['g_omega'] = omega_vec[:, 0]
        data['g_t0'] = t0_vec[:, 0]
        data['e_omega'] = omega_vec[:, 1]
        data['e_t0'] = t0_vec[:, 1]
        data['g_norm'] = g_norm
        data['e_norm'] = e_norm
        data['alpha_list'] = alpha_list
        data['time_peak_g'] = time_peaks[0]
        data['time_peak_e'] = time_peaks[1]
        data['detuning_g'] = detuning_g
        data['detuning_g_err'] = detuning_g_err
        data['Kerr'] = Kerr
        data['Kerr_err'] = Kerr_err
        if self.cfg.expt.do_g_and_e:
            data['chi'] = chi
            data['chi2'] = chi2
            data['chi_err'] = chi_err
            data['chi2_err'] = chi2_err
        data['gain_to_plot_g'] = gain_to_plot[0]
        data['gain_to_plot_e'] = gain_to_plot[1]
        data['fit_mask_g'] = (valid_g & sel_mask_g)
        data['fit_mask_e'] = (valid_e & sel_mask_e)
        data['fit_used_indices_g'] = np.where(valid_g & sel_mask_g)[0]
        if self.cfg.expt.do_g_and_e:
            data['fit_used_indices_e'] = np.where(valid_e & sel_mask_e)[0]
        else:
            data['fit_used_indices_e'] = np.array([], dtype=int)

    def display(self, data=None, fit=True, ylim=None, vlines=None, save_fig=False, title_str='CavityRamseyGainSweep', **kwargs):
        data = data or self.data
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(data["xpts"], data["avgi"], 'o-', label="I")
        ax.plot(data["xpts"], data["avgq"], 'o-', label="Q")

        if ylim:
            ax.set_ylim(ylim)

        if vlines:
            for vline in vlines:
                ax.axvline(x=vline, color='gray', linestyle='--', linewidth=1)

        if fit and "fit_avgi" in data:
            for key, label in [("fit_avgi", "Fit I"), ("fit_avgq", "Fit Q")]:
                if key in data:
                    ax.plot(data["xpts"], fitter.decaysin(data["xpts"], *data[key]), label=label)

        ax.set(title=title_str, xlabel="Wait Time [us]", ylabel="Amplitude")
        ax.legend()
        plt.tight_layout()

        if save_fig:
            self.save_plot(fig, filename=f"{title_str.replace(' ', '_')}.png")

        plt.show()