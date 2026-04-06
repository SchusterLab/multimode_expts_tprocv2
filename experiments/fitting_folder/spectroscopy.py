from .general_fitting import GeneralFitting
from .fitting import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import factorial

def _gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def photon_resolved_model(x, offset, scale, mu, chi, sigma, nbar, Nmax=6):
    """
    Calculate a photon-resolved spectroscopy model with thermal photon distribution.

    This function models the spectral response of a system where the signal consists
    of multiple Gaussian peaks corresponding to different photon number states. Each
    peak is weighted by a Poisson distribution representing the thermal photon population.

    The model is described by:
        model(x) = offset + scale * Σ(n=0 to Nmax) [P(n|nbar) * G(x; μ + n*χ, σ)]

    where:
        - P(n|nbar) = exp(-nbar) * nbar^n / n! is the Poisson weight for n photons
        - G(x; μ_n, σ) is a Gaussian with mean μ_n = μ + n*χ and std dev σ

    Parameters
    ----------
    x : array_like
        Input x-values (frequencies, wavelengths, etc.) at which to evaluate the model.
    offset : float
        Vertical offset of the model.
    scale : float
        Scaling factor for the model amplitude.
    mu : float
        Center position of the n=0 Gaussian peak (μ₀).
    chi : float
        Energy shift per photon (Δμ per photon state, χ).
    sigma : float
        Standard deviation (linewidth) of each Gaussian component (σ).
    nbar : float
        Mean number of thermal photons (average photon population, n̄ ≥ 0).
    Nmax : int, optional
        Maximum photon number to include in the sum (default: 6).
        Higher values give more accurate results but require more computation.

    Returns
    -------
    model : ndarray
        Calculated model values at each x-position, same shape as input x.

    Notes
    -----
    The photon-resolved model is commonly used in spectroscopy of systems with
    significant thermal photon populations or in cavity quantum electrodynamics.
    """
    n = np.arange(0, Nmax + 1)
    weights = np.exp(-nbar) * (nbar ** n) / factorial(n)
    model = np.zeros_like(x, dtype=float)
    for nn, w in zip(n, weights):
        model += w * _gaussian(x, mu + nn * chi, sigma)
    return offset + scale * model

class SpectroscopyFitting(GeneralFitting):
    def __init__(self, data, signs=[1, 1, 1], config=None, station=None):
        super().__init__(data, readout_per_round=1, threshold=-4.0, config=config, station=station)
        self.signs = signs


    def analyze(self, data_list=None, fit=True, fitparams=None):
        """
        Analyze spectroscopy data for one or multiple datasets.
        
        Parameters:
        -----------
        data_list : list of dict, optional
            List of data dictionaries to analyze. If None, uses self.data only.
        fit : bool
            Whether to perform fitting on the data.
        
        Returns:
        --------
        list of dict
            List of modified data dictionaries with fit results added.
        """
        if data_list is None:
            data_list = [self.data]
        
        modified_data_list = []
        
        for data in data_list:
            xdata = data['xpts'][1:-1]
            if fit:
                # print(dir(fitter))
                data['fit_amps'], data['fit_err_amps'] = fitlor(xdata, self.signs[0] * data['amps'][1:-1], fitparams=fitparams)
                data['fit_avgi'], data['fit_err_avgi'] = fitlor(xdata, self.signs[1] * data['avgi'][1:-1], fitparams=fitparams)
                data['fit_avgq'], data['fit_err_avgq'] = fitlor(xdata, self.signs[2] * data['avgq'][1:-1], fitparams=fitparams)
            
            modified_data_list.append(data)
        
        return modified_data_list

    def display(self, title='Spectroscopy Fitting', vlines=None, hlines=None, fit=True, data_list=None):
        """
        Display spectroscopy data with optional fitting.
        
        Parameters:
        -----------
        title : str
            Title for the plot
        vlines : list, optional
            List of x-values to draw vertical lines
        hlines : list, optional
            List of y-values to draw horizontal lines
        fit : bool
            Whether to overlay fitted curves
        data_list : list of dict, optional
            List of data dictionaries to plot on the same axes. If None, uses self.data only.
        """
        print('new display function for spectroscopy')
        
        # Handle single or multiple data
        if data_list is None:
            data_list = [self.data]
        
        xpts_list = [data['xpts'][1:-1] for data in data_list]
        keys = ["amps", "avgi", "avgq"]
        ylabels = ["Amplitude [ADC]", "I [ADC]", "Q [ADC]"]
        colors = plt.cm.tab10(range(len(data_list)))
        
        fig, axes = plt.subplots(3, 1, figsize=(5,5), sharex=True)
        axes[0].set_title(title)

        for i, (ax, key) in enumerate(zip(axes, keys)):
            # Plot each dataset
            for idx, (data, xpts, color) in enumerate(zip(data_list, xpts_list, colors)):
                y_data = data[key][1:-1]
                label_data = f'Data {idx+1}' if len(data_list) > 1 else 'Data'
                ax.plot(xpts, y_data, 'o-', label=label_data, alpha=0.7, color=color)

                # Handle Fitting
                fit_key = f'fit_{key}'
                if fit and fit_key in data:
                    p = data[fit_key]
                    kappa = 2 * p[3]
                    y_fit = self.signs[i] * lorfunc(xpts, *p)
                    label_fit = f'Fit {idx+1}: κ={kappa:.3f} MHz, f={p[2]:.3f} MHz' if len(data_list) > 1 else f'κ={kappa:.3f} MHz, f={p[2]:.3f} MHz'
                    ax.plot(xpts, y_fit, lw=2, label=label_fit, color=color, linestyle='--')
                    print(f'Data {idx+1} - Found peak in {key} at {p[2]:.3f} MHz, HWHM {p[3]:.3f}')

            ax.set_ylabel(ylabels[i])
            
            # Handle Vertical Lines (only plot once, not for each dataset)
            if vlines:
                for vline in vlines:
                    ax.axvline(vline, c='k', ls=':', alpha=0.5, linewidth=1, label = 'vline: '+str(vline))
            # Handle Horizontal Lines (only plot once, not for each dataset)
            if hlines:
                for hline in hlines:
                    ax.axhline(hline, c='k', ls=':', alpha=0.5, linewidth=1, label = 'hline: '+str(hline))
            ax.legend(loc='best', fontsize='small', framealpha=0.7)

        axes[-1].set_xlabel("Pulse Frequency (MHz)")
        plt.tight_layout()
        plt.show()
        
        # Save figure
        filename = title.replace(' ', '_').replace(':', '') + '.png'
        self.save_plot(fig, filename=filename)

    def analyze_photon_resolved(self, data_list=None, fit=True, Nmax=4, fitparams=None):
        """
        Analyze using photon-resolved Poisson-weighted multi-Gaussian fit.
        Adds fit_photon_amps, fit_photon_avgi, fit_photon_avgq to each data dict.

        Parameters
        ----------
        data_list : list of dict, optional
            List of data dictionaries to analyze. If None, uses self.data only.
        fit : bool
            Whether to perform fitting on the data.
        Nmax : int
            Maximum photon number to sum in the model.
        fitparams : dict or None
            Optional dictionary of initial guesses for each key (amps, avgi, avgq). Example:
                fitparams = {
                    'amps': [offset, scale, mu, chi, sigma, nbar],
                    'avgi': [...],
                    'avgq': [...]
                }
            If not provided or a key is missing, a default guess will be used for that key.
        """
        if data_list is None:
            data_list = [self.data]
        if fitparams is None:
            fitparams = {}
        modified_data_list = []
        for data in data_list:
            xdata = data['xpts'][1:-1]
            if fit:
                for key in ['amps', 'avgi', 'avgq']:
                    y = data[key][1:-1]
                    # Use user guess if provided, else default
                    p0 = fitparams.get(key, None)
                    if p0 is None:
                        offset0 = np.min(y)
                        scale0 = np.max(y) - offset0 if np.max(y) > offset0 else 1.0
                        mu0 = xdata[np.argmax(y)]
                        chi0 = (xdata[-1] - xdata[0]) / 10.0 if len(xdata) > 1 else 0.1
                        sigma0 = max((xdata[1] - xdata[0]) * 2.0, 0.01) if len(xdata) > 1 else 0.1
                        nbar0 = 1.0
                        p0 = [offset0, scale0, mu0, chi0, sigma0, nbar0]
                    lower = [-np.inf, 0.0, xdata[0] - (xdata[-1] - xdata[0]), -np.inf, 1e-6, 0.0]
                    upper = [np.inf, np.inf, xdata[-1] + (xdata[-1] - xdata[0]), np.inf, (xdata[-1] - xdata[0]) * 2 + 10, max(10.0, 5.0)]
                    try:
                        popt, pcov = curve_fit(lambda xx, offset, scale, mu, chi, sigma, nbar: photon_resolved_model(xx, offset, scale, mu, chi, sigma, nbar, Nmax),
                                               xdata, y, p0=p0, bounds=(lower, upper), maxfev=20000)
                    except Exception as e:
                        popt, pcov = None, None
                    data[f'fit_photon_{key}'] = popt
                    data[f'fit_photon_err_{key}'] = pcov
            modified_data_list.append(data)
        return modified_data_list

    def display_photon_resolved(self, title='Photon-Resolved Spectroscopy', vlines=None, fit=True, data_list=None, Nmax=6, fitparams=None):
        """
        Display data and photon-resolved fit (if present). Optionally overlay a Poisson photon-resolved model using user-supplied fitparams.

        Parameters
        ----------
        title : str
            Title for the plot
        vlines : list, optional
            List of x-values to draw vertical lines
        fit : bool
            Whether to overlay fitted curves
        data_list : list of dict, optional
            List of data dictionaries to plot on the same axes. If None, uses self.data only.
        Nmax : int
            Maximum photon number to sum in the model.
        fitparams : dict or None
            Optional dictionary of model parameters to overlay a Poisson model for each key (amps, avgi, avgq). Example:
                fitparams = {
                    'amps': [offset, scale, mu, chi, sigma, nbar],
                    'avgi': [...],
                    'avgq': [...]
                }
            If provided, overlays the corresponding model curve(s) on the plot.
        """
        if data_list is None:
            data_list = [self.data]
        if fitparams is None:
            fitparams = {}
        xpts_list = [data['xpts'][1:-1] for data in data_list]
        keys = ["amps", "avgi", "avgq"]
        ylabels = ["Amplitude [ADC]", "I [ADC]", "Q [ADC]"]
        colors = plt.cm.tab10(range(len(data_list)))
        fig, axes = plt.subplots(3, 1, figsize=(6, 6), sharex=True)
        axes[0].set_title(title)
        for i, (ax, key) in enumerate(zip(axes, keys)):
            for idx, (data, xpts, color) in enumerate(zip(data_list, xpts_list, colors)):
                y_data = data[key][1:-1]
                label_data = f'Data {idx+1}' if len(data_list) > 1 else 'Data'
                ax.plot(xpts, y_data, 'o-', label=label_data, alpha=0.7, color=color)
                pr_key = f'fit_photon_{key}'
                if fit and pr_key in data and data[pr_key] is not None:
                    popt = data[pr_key]
                    if popt is not None:
                        y_pr = photon_resolved_model(xpts, *popt, Nmax=Nmax)
                        label_pr = f'Photon fit: μ={popt[2]:.3f} MHz, χ={popt[3]:.3f} MHz, σ={popt[4]:.3f} MHz, n̄={popt[5]:.3f}'
                        # Make fitted photon-resolved curve more visible: red and thicker
                        ax.plot(xpts, y_pr, lw=3, label=label_pr, color='red', linestyle='-.', zorder=5)
            # Overlay user-supplied Poisson model if fitparams provided
            if key in fitparams and fitparams[key] is not None:
                # Use the first xpts (assume all data sets have same x grid)
                x_overlay = xpts_list[0]
                y_overlay = photon_resolved_model(x_overlay, *fitparams[key], Nmax=Nmax)
                label_overlay = f'User Poisson: μ={fitparams[key][2]:.3f}, χ={fitparams[key][3]:.3f}, σ={fitparams[key][4]:.3f}, n̄={fitparams[key][5]:.3f}'
                ax.plot(x_overlay, y_overlay, lw=2, label=label_overlay, color='tab:red', linestyle=':')
            ax.set_ylabel(ylabels[i])
            if vlines:
                for vline in vlines:
                    ax.axvline(vline, c='k', ls=':', alpha=0.5, linewidth=1)
            ax.legend(loc='best', fontsize='small', framealpha=0.7)
        axes[-1].set_xlabel("Pulse Frequency (MHz)")
        plt.tight_layout()
        plt.show()
        filename = title.replace(' ', '_').replace(':', '') + '.png'
        self.save_plot(fig, filename=filename)
        
    # ok so the above analysis and display is for a single sweep variable: freq
    # now we need to generalize to multiple sweep variables, lets start with two : freq and gain
    # here's how the data dict will look like:['amps', 'avgi', 'avgq', 'phases', 'start_time', 'xpts_freq', 'xpts_readout_probe_gain']
    # note that if len(xpts_freq) = N_freq and len(xpts_readout_probe_gain) = N_gain, then shape of 
    # amps, avgi, avgq will be (N_freq, N_gain)
    # so for each gain, i want to run photon resolved fitting over freq axis
    # and then i want to display the n_bar vs gain plot for each of amps, avgi, avgq
    # you are free to make changes to above function, but ideally i would make new function for sweep stuff, but do use the old ones to avoid 
    # code duplication
    def analyze_photon_resolved_gain_sweep(self, data_list=None, fit=True, Nmax=4, fitparams=None, freq_axis=0):
        """
        Analyze 2D data (e.g., frequency vs gain) using photon-resolved fitting along one axis.
        
        Parameters
        ----------
        data_list : list of dict, optional
            List of data dictionaries with 2D arrays. If None, uses self.data only.
        fit : bool
            Whether to perform fitting on the data.
        Nmax : int
            Maximum photon number to sum in the model.
        fitparams : dict or None
            Optional initial guesses per key.
        freq_axis : int
            Axis along which to fit (0 for frequency, 1 for gain).
        
        Returns
        -------
        list of dict
            Modified data dictionaries with fit results added.
        """
        if data_list is None:
            data_list = [self.data]
        if fitparams is None:
            fitparams = {}
        
        modified_data_list = []
        for data in data_list:
            # Extract xpts based on which axis we're fitting
            if freq_axis == 0:
                xdata = data['xpts_freq'][1:-1]
                n_slices = data['amps'].shape[1] if data['amps'].ndim > 1 else 1
            else:
                xdata = data['xpts_readout_probe_gain'][1:-1]
                n_slices = data['amps'].shape[0] if data['amps'].ndim > 1 else 1
            
            # Initialize storage for nbar values
            nbar_dict = {key: [] for key in ['amps', 'avgi', 'avgq']}
            
            if fit:
                for key in ['amps', 'avgi', 'avgq']:
                    popt_list = []
                    
                    for slice_idx in range(n_slices):
                        if freq_axis == 0:
                            y = data[key][1:-1, slice_idx]
                        else:
                            y = data[key][slice_idx, 1:-1]
                        
                        # Default initial guess
                        p0 = fitparams.get(key, None)
                        if p0 is None:
                            offset0 = np.min(y)
                            scale0 = np.max(y) - offset0 if np.max(y) > offset0 else 1.0
                            mu0 = xdata[np.argmax(y)]
                            chi0 = (xdata[-1] - xdata[0]) / 10.0 if len(xdata) > 1 else 0.1
                            sigma0 = max((xdata[1] - xdata[0]) * 2.0, 0.01) if len(xdata) > 1 else 0.1
                            nbar0 = 1.0
                            p0 = [offset0, scale0, mu0, chi0, sigma0, nbar0]
                        
                        lower = [-np.inf, 0.0, xdata[0] - (xdata[-1] - xdata[0]), -np.inf, 1e-6, 0.0]
                        upper = [np.inf, np.inf, xdata[-1] + (xdata[-1] - xdata[0]), np.inf, (xdata[-1] - xdata[0]) * 2 + 10, 10.0]
                        
                        try:
                            popt, pcov = curve_fit(
                                lambda xx, offset, scale, mu, chi, sigma, nbar: photon_resolved_model(xx, offset, scale, mu, chi, sigma, nbar, Nmax),
                                xdata, y, p0=p0, bounds=(lower, upper), maxfev=20000
                            )
                            popt_list.append(popt)
                            nbar_dict[key].append(popt[5])  # Extract nbar
                        except Exception as e:
                            popt_list.append(None)
                            nbar_dict[key].append(np.nan)
                    
                    data[f'fit_photon_{key}_list'] = popt_list
                    data[f'nbar_{key}'] = np.array(nbar_dict[key])
            
            modified_data_list.append(data)
        
        return modified_data_list

    def display_photon_resolved_gain_sweep(self, title='2D Photon-Resolved Spectroscopy', fit=True, data_list=None, Nmax=6):
        """
        Display 2D data with nbar extracted from photon-resolved fits across gain sweep.
        
        Parameters
        ----------
        title : str
            Title for the plot
        fit : bool
            Whether to overlay fitted curves
        data_list : list of dict, optional
            List of data dictionaries to plot. If None, uses self.data only.
        Nmax : int
            Maximum photon number to sum in the model.
        """
        if data_list is None:
            data_list = [self.data]
        
        keys = ["amps", "avgi", "avgq"]
        ylabels = ["Amplitude [ADC]", "I [ADC]", "Q [ADC]"]
        colors = plt.cm.tab10(range(len(data_list)))
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for i, (ax, key) in enumerate(zip(axes, keys)):
            for idx, (data, color) in enumerate(zip(data_list, colors)):
                if f'nbar_{key}' in data:
                    x_gain = data['xpts_readout_probe_gain']
                    nbar_vals = data[f'nbar_{key}']
                    label = f'Data {idx+1}' if len(data_list) > 1 else 'nbar'
                    ax.plot(x_gain, nbar_vals, 'o-', label=label, color=color, alpha=0.7)
            
            ax.set_xlabel("Readout Probe Gain")
            ax.set_ylabel(f"n̄ ({key})")
            ax.set_title(f"Mean Photon Number vs Gain - {key}")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        filename = title.replace(' ', '_').replace(':', '') + '.png'
        self.save_plot(fig, filename=filename)
    
