from .general_fitting import GeneralFitting
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf
from copy import deepcopy

# Utility Functions

def rotate(x, y, theta):
    """
    Rotate points in the x-y plane by angle theta.
    """
    return (
        x * np.cos(theta) - y * np.sin(theta),
        x * np.sin(theta) + y * np.cos(theta)
    )

def full_rotate(data, theta):
    """
    Rotate all IQ data in a dictionary by angle theta.
    """
    data = deepcopy(data)
    data['Ig'], data['Qg'] = rotate(data['Ig'], data['Qg'], theta)
    data['Ie'], data['Qe'] = rotate(data['Ie'], data['Qe'], theta)
    if 'If' in data and 'Qf' in data:
        data['If'], data['Qf'] = rotate(data['If'], data['Qf'], theta)
    return data

def make_hist(data, nbins=200):
    """
    Create a normalized histogram from data.
    """
    hist, bin_edges = np.histogram(data, bins=nbins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, hist

# Gaussian Fitting

def gaussian(x, mag, cen, wid):
    """
    Gaussian function for fitting.
    """
    return mag / np.sqrt(2 * np.pi) / wid * np.exp(-((x - cen) ** 2) / 2 / wid**2)

def fit_gaussian(data, nbins=200, p0=None):
    """
    Fit a Gaussian distribution to data.
    """
    v, hist = make_hist(data, nbins)
    if p0 is None:
        p0 = [np.mean(v * hist) / np.mean(hist), np.std(data)]
    try:
        params, _ = curve_fit(lambda x, a, b: gaussian(x, 1, a, b), v, hist, p0=p0)
    except RuntimeError:
        params = p0
    return params, v, hist

# Fidelity Calculation

def calculate_fidelity(ng, ne):
    """
    Calculate fidelity and optimal threshold for state discrimination.
    """
    contrast = np.abs(((np.cumsum(ng) - np.cumsum(ne)) / (0.5 * ng.sum() + 0.5 * ne.sum())))
    tind = contrast.argmax()
    fidelity = contrast[tind]
    return fidelity, tind

# T1 Decay Modeling

def distfn(v, vg, ve, sigma, tm):
    """
    Distribution function modeling T1 decay during measurement.
    """
    dv = ve - vg
    return np.abs(
        tm / 2 / dv * np.exp(tm * (tm * sigma**2 / 2 / dv**2 - (v - vg) / dv)) * (
            erf((tm * sigma**2 / dv + ve - v) / np.sqrt(2) / sigma)
            + erf((-tm * sigma**2 / dv + v - vg) / np.sqrt(2) / sigma)
        )
    )

class Histogram(GeneralFitting):
    def __init__(self, data, span=None, verbose=True, active_reset=False, readout_per_round=None, threshold=None, config=None, station=None):
        super().__init__(data, readout_per_round, threshold, config, station)
        self.span = span
        self.verbose = verbose
        # self.active_reset = self.cfg.expt.active_reset 
        self.results = {}

    def analyze(self, data=None):
        """
        Analyze IQ data, rotate, and calculate fidelities.

        Parameters:
            data: Optional dictionary containing Ig, Qg, Ie, and Qe. If None, uses self.data.
        """
        # Use provided data or default to self.data
        if data is None:
            data = self.data

        Ig, Qg = data['Ig'], data['Qg']
        Ie, Qe = data['Ie'], data['Qe']

        theta = -np.arctan2((np.median(Qe) - np.median(Qg)), (np.median(Ie) - np.median(Ig)))
        rotated_data = full_rotate(data, theta)

        ng, binsg = np.histogram(rotated_data['Ig'], bins=200, density=True)
        ne, binse = np.histogram(rotated_data['Ie'], bins=200, density=True)

        fidelity, threshold_index = calculate_fidelity(ng, ne)
        threshold = binsg[threshold_index]

        self.results = {
            'fidelity': fidelity,
            'threshold': threshold,
            'rotation_angle': theta * 180 / np.pi,
            'rotated_data': rotated_data
        }

    def display(self, subdir=None):
        """
        Display the analysis results including rotated IQ data and histograms.
        """
        rotated_data = self.results.get('rotated_data', {})
        fidelity = self.results.get('fidelity', None)
        rotation_angle = self.results.get('rotation_angle', None)
        threshold = self.results.get('threshold', None)

        if not rotated_data:
            raise ValueError("No analysis results found. Please run analyze() first.")

        print(f"Fidelity: {fidelity * 100:.2f}%")
        print(f"Rotation Angle: {rotation_angle:.2f} degrees")
        print(f"Threshold: {threshold:.2f}")

        Ig, Qg = self.data['Ig'], self.data['Qg']
        Ie, Qe = self.data['Ie'], self.data['Qe']

        xg, yg = np.median(Ig), np.median(Qg)
        xe, ye = np.median(Ie), np.median(Qe)
        xg_rot, yg_rot = np.median(rotated_data['Ig']), np.median(rotated_data['Qg'])
        xe_rot, ye_rot = np.median(rotated_data['Ie']), np.median(rotated_data['Qe'])

        fig, axs = plt.subplots(2, 2, figsize=(8, 6))
        axs[0, 0].scatter(Ig, Qg, label=f'g (center: {xg:.2f}, {yg:.2f})', color='blue', s=1)
        axs[0, 0].scatter(Ie, Qe, label=f'e (center: {xe:.2f}, {ye:.2f})', color='red', s=1)
        axs[0, 0].set(title='Unrotated', xlabel='I [ADC levels]', ylabel='Q [ADC levels]')
        axs[0, 0].legend()

        axs[0, 1].scatter(rotated_data['Ig'], rotated_data['Qg'], label=f'g (center: {xg_rot:.2f}, {yg_rot:.2f})', color='blue', s=1)
        axs[0, 1].scatter(rotated_data['Ie'], rotated_data['Qg'], label=f'e (center: {xe_rot:.2f}, {ye_rot:.2f})', color='red', s=1)
        axs[0, 1].axvline(threshold, color='black', linestyle='--', label=f'Threshold: {threshold:.2f}')
        axs[0, 1].set(title='Rotated', xlabel='I [ADC levels]', ylabel='Q [ADC levels]')
        axs[0, 1].legend()

        axs[1, 0].hist(rotated_data['Ig'], bins=200, alpha=0.5, label='g', color='blue', density=True)
        axs[1, 0].hist(rotated_data['Ie'], bins=200, alpha=0.5, label='e', color='red', density=True)
        axs[1, 0].axvline(threshold, color='black', linestyle='--', label=f'Threshold: {threshold:.2f}')
        axs[1, 0].set(title=f'Histogram (Fidelity: {fidelity * 100:.2f}%)', xlabel='I [ADC levels]', ylabel='Density')
        axs[1, 0].legend()

        bins = np.linspace(min(rotated_data['Ig']), max(rotated_data['Ig']), 201)
        cumsum_g = np.cumsum(np.histogram(rotated_data['Ig'], bins=bins, density=True)[0])
        cumsum_e = np.cumsum(np.histogram(rotated_data['Ie'], bins=bins, density=True)[0])
        axs[1, 1].plot(bins[:-1], cumsum_g, label='g', color='blue')
        axs[1, 1].plot(bins[:-1], cumsum_e, label='e', color='red')
        axs[1, 1].axvline(threshold, color='black', linestyle='--', label=f'Threshold: {threshold:.2f}')
        axs[1, 1].set(title='Cumulative Counts', xlabel='I [ADC levels]', ylabel='Cumulative Density')
        axs[1, 1].legend()

        plt.tight_layout()
        self.save_plot(fig, filename="histogram.png", subdir=subdir)
        plt.show()

    # sweep 
    def parse_swept_histogram_data(self, sweep_params):
        """
        Parse histogram data with sweep parameters.
        
        Parameters:
            hstgrm_data: Dictionary containing 'iq_list_g' and 'iq_list_e' keys and xpts of swept params
            for instance 'xpts_readout_frequency'
            sweep_params: Dictionary describing the sweep parameters
            
        Returns:
            Dictionary with parsed I and Q data organized by sweep indices for g and e states
        """
        hstgrm_data = self.data
        # Get the number of sweep dimensions
        sweep_keys = list(sweep_params.keys())
        
        # Get shape information from sweep parameters
        sweep_expts = [sweep_params[key]['expts'] for key in sweep_keys]
        
        # Extract I and Q lists for g and e states
        iq_list_g = hstgrm_data['iqlist_g']  # shape: (..., n_shots, 2)
        iq_list_e = hstgrm_data['iqlist_e']  # shape: (..., n_shots, 2)
        
        # Separate I and Q components
        ilist_g = iq_list_g[..., 0]
        qlist_g = iq_list_g[..., 1]
        ilist_e = iq_list_e[..., 0]
        qlist_e = iq_list_e[..., 1]
        
        # Reshape data according to sweep dimensions
        shape_with_shots = tuple(sweep_expts) + (iq_list_g.shape[-2],)

        Ig = ilist_g.reshape(shape_with_shots)
        Qg = qlist_g.reshape(shape_with_shots)
        Ie = ilist_e.reshape(shape_with_shots)
        Qe = qlist_e.reshape(shape_with_shots)
        
        # sweep data 
        sweep_keys_and_xpts = {}
        for key in sweep_keys: 
            sweep_keys_and_xpts[key] = hstgrm_data['xpts_'+str(key)]
            
        
        # Organize results
        results = {
            'Ig': Ig,
            'Qg': Qg,
            'Ie': Ie,
            'Qe': Qe,
            'sweep_keys_and_xpts': sweep_keys_and_xpts
        }
        
        return results
    
    def analyze_swept_histogram_data(self, parsed_data):
        """
        Analyze each quartet of Ig, Qg, Ie, and Qe from parsed swept histogram data.

        Parameters:
            parsed_data: Dictionary containing parsed I and Q data organized by sweep indices.

        Returns:
            Dictionary with fidelity results added for each sweep index.
        """
        # Extract parsed data
        Ig = parsed_data['Ig']
        Qg = parsed_data['Qg']
        Ie = parsed_data['Ie']
        Qe = parsed_data['Qe']

        # Initialize fidelity results
        fidelity_results = np.zeros(Ig.shape[:-1])

        # Iterate over all sweep indices
        it = np.nditer(fidelity_results, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            # Get the current index
            idx = it.multi_index

            # Extract the current quartet of Ig, Qg, Ie, and Qe
            current_data = {
                'Ig': Ig[idx],
                'Qg': Qg[idx],
                'Ie': Ie[idx],
                'Qe': Qe[idx]
            }

            # Use the existing analyze method
            # self.data = current_data
            self.analyze(current_data)

            # Store fidelity result
            fidelity_results[idx] = self.results['fidelity']

            it.iternext()

        # Add fidelity results to the parsed data dictionary
        parsed_data['fidelity'] = fidelity_results

        return parsed_data