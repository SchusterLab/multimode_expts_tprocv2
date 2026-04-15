import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from ZNB import ZNB20
from VNA_funcs import *
from datamanagement import SlabFile
import fitting as fitter
import copy
from scipy.optimize import curve_fit
import scipy.constants as cs
from fit_resonator import ana_tls
from fit_resonator.ana_resonator import ResonatorFitter
from fit_resonator.ana_resonator import ResonatorData
import fit_resonator.pyCircFit_v3 as cf
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, NamedTuple


@dataclass
class ResonatorMeasurement:
    """
    Data class to store all measurements for a single resonator at a specific power level.
    """

    # Basic measurement parameters
    frequency: float  # Resonance frequency in Hz
    power: float  # Power in dBm
    power_at_device: float  # Power at device in dBm

    # Quality factors
    q_total: float  # Total quality factor
    q_internal: float  # Internal quality factor
    q_coupling: float  # Coupling quality factor

    # Alternative fit results (if available)
    q_total_alt: Optional[float] = None
    q_internal_alt: Optional[float] = None
    q_coupling_alt: Optional[float] = None
    frequency_alt: Optional[float] = None

    # Measurement details
    kappa: float = 0.0  # Linewidth in Hz
    photon_number: float = 0.0  # Average photon number
    averages: int = 1  # Number of averages used
    fit_parameters: List[float] = field(default_factory=list)  # Raw fit parameters

    # Measurement data
    raw_data: Dict[str, Any] = field(default_factory=dict)  # Raw measurement data


@dataclass
class PowerSweepResult:
    """
    Data class to store results from a power sweep for multiple resonators.
    """

    # Measurement results organized by frequency index and power index
    # Dictionary structure: {freq_idx: {power_idx: ResonatorMeasurement}}
    measurements: Dict[int, Dict[int, ResonatorMeasurement]]

    # Original and current frequencies
    frequencies: List[float]  # Original frequencies
    current_frequencies: List[
        float
    ]  # Current frequencies (may be updated during sweep)
    powers: List[float]  # Power values

    # Configuration used for the sweep
    config: Dict[str, Any]

    # Tracking parameters
    spans: List[float]  # Frequency spans for each resonator
    averaging_factors: List[float]  # Averaging factors for each resonator
    q_adjustment_factors: List[float]  # Q adjustment factors for each resonator
    keep_measuring: List[bool]  # Whether to continue measuring each resonator


def plot_scan(freq, amps, phase, pars=None, pinit=None, power=None, slope=None):
    """
    Plot the scan data with amplitude, phase, and IQ plots.

    Parameters:
    -----------
    freq : array
        Frequency points
    amps : array
        Amplitude data
    phase : array
        Phase data
    pars : list, optional
        Fit parameters
    pinit : list, optional
        Initial fit parameters
    power : float, optional
        Power level in dBm
    slope : float, optional
        Phase slope for unwrapping

    Returns:
    --------
    None
    """
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].plot(freq / 1e6, amps, "k.", markersize=3)
    ax[0].set_xlabel("Frequency (MHz)")
    ax[0].set_ylabel("Amplitude")
    if pars is not None:
        q = 1 / (1 / pars[1] + 1 / pars[2]) * 1e4
        qi = pars[1] * 1e4
        qc = pars[2] * 1e4
        lab = f"$Q$={q:.3g}\n $Q_i$={qi:.3g} \n $Q_c$={qc:.3g}"
        ax[0].plot(freq / 1e6, fitter.hangerS21func(freq, *pars))
        # ax[0].plot(freq / 1e6, fitter.hangerS21func(freq, *pinit))
    ax[0].set_title(f"Power: {power:.1f} dB")
    if slope is None:
        phase = np.unwrap(phase)
        slope, of = np.polyfit(freq, phase, 1)
        phase_sub = phase - slope * freq - of
    else:
        phase_sub = phase - slope * freq
    ax[1].plot(
        freq / 1e6,
        np.unwrap(phase_sub) - np.mean(np.unwrap(phase_sub)),
        "k.-",
        markersize=3,
    )
    ax[1].set_xlabel("Frequency (MHz)")
    ax[1].set_ylabel("Phase")
    # ax[0].legend(loc="lower right")
    ax[0].text(
        0.95,
        0.05,
        lab,
        transform=ax[0].transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(facecolor="white", alpha=0.5, edgecolor="black"),
    )
    ax2 = ax[0].twinx()
    if pars is not None:
        ax2.set_ylim(ax[0].get_ylim()[0] / pars[4], ax[0].get_ylim()[1] / pars[4])
    ax2.set_ylabel("Normalized Amplitude")

    ax[2].plot(amps * np.cos(phase_sub), amps * np.sin(phase_sub), "k.-")
    fig.tight_layout()
    # plt.show()


def _determine_scan_parameters(config, result, freq_idx, power_idx, q_total):
    """
    Determine the scan parameters based on the power index.

    Parameters:
    -----------
    config : dict
        Configuration dictionary
    result : PowerSweepResult
        Current result object
    freq_idx : int
        Index of the frequency being measured
    power_idx : int
        Index of the power being measured
    q_total : float
        Total quality factor

    Returns:
    --------
    tuple
        (npoints, span) - Number of points and span for the scan
    """
    # Use more points for higher powers
    if power_idx < 6:
        npoints = 2 * config["npoints"]
        span = result.spans[freq_idx] * 1.25
    elif power_idx > 0 and "next_time" in globals():
        if globals()["next_time"] > 2400:
            npoints = int(np.ceil(2 * config["npoints"]))
            config["span_inc"] = 0.85 * config["span_inc"]
            print("Reducing points due to long scan")
        else:
            npoints = config["npoints"]
            span = result.spans[freq_idx]
    else:
        npoints = config["npoints"]
        span = result.spans[freq_idx]

    return npoints, span


def _should_stop_measuring(result, freq_idx, next_time):
    """
    Determine if we should stop measuring a frequency.

    Parameters:
    -----------
    result : PowerSweepResult
        Current result object
    freq_idx : int
        Index of the frequency being measured
    next_time : float
        Estimated time for the next measurement in seconds

    Returns:
    --------
    bool
        True if we should stop measuring, False otherwise
    """
    print(result.q_adjustment_factors[freq_idx])
    print(next_time)
    return (result.q_adjustment_factors[freq_idx] > 1 and next_time > 3600) or (
        result.q_adjustment_factors[freq_idx] > 0.985 and next_time > 2100
    )


def _calculate_next_measurement_time(config, result, freq_idx):
    """
    Calculate the estimated time for the next measurement.

    Parameters:
    -----------
    config : dict
        Configuration dictionary
    result : PowerSweepResult
        Current result object
    freq_idx : int
        Index of the frequency being measured

    Returns:
    --------
    float
        Estimated time for the next measurement in seconds
    """
    return (
        1 / config["bandwidth"] * config["npoints"] * result.averaging_factors[freq_idx]
    )


def power_sweep_v2(config, VNA):
    """
    Perform a power sweep scan using the do_vna_scan function.
    This is an improved version that uses structured data storage.

    Parameters:
    -----------
    config : dict
        Configuration dictionary with measurement parameters:
        - base_path: Base directory path
        - folder: Folder name for saving data
        - freqs: List of center frequencies to scan
        - nvals: Number of power values to sweep
        - pow_inc: Power increment
        - pow_start: Starting power
        - span_inc: Span increment factor [number of linewidths for scan]
        - kappa_start: Initial kappa value
        - npoints: Number of frequency points
        - bandwidth: Measurement bandwidth in Hz
        - averages: Number of averages
        - att: Attenuation value (optional, default is 0)
    VNA : ZNB object
        The VNA instrument object

    Returns:
    --------
    PowerSweepResult
        Object containing all measurement results and parameters
    """
    # Create experiment path
    expt_path = os.path.join(config["base_path"], config["folder"])
    try:
        os.makedirs(expt_path)
        print('hi')
    except FileExistsError:
        print('hi2')
        pass
    except Exception as e:
        print(f"Error creating directory: {str(e)}")
        raise

    # Copy frequency lists to avoid modifying the original
    freqs0 = copy.deepcopy(config["freqs"])
    freqs = copy.deepcopy(config["freqs"])

    # Generate power values to sweep
    powers = np.arange(0, config["nvals"]) * config["pow_inc"] + config["pow_start"]

    # Initialize arrays for spans and averaging
    spans = config["span_inc"] * config["kappa_start"] * np.ones(len(freqs0))
    new_avgs = np.ones(len(freqs0))
    min_avg = 12

    # Initialize result storage
    keep_going = [True] * len(freqs0)
    q_adj = 0.9 * np.ones(len(freqs0))

    # Get attenuation value
    att = config.get("att", 0)

    # Create result object to track state
    result = PowerSweepResult(
        measurements={},
        frequencies=freqs0,
        current_frequencies=freqs,
        powers=powers,
        config=config,
        spans=spans,
        averaging_factors=new_avgs,
        q_adjustment_factors=q_adj,
        keep_measuring=keep_going,
    )
    output_path = os.path.join(expt_path)
    # Perform power sweep for each frequency
    for freq_idx, freq in enumerate(result.current_frequencies):
        result.measurements[freq_idx] = {}

        # Process each power point for this frequency
        for power_idx, power in enumerate(powers):
            # Skip if the quality factor is no longer changing
            if not result.keep_measuring[freq_idx]:
                continue

            # Set initial averaging for first power point
            if power_idx == 0:
                result.averaging_factors[freq_idx] = 1

            # Track averaging
            curr_avg = max(result.averaging_factors[freq_idx], min_avg)

            # Create filenames
            pow_name = f"{power:.0f}"
            fname = f"{result.frequencies[freq_idx]:1.0f}"

            # For first power point, do an initial scan with wider span
            if power_idx == 0:
                # Perform initial scan to find resonance frequency and linewidth
                measurement = _perform_initial_scan(
                    VNA, expt_path, result, freq_idx, power, att, fname, config["spar"]
                )

                # Update frequency and span based on measurement
                result.current_frequencies[freq_idx] = measurement.frequency
                result.spans[freq_idx] = measurement.kappa * config["span_inc"]

                # Store measurement
                result.measurements[freq_idx][power_idx] = measurement

                # Store parameters for next iteration
                prev_q = measurement.q_total
                prev_fit_params = measurement.fit_parameters

            # Use parameters from previous power point
            else:
                prev_measurement = result.measurements[freq_idx][power_idx - 1]
                prev_q = prev_measurement.q_total
                prev_fit_params = prev_measurement.fit_parameters

            # Determine scan parameters based on power index
            npoints, span = _determine_scan_parameters(
                config, result, freq_idx, power_idx, prev_q
            )

            # Configure VNA scan for this power point
            vna_config = {
                "freq_center": float(result.current_frequencies[freq_idx]),
                "span": span,
                "kappa": result.spans[freq_idx] / config["span_inc"],
                "npoints": npoints,
                "npoints1": config["npoints1"],
                "npoints2": config["npoints2"],
                "power": power,
                "bandwidth": config["bandwidth"],
                "averages": int(curr_avg),
                "spar": config["spar"],
            }

            # Perform the VNA scan
            tstart = datetime.datetime.now()
            time_expected = (
                1 / vna_config["bandwidth"] * npoints * vna_config["averages"]
            )
            print(f"Expected time: {time_expected / 60:.2f} min")

            file_name = f"res_{fname}_{pow_name}dbm.h5"
            data = _perform_vna_scan(VNA, file_name, expt_path, vna_config, config, att)

            tfinish = datetime.datetime.now()
            elapsed_time = (tfinish - tstart).total_seconds()
            print(
                f"Time elapsed: {elapsed_time / 60:.2f} min, expected time: {time_expected / 60:.2f} min"
            )

            # Fit data to find resonator parameters
            min_freq = result.current_frequencies[
                freq_idx
            ]  # Use previous frequency as initial guess

            # Determine fit parameters based on power index
            # if power_idx < 8:
            #     fitparams = [
            #         min_freq,
            #         prev_fit_params[1],
            #         prev_fit_params[2],
            #         prev_fit_params[3],
            #         np.max(10 ** (data["amps"] / 20)),
            #     ]
            #     freq_center, q_total, kappa, fit_params = fit_resonator(
            #         data, power, fitparams
            #     )
            #     q_coupling = fit_params[2] * 1e4
            # else:
            #     # For higher power indices, use mean of previous coupling Q values
            #     qc_values = [
            #         result.measurements[freq_idx][i].q_coupling
            #         for i in range(3, 7)
            #         if i < power_idx
            #     ]
            #     qc_best = np.mean(qc_values) if qc_values else prev_fit_params[2] * 1e4

            #     fitparams = [
            #         min_freq,
            #         prev_fit_params[1],
            #         prev_fit_params[3],
            #         np.max(10 ** (data["amps"] / 20)),
            #     ]
            #     freq_center, q_total, kappa, fit_params = fit_resonator(
            #         data, power, fitparams, qc_best
            #     )
            #     q_coupling = qc_best

            # Calculate internal Q
            # q_internal = fit_params[1] * 1e4

            data = ResonatorData.fit_phase(data)
            output = ResonatorFitter.fit_resonator(
                data, fname, output_path, plot=True, fix_freq=False
            )
            # freq_center, q_total, kappa, fit_params
            q_total = output[0][0]
            q_coupling = output[0][1]
            freq_center = output[0][2]
            q_internal = 1 / (1 / q_total - 1 / q_coupling)
            kappa = freq_center / q_total

            # Calculate photon number
            pin = (
                power
                - config["att"]
                - config["db_slope"] * (freq_center / 1e9 - config["freq_0"])
            )
            photon_number = n(pin, freq_center, q_total, q_coupling)

            # Create measurement object
            measurement = ResonatorMeasurement(
                frequency=freq_center,
                power=power,
                power_at_device=power - att,
                q_total=q_total,
                q_internal=q_internal,
                q_coupling=q_coupling,
                kappa=kappa,
                photon_number=photon_number,
                averages=int(curr_avg),
                fit_parameters=output,
                raw_data=data,
            )

            # Store the measurement
            result.measurements[freq_idx][power_idx] = measurement

            # Plot Qi vs power
            _plot_qi_vs_photon(result.measurements, freq_idx, expt_path)

            # Calculate new averaging based on photon number
            if power_idx > 0:
                result.q_adjustment_factors[freq_idx] = measurement.q_total / prev_q

            if "avg_corr" in config:
                pin = (
                    power
                    - config["att"]
                    - config["db_slope"]
                    * (measurement.frequency / 1e9 - config["freq_0"])
                )
                tau_prop = (
                    10 ** (-pin / 10)
                    * (measurement.q_coupling / measurement.q_total) ** 2
                    * 1e-11
                )
                print(f"Tau proportionality: {tau_prop}")

                result.averaging_factors[freq_idx] = np.round(
                    config["avg_corr"]
                    * tau_prop
                    / result.q_adjustment_factors[freq_idx] ** 2
                )
                print(
                    f"Pin {power - config['att']:.1f}, N photons: {measurement.photon_number:.3g}, navg: {int(result.averaging_factors[freq_idx])}"
                )

            # Determine if we should continue measuring this frequency
            next_time = _calculate_next_measurement_time(config, result, freq_idx)
            print(
                f"Next time: {next_time / 60:.2f} min, q_adj: {result.q_adjustment_factors[freq_idx]:.3f}"
            )

            if _should_stop_measuring(result, freq_idx, next_time):
                result.keep_measuring[freq_idx] = False
                print(
                    f"Stopping frequency {result.current_frequencies[freq_idx] / 1e9:.5f} GHz"
                )

            # Update span for next measurement
            result.spans[freq_idx] = measurement.kappa * config["span_inc"]

    return result


def _plot_qi_vs_photon(measurements, freq_idx, expt_path):
    """
    Plot internal quality factor vs photon number.

    Parameters:
    -----------
    measurements : dict
        Dictionary of measurements
    freq_idx : int
        Frequency index
    expt_path : str
        Path to save the plot
    """
    if freq_idx not in measurements:
        return

    # Get data for plotting
    power_indices = sorted(measurements[freq_idx].keys())
    photon_numbers = [measurements[freq_idx][i].photon_number for i in power_indices]
    qi_values = [measurements[freq_idx][i].q_internal for i in power_indices]
    qc_values = [measurements[freq_idx][i].q_coupling for i in power_indices]

    # Get alternative fit data if available
    qi_alt_values = []
    qc_alt_values = []
    for i in power_indices:
        if measurements[freq_idx][i].q_internal_alt is not None:
            qi_alt_values.append(measurements[freq_idx][i].q_internal_alt)
            qc_alt_values.append(measurements[freq_idx][i].q_coupling_alt)

    # Create plot
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))

    # Plot Qi vs photon number
    ax[0].semilogx(photon_numbers, qi_values, "o-", label="Primary fit")
    if qi_alt_values:
        ax[0].semilogx(
            photon_numbers[: len(qi_alt_values)], qi_alt_values, "s--", label="Alt fit"
        )

    # Fit Qi vs power to an exponential if we have enough points
    if len(qi_values) > 6:
        try:
            min_freq = measurements[freq_idx][power_indices[0]].frequency
            q_fitn = lambda n, Qtls0, Qoth, nc, beta: ana_tls.Qtotn(
                n, 0.04, min_freq, Qtls0, Qoth, nc, beta
            )
            p = [np.min(qi_values), np.max(qi_values), 3, 0.4]
            p, err = curve_fit(q_fitn, photon_numbers, qi_values, p0=p)
            ax[0].plot(
                photon_numbers,
                q_fitn(np.array(photon_numbers), *p),
                "r--",
                label="Exp fit",
            )
            print(f"Fit parameters: {p}")
        except Exception as e:
            print(f"Fit failed: {str(e)}")

    ax[0].set_xlabel("Number of Photons")
    ax[0].set_ylabel("Internal Quality Factor ($Q_i$)")
    ax[0].set_title(
        f"Frequency: {measurements[freq_idx][power_indices[0]].frequency/1e9:.5f} GHz"
    )
    if len(qi_alt_values) > 0:
        ax[0].legend()

    # Plot Qc vs photon number
    ax[1].semilogx(photon_numbers, qc_values, "o-", label="Primary fit")
    if qc_alt_values:
        ax[1].semilogx(
            photon_numbers[: len(qc_alt_values)], qc_alt_values, "s--", label="Alt fit"
        )

    ax[1].set_xlabel("Number of Photons")
    ax[1].set_ylabel("Coupling Quality Factor ($Q_c$)")
    if len(qc_alt_values) > 0:
        ax[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(expt_path, f"Qi_vs_power_{freq_idx}.png"))
    plt.close(fig)


# For backward compatibility
def power_sweep(config, VNA):
    """
    Perform a power sweep scan using the do_vna_scan function.
    This is a wrapper around power_sweep_v2 for backward compatibility.

    Parameters:
    -----------
    config : dict
        Configuration dictionary with measurement parameters
    VNA : ZNB object
        The VNA instrument object

    Returns:
    --------
    list
        List of parameter lists for each frequency and power
    """
    result = power_sweep_v2(config, VNA)

    # Convert the result to the old format for backward compatibility
    pars_list = []
    for freq_idx in sorted(result.measurements.keys()):
        freq_pars = []
        for power_idx in sorted(result.measurements[freq_idx].keys()):
            freq_pars.append(result.measurements[freq_idx][power_idx].fit_parameters)
        pars_list.append(freq_pars)

    return pars_list


def fit_resonator(data, power, fitparams=None, qc=None, plot=False):

    # Convert amplitude from dB to linear scale
    amps_linear = 10 ** (data["amps"] / 20)
    # f0, Qi, Qe, phi, scale, a0, slope # Qi/Qe in units of 10k
    if qc is not None:
        hangerfit = lambda f, f0, qi, phi, scale: fitter.hangerS21func(
            f, f0, qi, qc / 1e4, phi, scale
        )

        pars, err = curve_fit(hangerfit, data["freqs"], amps_linear, p0=fitparams)
        freq_center = pars[0]
        q = 1 / (1 / pars[1] / 1e4 + 1 / qc)
        kappa = freq_center / q
        r2 = fitter.get_r2(data["freqs"], amps_linear, hangerfit, pars)
        err = np.sqrt(np.diag(err))
        # print('f error: ', err[0], 'Qi error: ', err[1], 'phi error: ', err[2], 'scale error: ', err[3])
        print(f"Qi err: {err[1]/pars[1]}")
        pars = [pars[0], pars[1], qc / 1e4, pars[2], pars[3]]
        fitparams = [pars[0], pars[1], qc / 1e4, pars[3], pars[4]]
    else:
        if fitparams is None:
            min_freq = data["freqs"][np.argmin(amps_linear)]
            fitparams = [min_freq, 100, 100, 0, np.max(amps_linear)]

        pars, err, pinit = fitter.fithanger(
            data["freqs"], amps_linear, fitparams=fitparams
        )

        pars, err, pinit = fitter.fithanger(data["freqs"], amps_linear, fitparams=pars)

        freq_center = pars[0]
        q = 1 / (1 / pars[1] + 1 / pars[2]) * 1e4
        kappa = freq_center / q
    if plot:
        plot_scan(
            data["freqs"],
            amps_linear,
            data["phases"],
            pars,
            fitparams,
            power,
        )

    return freq_center, q, kappa, pars


def n(p, f, q, qc):
    return pow_res(p) * q**2 / qc / (cs.h * f**2 * np.pi)


def pow_res(p):
    return 10 ** (p / 10) * 1e-3


def get_default_power_sweep_config(custom_config=None):
    """
    Get default configuration for power_sweep function.

    Parameters:
    -----------
    custom_config : dict, optional
        Dictionary with custom configuration values to override defaults

    Returns:
    --------
    dict
        Configuration dictionary with default values for power_sweep
    """
    # Define default configuration
    default_config = {
        # File paths
        "base_path": "./data",
        "folder": f"power_sweep_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        # Frequency settings
        "freqs": np.array([6]) * 1e9,  # Default center frequency in Hz
        "span_inc": 10,  # Span as multiple of linewidth
        "kappa_start": 30000,  # Initial linewidth estimate in Hz
        # Power sweep settings
        "nvals": 18,  # Number of power points
        "pow_start": -5,  # Starting power in dBm
        "pow_inc": -5,  # Power increment in dB
        # Measurement settings
        "npoints": 201,  # Number of frequency points
        "npoints1": 10,
        "npoints2": 27,
        "bandwidth": 100,  # Measurement bandwidth in Hz
        "averages": 1,  # Number of averages
        "att": 60,  # Attenuation in dB
        "type": "lin",
        "spar": "s21",
        "freq_0": 6,
        "db_slope": 4,
        # Analysis settings
        "avg_corr": 1e6,  # Correction factor for averaging
    }
    if (
        custom_config is not None
        and "type" in custom_config
        and custom_config["type"] == "single"
    ):
        default_config["npoints"] = 31

    # Override defaults with custom values if provided
    if custom_config is not None:
        for key, value in custom_config.items():
            default_config[key] = value

    return default_config


def get_homophase(config):
    """
    Calculate the list of frequencies that gives you equal phase spacing
    Parameters:
    config (dict): A dictionary containing the following keys:
        - "npoints" (int): Number of points in the frequency list.
        - "span" (float): Frequency span.
        - "kappa" (float): linewidth
        - "kappa_inc" (float): expected linewidth fudge factor (fix me).
        - "center_freq" (float): Center frequency.
    Returns:
    numpy.ndarray: An array containing the calculated frequency list.
    """
    nlin = 2

    N = config["npoints"] - nlin * 2
    df = config["span"]
    w = df / config["kappa"] * config["kappa_inc"]
    at = np.arctan(2 * w / (1 - w**2)) + np.pi
    R = w / np.tan(at / 2)
    fr = config["freq_center"]
    n = np.arange(N) - N / 2 + 1 / 2
    flist = fr + R * df / (2 * w) * np.tan(n / (N - 1) * at)
    flist_lin = (
        -np.arange(nlin, 0, -1) * df / N * 3
        + config["freq_center"]
        - config["span"] / 2
    )
    flist_linp = (
        np.arange(1, nlin + 1) * df / N * 3 + config["freq_center"] + config["span"] / 2
    )
    flist = np.concatenate([flist_lin, flist, flist_linp])
    return flist


def _perform_vna_scan(VNA, file_name, expt_path, vna_config, config, att):
    """
    Perform a VNA scan based on the scan type specified in the config.

    Parameters:
    -----------
    VNA : ZNB object
        The VNA instrument object
    file_name : str
        Name for saving the data
    expt_path : str
        Path for saving data
    vna_config : dict
        VNA configuration dictionary
    config : dict
        Main configuration dictionary
    att : float
        Attenuation value

    Returns:
    --------
    dict
        Dictionary containing the measurement data
    """
    if config["type"] == "lin":
        return do_vna_scan(VNA, file_name, expt_path, vna_config, att=att, plot=False)
    elif config["type"] == "single":
        return do_vna_scan_single_point(
            VNA, file_name, expt_path, vna_config, plot=False
        )
    else:
        return do_vna_scan_segments(VNA, file_name, expt_path, vna_config, plot=False)


def _perform_initial_scan(
    VNA, expt_path, result, freq_idx, power, att, fname, spar="S21"
):
    """
    Perform an initial scan to find the resonance frequency and linewidth.

    Parameters:
    -----------
    VNA : ZNB object
        The VNA instrument object
    expt_path : str
        Path for saving data
    result : PowerSweepResult
        Current result object
    freq_idx : int
        Index of the frequency being measured
    power : float
        Power level in dBm
    att : float
        Attenuation value
    fname : str
        Base filename

    Returns:
    --------
    ResonatorMeasurement
        Measurement result for the initial scan
    """
    # Configure VNA scan with wider span
    vna_config = {
        "freq_center": float(result.current_frequencies[freq_idx]),
        "span": float(result.spans[freq_idx]) * 1.3,
        "npoints": 800,
        "power": power,
        "bandwidth": 10 * result.config["bandwidth"],
        "averages": 1,
        'spar':spar
    }

    # Perform VNA scan
    file_name = f"res_{fname}_single.h5"
    data = do_vna_scan(VNA, file_name, expt_path, vna_config, att=att, plot=False)

    # Fit resonator to find center frequency and kappa
    min_freq = result.current_frequencies[freq_idx]  # Initial guess
    #freq_center, q_total, kappa, fit_params = fit_resonator(data, power, plot=True)

    # Calculate quality factors
    # q_internal = fit_params[1] * 1e4
    # q_coupling = fit_params[2] * 1e4

    data = ResonatorData.fit_phase(data)
    output = ResonatorFitter.fit_resonator(
        data, fname, expt_path, plot=True, fix_freq=False
    )
    # freq_center, q_total, kappa, fit_params
    q_total = output[0][0]
    q_coupling = output[0][1]
    freq_center = output[0][2]
    q_internal = 1 / (1 / q_total - 1 / q_coupling)
    kappa = freq_center / q_total

    # Calculate photon number
    pin = (
        power
        - result.config["att"]
        - result.config["db_slope"] * (freq_center / 1e9 - result.config["freq_0"])
    )
    photon_number = n(pin, freq_center, q_total, q_coupling)

    # Create and return measurement object
    return ResonatorMeasurement(
        frequency=freq_center,
        power=power,
        power_at_device=power - att,
        q_total=q_total,
        q_internal=q_internal,
        q_coupling=q_coupling,
        kappa=kappa,
        photon_number=photon_number,
        averages=1,
        raw_data=data,
    )


def do_vna_scan(VNA, file_name, expt_path, cfg, att=0, plot=True):
    """
    Perform a VNA scan and save the data to a file.

    Parameters:
    -----------
    VNA : ZNB object
        The VNA instrument object
    file_name : str
        Name for saving the data
    expt_path : str
        Path for saving the data
    cfg : dict
        Configuration dictionary with measurement parameters:
        - freq_center: Center frequency in MHz
        - span: Frequency span in MHz
        - npoints: Number of frequency points
        - power: VNA output power in dBm
        - bandwidth: Measurement bandwidth in Hz
        - averages: Number of averages
        - spar: Scattering parameter (e.g., 'S21')
    att : float, optional
        Attenuation value in dB, default is 0
    plot : bool, optional
        Whether to plot the data, default is True

    Returns:
    --------
    dict
        Dictionary containing the measurement data
    """
    try:
        # Calculate frequency range
        freq_center = cfg["freq_center"]
        freq_span = cfg["span"]
        freq_start = freq_center - 0.5 * freq_span
        freq_stop = freq_center + 0.5 * freq_span

        # Generate frequency sweep points
        freq_sweep = np.linspace(freq_start, freq_stop, cfg["npoints"])

        # Get power setting
        power = cfg["power"]

        # Prepare trace and scattering parameter
        trace_name = ("trace1",)
        scattering_parameter = (cfg["spar"],)

        # Configure VNA
        VNA.initialize_one_tone_spectroscopy(trace_name, scattering_parameter)
        VNA.set_startfrequency(freq_start)
        VNA.set_stopfrequency(freq_stop)
        VNA.set_points(cfg["npoints"])
        VNA.set_power(power)
        VNA.set_measBW(cfg["bandwidth"])
        VNA.set_sweeps(cfg["averages"])
        VNA.set_averages(cfg["averages"])
        VNA.set_averagestatus(status="on")

        # Calculate actual power at device
        power_at_device = power - att

        # Perform measurement
        VNA.measure()

        # Get measurement data
        [amps, phases] = VNA.get_traces(trace_name)[0]

        # Create data dictionary with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        data = {
            "series": timestamp,
            "amps": amps,
            "phases": phases,
            "freqs": freq_sweep,
            "vna_power": power,
            "power_at_device": power_at_device,
            "bandwidth": cfg["bandwidth"],
            "averages": cfg["averages"],
            "npoints": cfg["npoints"],
        }
        tfinish = datetime.datetime.now()
        # print(f"Time elapsed: {(tfinish-tstart)/60} min, expected time: {time_expected/60} min")
        # Save data to file using native SlabFile methods
        file_path = os.path.join(expt_path, file_name)
        with SlabFile(file_path, "w") as f:
            # Save arrays using add_data
            f.add_data(f, "fpts", freq_sweep)
            f.add_data(f, "mags", amps)
            f.add_data(f, "phases", phases)

            # Save scalar values as a dictionary
            metadata = {
                "vna_power": power,
                "power_at_device": power_at_device,
                "averages": cfg["averages"],
                "bandwidth": cfg["bandwidth"],
                "npoints": cfg["npoints"],
                "timestamp": timestamp,
            }
            f.save_dict(metadata)

        # Plot data if requested
        if plot:
            plot_all(data, filepath=expt_path)

        return data

    except Exception as e:
        print(f"Error in do_vna_scan: {str(e)}")
        raise


def do_vna_scan_segments(
    VNA, file_name, expt_path, cfg, warm_att=0, cold_att=0, plot=True
):
    """
    Perform a VNA scan with segmented frequency ranges and save the data to a file.

    This function divides the frequency span into three segments:
    1. Lower frequency range with fewer points
    2. Center frequency range with more points (higher resolution)
    3. Upper frequency range with fewer points

    Parameters:
    -----------
    VNA : ZNB object
        The VNA instrument object
    file_name : str
        Name for saving the data
    expt_path : str
        Path for saving the data
    cfg : dict
        Configuration dictionary with measurement parameters:
        - freq_center: Center frequency in MHz
        - span: Frequency span in MHz
        - npoints1: Number of frequency points for outer segments
        - npoints2: Number of frequency points for center segment
        - power: VNA output power in dBm
        - bandwidth: Measurement bandwidth in Hz
        - averages: Number of averages
        - npoints: Total number of points (for metadata)
    warm_att : float, optional
        Warm attenuation value in dB, default is 0
    cold_att : float, optional
        Cold attenuation value in dB, default is 0
    plot : bool, optional
        Whether to plot the data, default is True

    Returns:
    --------
    dict
        Dictionary containing the measurement data
    """
    try:
        # Get frequency center and bandwidth
        freq_center = cfg["freq_center"]
        bandwidth = cfg["bandwidth"]
        power = cfg["power"]

        # Prepare trace and scattering parameter
        scattering_parameter = (cfg["spar"],)
        trace_name = ("trace1",)

        # Define segment width ratio (center segment width / total span)
        wid = 0.2

        # Calculate point spacing for smooth transitions between segments
        pt_spc = cfg["span"] * (1 - wid) / 2 / cfg["npoints1"]

        # Generate frequency points for each segment
        freq_sweep0 = np.linspace(
            freq_center - cfg["span"] / 2,
            freq_center - wid / 2 * cfg["span"] - pt_spc,
            cfg["npoints1"],
        )
        freq_sweep1 = np.linspace(
            freq_center - wid * cfg["span"] / 2,
            freq_center + wid * cfg["span"] / 2,
            cfg["npoints2"],
        )
        freq_sweep2 = np.linspace(
            freq_center + wid / 2 * cfg["span"] + pt_spc,
            freq_center + cfg["span"] / 2,
            cfg["npoints1"],
        )

        # Combine all frequency points
        freq_sweep = np.concatenate((freq_sweep0, freq_sweep1, freq_sweep2))

        # Initialize VNA
        VNA.initialize_one_tone_spectroscopy_seg(trace_name, scattering_parameter)

        # Define sweep type
        swp_typ = "dwell"

        # Configure VNA segments
        # Segment 0: Lower frequency range
        npoints = cfg["npoints1"] + cfg["npoints2"] + cfg["npoints1"]
        time_expected = 1 / cfg["bandwidth"] * npoints * cfg["averages"]
        print(f"expected time: {time_expected / 60} min")
        t = 1 / cfg["bandwidth"] / 6
        tstart = datetime.datetime.now()
        VNA.define_segment(
            1,
            freq_center - cfg["span"] / 2,
            freq_center - wid / 2 * cfg["span"] - pt_spc,
            cfg["npoints1"],
            cfg["power"],
            t,
            cfg["bandwidth"],
            set_time=swp_typ,
        )

        # Segment 1: Center frequency range (higher resolution)
        VNA.define_segment(
            2,
            freq_center - wid * cfg["span"] / 2,
            freq_center + wid * cfg["span"] / 2,
            cfg["npoints2"],
            cfg["power"],
            t,
            cfg["bandwidth"],
            set_time=swp_typ,
        )

        # Segment 2: Upper frequency range
        VNA.define_segment(
            3,
            freq_center + wid / 2 * cfg["span"] + pt_spc,
            freq_center + cfg["span"] / 2,
            cfg["npoints1"],
            cfg["power"],
            t,
            cfg["bandwidth"],
            set_time=swp_typ,
        )

        # Set averaging parameters
        VNA.set_averages(cfg["averages"])
        VNA.set_averagestatus(status="on")
        VNA.set_sweeps(cfg["averages"])

        # Calculate actual power at device
        power_at_device = cfg["power"] - warm_att - cold_att

        # Perform measurement
        VNA.measure()

        # Get measurement data
        [amps, phases] = VNA.get_traces(trace_name)[0]
        tfinish = datetime.datetime.now()
        elapsed_time = (tfinish - tstart).total_seconds()
        print(f"Time elapsed: {elapsed_time / 60} min")

        # Create data dictionary with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        data = {
            "series": timestamp,
            "amps": amps,
            "phases": phases,
            "freqs": freq_sweep,
            "vna_power": power,
            "power_at_device": power_at_device,
            "bandwidth": bandwidth,
            "averages": cfg["averages"],
            "npoints": cfg["npoints"],
        }

        # Save data to file using native SlabFile methods
        file_path = os.path.join(expt_path, file_name)
        with SlabFile(file_path, "w") as f:
            # Save arrays using add_data
            f.add_data(f, "freqs", freq_sweep)
            f.add_data(f, "amps", amps)
            f.add_data(f, "phases", phases)

            # Save scalar values as a dictionary
            metadata = {
                "vna_power": power,
                "power_at_device": power_at_device,
                "averages": cfg["averages"],
                "bandwidth": bandwidth,
                "npoints": cfg["npoints"],
                "timestamp": timestamp,
            }
            f.save_dict(metadata)

        # Plot data if requested
        if plot:
            plot_all(data, filepath=expt_path)

        return data

    except Exception as e:
        print(f"Error in do_vna_scan_segments: {str(e)}")
        raise


def do_vna_scan_single_point(
    VNA, file_name, expt_path, cfg, warm_att=0, cold_att=0, plot=True
):
    """
    Perform a VNA scan with segmented frequency ranges and save the data to a file.

    This function divides the frequency span into three segments:
    1. Lower frequency range with fewer points
    2. Center frequency range with more points (higher resolution)
    3. Upper frequency range with fewer points

    Parameters:
    -----------
    VNA : ZNB object
        The VNA instrument object
    file_name : str
        Name for saving the data
    expt_path : str
        Path for saving the data
    cfg : dict
        Configuration dictionary with measurement parameters:
        - freq_center: Center frequency in MHz
        - span: Frequency span in MHz
        - npoints: Number of frequency points for outer segments
        - power: VNA output power in dBm
        - bandwidth: Measurement bandwidth in Hz
        - averages: Number of averages
        - npoints: Total number of points (for metadata)
    warm_att : float, optional
        Warm attenuation value in dB, default is 0
    cold_att : float, optional
        Cold attenuation value in dB, default is 0
    plot : bool, optional
        Whether to plot the data, default is True

    Returns:
    --------
    dict
        Dictionary containing the measurement data
    """
    try:
        # Get frequency center and bandwidth

        bandwidth = cfg["bandwidth"]
        power = cfg["power"]
        cfg["kappa_inc"] = 1.1

        freq_list = get_homophase(cfg)

        # Prepare trace and scattering parameter
        scattering_parameter = (cfg["spar"],)
        trace_name = ("trace1",)

        # Initialize VNA
        # start by deleting all previous segments.
        VNA.delete_segments()

        # Define sweep type
        swp_typ = "dwell"

        # Configure VNA segments
        # Segment 0: Lower frequency range

        t = 1 / cfg["bandwidth"] / 6

        for i, freq in enumerate(freq_list):
            VNA.define_segment(
                i + 1,
                freq,
                freq,
                1,
                cfg["power"],
                t,
                cfg["bandwidth"],
                set_time=swp_typ,
            )
        VNA.initialize_one_tone_spectroscopy_seg(trace_name, scattering_parameter)
        # Set averaging parameters
        VNA.set_averages(cfg["averages"])
        VNA.set_averagestatus(status="on")
        VNA.set_sweeps(cfg["averages"])

        # Calculate actual power at device
        power_at_device = cfg["power"] - warm_att - cold_att

        # Perform measurement
        VNA.measure()

        # Get measurement data
        [amps, phases] = VNA.get_traces(trace_name)[0]

        # Create data dictionary with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        data = {
            "series": timestamp,
            "amps": amps,
            "phases": phases,
            "freqs": freq_list,
            "vna_power": power,
            "power_at_device": power_at_device,
            "bandwidth": bandwidth,
            "averages": cfg["averages"],
            "npoints": cfg["npoints"],
        }

        # Save data to file
        file_path = os.path.join(expt_path, file_name)

        with SlabFile(file_path, "w") as f:
            # Save arrays using add_data
            f.add_data(f, "fpts", freq_list)
            f.add_data(f, "mags", amps)
            f.add_data(f, "phases", phases)

            # Save scalar values as a dictionary
            metadata = {
                "vna_power": power,
                "averages": cfg["averages"],
                "bandwidth": bandwidth,
                "npoints": cfg["npoints"],
                "timestamp": timestamp,
            }
            f.save_dict(metadata)
        #           f.append_pt["attenuation"] = {"warm": warm_att, "cold": cold_att}

        # Plot data if requested
        if plot:
            plot_all(data, filepath=expt_path)

        return data

    except Exception as e:
        print(f"Error in do_vna_scan_segments: {str(e)}")
        raise
