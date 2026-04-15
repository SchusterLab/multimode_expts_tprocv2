import numpy as np
import matplotlib.pyplot as plt


import os
import json
import datetime
from time import *

from ZNB import ZNB20


## ------------------
## Import Functions
## ------------------
def write_file(data, filepath, filename=None):
    filename = data["series"] + "_" + filename
    print("Storing data to filepath: ", filepath)
    print("Storing data to filename: ", filename)
    if not os.path.isdir(filepath):
        os.makedirs(filepath)
    np.savez(os.path.join(filepath, filename), data=data)
    return filepath, filename


def read_file(filepath, filename):
    readin = np.load(os.path.join(filepath, filename), allow_pickle=True)
    return readin["data"].item()


## ------------------
## Plotting Functions
## ------------------
def plot_amp(data, filepath=None, filename=None, save_fig=True):
    plt.plot(data["freqs"], data["amps"])
    plt.title(
        f"{data['series']}, device power={data['power_at_device']} dBm \n IFBW={data['bandwidth']} Hz, avgs={data['averages']}"
    )
    plt.xlabel("Frequency (Hz)")
    plt.ylabel(f"Amplitude (dB), relative to {data['vna_power']} dBm at VNA")
    if save_fig:
        if filepath is None:
            print(
                "Error: no filepath specified.  Either specify a filepath in the arguments or set save_fig=False"
            )
            return
        if filename is None:
            filename = data["series"] + "_amp.pdf"
        else:
            filename = data["series"] + "_amp_" + filename + ".pdf"
        plt.savefig(os.path.join(filepath, filename))
    plt.show()
    return


def plot_real_imag(data, filepath=None, filename=None, save_fig=True):
    plt.plot(data["freqs"], data["reals"])
    plt.plot(data["freqs"], data["imags"])
    plt.title(
        f"{data['series']}, device power={data['power_at_device']} dBm \n IFBW={data['bandwidth']} Hz, avgs={data['averages']}"
    )
    plt.xlabel("Frequency (Hz)")
    plt.ylabel(f"Amplitude, relative to {data['vna_power']} dBm at VNA")
    if save_fig:
        if filepath is None:
            print(
                "Error: no filepath specified.  Either specify a filepath in the arguments or set save_fig=False"
            )
            return
        if filename is None:
            filename = data["series"] + "_amp.pdf"
        else:
            filename = data["series"] + "_amp_" + filename + ".pdf"
        plt.savefig(os.path.join(filepath, filename))
    plt.show()
    return


def plot_phase(data, filepath=None, filename=None, save_fig=True):
    plt.plot(data["freqs"], data["phases"])
    plt.title(
        f"{data['series']}, device power={data['power_at_device']} dBm \n IFBW={data['bandwidth']} Hz, avgs={data['averages']}"
    )
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (rad)")
    if save_fig:
        if filepath is None:
            print(
                "Error: no filepath specified.  Either specify a filepath in the arguments or set save_fig=False"
            )
            return
        if filename is None:
            filename = data["series"] + "_phase.pdf"
        else:
            filename = data["series"] + "_phase_" + filename + ".pdf"
        plt.savefig(os.path.join(filepath, filename))
    plt.show()
    return


## ------------------
## Correcting phase for line delay
## ------------------
def unwrap_phases(data, force_line_delay_val=None):
    ## Set up output containers, unwrap phases (i.e. remove 2pi jumps)
    corrected_phases = np.zeros(len(data["phases"] - 1))
    unwrapped = np.unwrap(data["phases"])
    ## Give user the option to manually set a line delay
    ## If no value is supplied, calculate the line delay from the data
    if force_line_delay_val is None:
        line_delay = np.mean(unwrapped[1:] - unwrapped[:-1]) / (
            data["freqs"][1:] - data["freqs"][:-1]
        )
        line_delay = np.mean(line_delay)
        print("Calculated line delay:", line_delay)
    else:
        print("Manually set line delay:", force_line_delay_val)
        line_delay = force_line_delay_val
    for n, phase in enumerate(unwrapped):
        corrected_phases[n] = phase - (data["freqs"][n] - data["freqs"][0]) * line_delay
    return corrected_phases, line_delay


def plot_unwrapped_phase(
    data, filepath=None, filename=None, save_fig=True, force_line_delay_val=None
):
    ## Unwrap phases
    corrected_phases, line_delay = unwrap_phases(data, force_line_delay_val)
    ## Plot unwrapped phases
    plt.plot(data["freqs"], corrected_phases)
    plt.title(
        f"{data['series']}, device power={data['power_at_device']} dBm \n IFBW={data['bandwidth']} Hz, avgs={data['averages']}"
    )
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Corrected phase (rad)")
    if save_fig:
        if filepath is None:
            print(
                "Error: no filepath specified.  Either specify a filepath in the arguments or set save_fig=False"
            )
            return
        if filename is None:
            filename = data["series"] + "_phase.pdf"
        else:
            filename = data["series"] + "_phase_" + filename + ".pdf"
        plt.savefig(os.path.join(filepath, filename))
    plt.show()
    return


def plot_all(
    data, filepath=None, filename=None, save_fig=True, force_line_delay_val=None
):
    fig, ax = plt.subplots(2, 1, figsize=(8, 7))
    ax[0].plot(data["freqs"], data["amps"], ".-")
    fig.suptitle(
        f"{data['series']}, device power={data['power_at_device']} dBm \n IFBW={data['bandwidth']} Hz, avgs={data['averages']}"
    )
    ax[1].set_xlabel("Frequency (Hz)")
    ax[0].set_ylabel(f"Amplitude (dB), relative to {data['vna_power']} dBm at VNA")
    corrected_phases, line_delay = unwrap_phases(data, force_line_delay_val)

    ax[1].plot(data["freqs"], corrected_phases)
    ax[1].set_ylabel("Corrected phase (rad)")

    if save_fig:
        if filepath is None:
            print(
                "Error: no filepath specified.  Either specify a filepath in the arguments or set save_fig=False"
            )
            return
        if filename is None:
            filename = data["series"] + ".pdf"
        else:
            filename = data["series"] + filename + ".pdf"
        plt.savefig(os.path.join(filepath, filename))
    plt.show()

    ## Plot unwrapped phases
    return


## Handle power conversion
def dBm_to_mW(dBm):
    return 10 ** (dBm / 10)


def mW_to_dBm(mW):
    return 10 * np.log10(mW)


def add_powers_dbm(a, b):
    return mW_to_dBm(dBm_to_mW(a) + dBm_to_mW(b))
