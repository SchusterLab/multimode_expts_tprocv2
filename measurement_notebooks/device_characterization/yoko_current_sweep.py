"""
yoko_current_sweep.py
Live 2D colorplot of resonator spectroscopy vs DC current (Yokogawa GS200).
Add to calibration_helpers.py or run as a standalone cell block.
"""

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from slab.instruments import YokogawaGS200
import time 


# ════════════════════════════════════════════════════════════
# YOKO HELPERS
# ════════════════════════════════════════════════════════════

def connect_yoko(address='10.108.30.37'):
    """Connect to Yokogawa and put it in current mode."""
    dcflux = YokogawaGS200(address=address)
    dcflux.set_output(True)
    dcflux.set_mode('current')
    print(f'Connected to Yokogawa at {address}')
    print(f'Current output: {1e3 * dcflux.get_current():.4f} mA')
    return dcflux





# default ramp speed — change this once at the top of your notebook if needed
DEFAULT_RAMP_SPEED_mA_PER_S = 0.01   # mA/s


def set_current_mA(dcflux, target_mA, ramp_speed_mA_per_s=DEFAULT_RAMP_SPEED_mA_PER_S):
    """
    Ramp Yokogawa output current to target_mA at ramp_speed_mA_per_s.
    Replaces the old instant set — always ramps safely.

    Parameters
    ----------
    dcflux              : YokogawaGS200
    target_mA           : float — target current [mA]
    ramp_speed_mA_per_s : float — ramp rate [mA/s], default 0.1 mA/s
    """
    current_now_mA = dcflux.get_current() * 1e3
    delta_mA       = abs(target_mA - current_now_mA)

    if delta_mA < 1e-6:
        return   # already there

    # number of steps: at least 2, step size ~0.001 mA
    step_size_mA = 0.01
    n_steps      = max(2, int(round(delta_mA / step_size_mA)))
    delay_s      = (delta_mA / ramp_speed_mA_per_s) / n_steps

    ramp = np.linspace(current_now_mA, target_mA, n_steps)
    for val in ramp:
        dcflux.set_current(val * 1e-3)
        time.sleep(delay_s)


def ramp_current_mA(dcflux, target_mA, ramp_speed_mA_per_s=DEFAULT_RAMP_SPEED_mA_PER_S):
    """
    Ramp Yokogawa output current to target_mA at ramp_speed_mA_per_s.
    Prints start/end current for visibility.

    Parameters
    ----------
    dcflux              : YokogawaGS200
    target_mA           : float — target current [mA]
    ramp_speed_mA_per_s : float — ramp rate [mA/s], default 0.1 mA/s
    """
    current_now_mA = dcflux.get_current() * 1e3
    delta_mA       = abs(target_mA - current_now_mA)
    eta_s          = delta_mA / ramp_speed_mA_per_s if ramp_speed_mA_per_s > 0 else 0

    print(f'Ramping {current_now_mA:.4f} → {target_mA:.4f} mA  '
          f'({ramp_speed_mA_per_s} mA/s, ETA {eta_s:.1f}s)')

    set_current_mA(dcflux, target_mA, ramp_speed_mA_per_s)
    print(f'Done. Current: {dcflux.get_current()*1e3:.4f} mA')


# ════════════════════════════════════════════════════════════
# MAIN SWEEP
# ════════════════════════════════════════════════════════════


def replot_yoko_sweep(result, plot_key='amps', cmap='RdBu_r', clim=None, figsize=(14, 10)):
    """
    Re-plot a completed yoko sweep from the returned result dict.
    Useful for adjusting colorscale after the fact.

    Parameters
    ----------
    result   : dict returned by do_yoko_res_sweep
    plot_key : str — only used for axis label, data is already in amp_matrix
    cmap     : str — colormap
    clim     : tuple or None — (vmin, vmax) color limits, e.g. (0, 30)
    figsize  : tuple
    """
    currents   = result['currents']
    freq_array = result['freq_array']
    amp_matrix = result['amp_matrix']
    rspecs     = result['rspecs']

    peak_freqs = [
        rspecs[i].data['xpts'][np.argmax(np.abs(np.array(rspecs[i].data[plot_key])))]
        for i in range(len(rspecs))
    ]

    fig, axes = plt.subplots(2, 1, figsize=figsize)

    im = axes[0].pcolormesh(freq_array, currents * 1e3, amp_matrix,
                            shading='auto', cmap=cmap)
    if clim is not None:
        im.set_clim(*clim)
    fig.colorbar(im, ax=axes[0], label=plot_key)
    axes[0].set_xlabel('Frequency (MHz)')
    axes[0].set_ylabel('DC Current (muA)')
    axes[0].set_title(f'Res spec vs Yoko current [{plot_key}]')

    axes[1].plot(currents * 1e3, peak_freqs, 'o-', color='C1', ms=4)
    axes[1].set_xlabel('DC Current (mA)')
    axes[1].set_ylabel('Peak frequency (MHz)')
    axes[1].set_title('Peak frequency vs current')
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()
    return fig