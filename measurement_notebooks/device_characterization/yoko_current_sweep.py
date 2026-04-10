"""
yoko_current_sweep.py
Live 2D colorplot of resonator spectroscopy vs DC current (Yokogawa GS200).
Add to calibration_helpers.py or run as a standalone cell block.
"""

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from slab.instruments import YokogawaGS200


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


def set_current_mA(dcflux, current_mA):
    """Set Yokogawa output current in mA."""
    dcflux.set_current(current_mA * 1e-3)


def ramp_current_mA(dcflux, target_mA, n_steps=20, delay_s=0.05):
    """
    Ramp current slowly from current value to target.
    Safer than jumping directly — avoids flux jumps.
    """
    import time
    current_now_mA = dcflux.get_current() * 1e3
    ramp = np.linspace(current_now_mA, target_mA, n_steps)
    for val in ramp:
        dcflux.set_current(val * 1e-3)
        time.sleep(delay_s)
    print(f'Ramped to {target_mA:.4f} mA')


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