from .general_fitting import GeneralFitting
from .fitting import *
import numpy as np
import matplotlib.pyplot as plt

class LengthRabiFitting(GeneralFitting):
    def __init__(self, data, fit=True, fitparams=None, normalize=[False, 'g_data', 'e_data'], vlines=None, title='length_rabi',
                    active_reset=False, readout_per_round=4, threshold=-4.0, fit_sin=False, config=None):
        super().__init__(data, readout_per_round, threshold, config)
        self.fit = fit
        self.fitparams = fitparams
        self.normalize = normalize
        self.vlines = vlines
        self.title = title
        self.active_reset = active_reset
        self.fit_sin = fit_sin
        self.results = {}

    def analyze(self, fitparams = None):
        if fitparams is None:
            fitparams = self.fitparams
        xlist = self.data['xpts'][0:-1]
        if self.active_reset:
            Ilist, Qlist = self.post_select_raverager_data(self.data)
        else:
            Ilist = self.data["avgi"][0:-1]
            Qlist = self.data["avgq"][0:-1]

        fit_func = fitter.fitsin if self.fit_sin else fitter.fitdecaysin
        func = fitter.sinfunc if self.fit_sin else fitter.decaysin

        p_avgi, pCov_avgi = fit_func(xlist, Ilist, fitparams=fitparams)
        p_avgq, pCov_avgq = fit_func(xlist, Qlist, fitparams=fitparams)

        self.data['fit_avgi'] = p_avgi
        self.data['fit_avgq'] = p_avgq
        self.data['fit_err_avgi'] = pCov_avgi
        self.data['fit_err_avgq'] = pCov_avgq

        self.results = {
            'fit_avgi': p_avgi,
            'fit_avgq': p_avgq,
            'fit_err_avgi': pCov_avgi,
            'fit_err_avgq': pCov_avgq
        }

    def display(self, return_fit_params=False, title_str='Length Rabi', vlines=None, **kwargs):
        """
        Displays the I and Q data with optional fit overlays and vertical markers.
        Plots the averaged I and Q data as a function of pulse length, optionally overlaying fitted curves and vertical lines indicating π and π/2 pulse lengths. Additional vertical lines can be added via `vlines`. The plot is shown and optionally saved to a file.
        Args:
            return_fit_params (bool, optional): If True, returns fit parameters and data arrays. Defaults to False.
            title_str (str, optional): Title string for the plot and output filename. Defaults to 'Length Rabi'.
            vlines (list, optional): List of x-values to draw additional vertical lines.
            **kwargs: Additional keyword arguments (currently unused).
        Returns:
            tuple: If `return_fit_params` is True, returns a tuple containing:
                - fit_avgi (array-like): Fitted parameters for the I data.
                - fit_err_avgi (array-like): Fit errors for the I data.
                - xlist (array-like): X data points used for fitting.
                - Ilist (array-like): Averaged I data points.
        Side Effects:
            - Displays the plot using matplotlib.
            - Saves the plot as a PNG file with a name based on `title_str`.
        Notes:
            - Requires `self.data` to contain keys: 'xpts', 'avgi', 'avgq', 'fit_avgi', 'fit_avgq'.
            - Uses `self.fit`, `self.fit_sin`, `self.title`, `self.vlines`, and `self.results` attributes.
            - The fitting function is selected based on `self.fit_sin`.
            - The plot is saved using `self.save_plot`.
        """
        
        xlist = np.array(self.data['xpts'][0:-1])
        xpts_ns = np.array(self.data['xpts']) * 1e3
        Ilist = self.data["avgi"][0:-1]
        Qlist = self.data["avgq"][0:-1]

        func = fitter.sinfunc if self.fit_sin else fitter.decaysin

        fig = plt.figure(figsize=(10, 8))

        axi = plt.subplot(211, title=self.title, ylabel="I [adc level]")
        plt.plot(xpts_ns[1:-1], Ilist[1:], 'o-')
        if self.fit:
            p = self.data['fit_avgi'] # yscale, freq, phase_deg, decay, y0, x0 = p
            plt.plot(xpts_ns[0:-1], func(xlist, *p))
            pi_length, pi2_length = self._calculate_pi_lengths(p)
            self.results['pi_length'] = pi_length
            self.results['pi2_length'] = pi2_length
            plt.axvline(pi_length * 1e3, color='0.2', linestyle='--', label='pi')
            plt.axvline(pi2_length * 1e3, color='0.2', linestyle='--', label='pi/2')
            # Draw additional vlines if provided (argument takes precedence)
            vlines_to_draw = vlines if vlines is not None else self.vlines
            if vlines_to_draw:
                for vline in vlines_to_draw:
                    plt.axvline(vline, color='r', ls='--')
                    print(f'vline: {vline} ns')

        axq = plt.subplot(212, xlabel="Pulse length [ns]", ylabel="Q [adc levels]")
        plt.plot(xpts_ns[1:-1], Qlist[1:], 'o-')
        if self.fit:
            p = self.data['fit_avgq']
            plt.plot(xpts_ns[0:-1], func(xlist, *p))
            pi_length, pi2_length = self._calculate_pi_lengths(p)
            plt.axvline(pi_length * 1e3, color='0.2', linestyle='--', label='pi')
            plt.axvline(pi2_length * 1e3, color='0.2', linestyle='--', label='pi/2')
            vlines_to_draw = vlines if vlines is not None else self.vlines
            if vlines_to_draw:
                for vline in vlines_to_draw:
                    plt.axvline(vline, color='r', ls='--')

        plt.tight_layout()
        plt.legend()
        plt.show()

        if return_fit_params:
            return self.results['fit_avgi'], self.results['fit_err_avgi'], xlist, Ilist
        # save figure 
        filename = title_str.replace(' ', '_').replace(':', '') + '.png'
        self.save_plot(fig, filename=filename) 

    def _calculate_pi_lengths(self, p):
        """
        Calculate the π (pi) and π/2 (pi/2) pulse lengths based on input parameters.

        Args:
            p (list or tuple): A sequence of parameters where:
                - p[1]: Frequency (Hz)
                - p[2]: Phase in degrees

        Returns:
            tuple: A tuple containing:
                - pi_length (float): The calculated π pulse length.
                - pi2_length (float): The calculated π/2 pulse length.

        Notes:
            - The phase (p[2]) is normalized to the range [-180, 180] degrees.
            - The calculation assumes a specific relationship between phase, frequency, and pulse length.
            - Prints the calculated π and π/2 lengths for debugging purposes.
        """
        if p[2] > 180:
            p[2] -= 360
        elif p[2] < -180:
            p[2] += 360
        if p[2] < 0:
            pi_length = (1 / 2 - p[2] / 180) / 2 / p[1]
        else:
            pi_length = (3 / 2 - p[2] / 180) / 2 / p[1]
        T = 1/p[1]# TIME PERIOD
        pi2_length = pi_length - T/4
        print('p1:', p[1])
        print('p2:', p[2])
        print('Pi length:', pi_length)
        print('Pi/2 length:', pi2_length)
        return pi_length, pi2_length