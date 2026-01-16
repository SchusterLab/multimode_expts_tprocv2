from .general_fitting import GeneralFitting
from .fitting import *
import numpy as np
import matplotlib.pyplot as plt

class SpectroscopyFitting(GeneralFitting):
    def __init__(self, data, signs=[1, 1, 1], config=None):
        super().__init__(data, readout_per_round=1, threshold=-4.0, config=config)
        self.signs = signs


    def analyze(self, data_list=None, fit=True):
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
                data['fit_amps'], data['fit_err_amps'] = fitlor(xdata, self.signs[0] * data['amps'][1:-1])
                data['fit_avgi'], data['fit_err_avgi'] = fitlor(xdata, self.signs[1] * data['avgi'][1:-1])
                data['fit_avgq'], data['fit_err_avgq'] = fitlor(xdata, self.signs[2] * data['avgq'][1:-1])
            
            modified_data_list.append(data)
        
        return modified_data_list

    def display(self, title='Qubit Spectroscopy', vlines=None, fit=True, data_list=None):
        """
        Display spectroscopy data with optional fitting.
        
        Parameters:
        -----------
        title : str
            Title for the plot
        vlines : list, optional
            List of x-values to draw vertical lines
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
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
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
                    ax.axvline(vline, c='k', ls=':', alpha=0.5, linewidth=1)

            ax.legend(loc='best', fontsize='small', framealpha=0.7)

        axes[-1].set_xlabel("Pulse Frequency (MHz)")
        plt.tight_layout()
        plt.show()
        
        # Save figure
        filename = title.replace(' ', '_').replace(':', '') + '.png'
        self.save_plot(fig, filename=filename)
