from slab import Experiment
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm_notebook as tqdm
from qick.asm_v2 import QickSweep1D
from datetime import datetime
from .MM_base import MMBase
"""
QICK Experiment Module

This module provides classes for quantum experiments using the QICK (Quantum Instrumentation Control Kit) framework.
It extends the base Experiment class with specialized functionality for:
- Running quantum experiments on QICK hardware
- Acquiring and analyzing measurement data
- Fitting experimental results to theoretical models
- Visualizing and saving experiment data

Adapted from Shannon's Experiments

These classes work with the QickProgram classes to implement complete quantum experiments.
"""


class MMExperiment(Experiment, MMBase):
    """
    Base class for quantum experiments using the QICK framework.

    This class extends the Experiment base class to provide specialized functionality
    for quantum experiments on QICK hardware. It handles experiment configuration,
    data acquisition, analysis, visualization, and data storage.

    The class is designed to be extended by specific experiment implementations
    that override methods like acquire(), analyze(), and display() to implement
    specific experiment types (e.g., T1, T2, Rabi oscillations).
    """

    def __init__(
        self,
        cfg_dict=None,
        prefix="QickExp",
        fname=None,
        progress=None,
    ):
        """
        Initialize the QickExperiment with hardware configuration and experiment parameters.

        Args:
            cfg_dict: Dictionary containing configuration parameters including:
                - soc: System-on-chip configuration
                - expt_path: Path for saving experiment data
                - cfg_file: Configuration file path
                - im: Instrument manager instance
            prefix: Prefix for saved data files
            progress: Whether to show progress bars during execution
            qi: Qubit index to use for the experiment
            check_params: Whether to check for unexpected parameters (default True)
        """
        soccfg = cfg_dict["soccfg"]
        path = cfg_dict["expt_path"] 
        config_file = cfg_dict["cfg_file"]
        im = cfg_dict["im"]
        # cfg = AttrDict(config_file)
        # print(cfg['aliases'])
       
        # self.soc = im[cfg['aliases']['soc']]
        # self.set_all_filters(cfg)
        super().__init__(
            soccfg=soccfg,
            path=path,
            prefix=prefix,
            fname=fname,
            config_file=config_file,
            progress=progress,
            im=im,
        )
        # Store the check_params parameter for use in child classes
        # self._check_params = check_params
        
        # print(self.cfg)
        # print(self.cfg.aliases.soc)
        # print(im[self.cfg.aliases.soc])
        # print(self.cfg.hw.soc.dacs.readout.ch[0])
        # print(self.cfg.aliases.soc)
        self.soc = im[self.cfg.aliases.soc]
        self.parse_config()
        # print(self.soc)
        self.set_all_filters_()
        # print("all filters set")
# 
    
    def set_all_filters_(self):
        '''Set filters for all channels'''
        
        def set_dac_filter_and_att(ch, ftype, fc, bw=None, att1=0, att2=0): 
            """only bandpass filter"""
            self.soc.rfb_set_gen_filter(ch, fc=fc/1000, ftype=ftype, bw=bw)
            self.soc.rfb_set_gen_rf(ch, att1, att2)
            # print(f"Set DAC channel {ch} filter: ftype={ftype}, fc={fc} GHz, bw={bw} GHz, att1={att1} dB, att2={att2} dB")
        
        def set_adc_filter_and_att(ch, ftype, fc, bw=None, att=0    ):
            """only bandpass filter"""
            self.soc.rfb_set_ro_filter(ch, fc=fc/1000, ftype=ftype, bw=bw)
            self.soc.rfb_set_ro_rf(ch, att)
            # print(f"Set ADC channel {ch} filter: ftype={ftype}, fc={fc} GHz, bw={bw} GHz, att={att} dB")
        # Readout 
        _= set_dac_filter_and_att(self.res_ch, 
                                                             self.res_ftype, 
                                                            self.res_fc,
                                                            self.res_bw,
                                                            self.res_att[0],
                                                            self.res_att[1])
        _= set_adc_filter_and_att(self.adc_ch, 
                                                             self.adc_ftype,
                                                                self.adc_fc,
                                                                self.adc_bw,
                                                                self.adc_att)
        
        # qubit 
        _= set_dac_filter_and_att(self.qubit_ch, 
                                                             self.qubit_ftype, 
                                                            self.qubit_fc,
                                                            self.qubit_bw,
                                                            self.qubit_att[0],
                                                            self.qubit_att[1])
        # manipulate 
        _= set_dac_filter_and_att(self.manipulate_ch, 
                                                             self.manipulate_ftype, 
                                                            self.manipulate_fc,
                                                            self.manipulate_bw,
                                                            self.manipulate_att[0],
                                                            self.manipulate_att[1])
        

    def set_filter_custom(self, gen_ch, ro_ch, lpf, hpf, state=0):
        soc = self.soc
        sw = 0xc0 + (hpf<<3) + lpf
        filt_bits = (state<<4) + state
        rfb_ch = soc.gens[gen_ch].rfb_ch
        with soc.board_sel.enable_context(rfb_ch.card_num):
            rfb_ch.filter.write_reg('WR0_SW', sw)
            rfb_ch.filter.write_reg('WR0_FILTER', filt_bits)
            
        if ro_ch is None:
            return None

        rfb_ch = soc.avg_bufs[ro_ch].rfb_ch
        with soc.board_sel.enable_context(rfb_ch.card_num):
            rfb_ch.filter.write_reg('WR0_SW', sw)
            rfb_ch.filter.write_reg('WR0_FILTER', filt_bits)

    def acquire(
        self, prog_name, progress=True, get_hist=True, single=True, compact=False
    ):
        """
        Acquire measurement data by running the specified quantum program.

        This method:
        1. Creates an instance of the specified program
        2. Runs the program on the QICK hardware
        3. Processes the raw measurement data
        4. Optionally generates histograms of measurement results

        Args:
            prog_name: Class reference to the QickProgram to run
            progress: Whether to show progress bar during acquisition
            get_hist: Whether to generate histogram of measurement results

        Returns:
            Dictionary containing measurement data including:
            - xpts: Swept parameter values
            - avgi/avgq: I and Q quadrature data
            - amps/phases: Amplitude and phase data
            - bin_centers/hist: Histogram data (if get_hist=True)
        """
        # Set appropriate final delay based on whether active reset is enabled
       
        final_delay = self.cfg.device.readout.final_delay
        # print('final delay: ', final_delay)

        # Create program instance
        # print(self.cfg)
        prog = prog_name(
            soccfg=self.soccfg,
            final_delay=final_delay,
            cfg=self.cfg,
        )
    
        # Record start time
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        current_time = current_time.encode("ascii", "replace")

        # Run the program and acquire data
        iq_list = prog.acquire(
            self.im[self.cfg.aliases.soc],
            rounds=self.cfg.expt.rounds,
            threshold=None,
            #load_pulses=True,
            progress=progress,
        )

        

        # Process I/Q data to get amplitude and phase
        # Shape: Readout channels / Readouts in Program / Loops / I and Q
        iq = iq_list[0][0]
        amps = np.abs(iq.dot([1, 1j]))
        phases = np.angle(iq.dot([1, 1j]))
        avgi = np.squeeze(iq[..., 0])
        avgq = np.squeeze(iq[..., 1])

        # Genera/= self.make_hist(prog, single=single)

        # Compile data dictionary
        data = {}

        # Add each swept parameter as a separate entry in the data dictionary
        self.add_swept_parameters_to_data(prog, data)

        # Add other data based on the compact flag
        if compact:
            data.update({
                "avgi": avgi,
                "avgq": avgq,
                "start_time": current_time,
            })
        else:
            data.update({
                "avgi": avgi,
                "avgq": avgq,
                "amps": amps,
                "phases": phases,
                "start_time": current_time,
            })

        # Convert all data to numpy arrays
        for key in data:
            data[key] = np.array(data[key])
        self.data = data
        # print(self.data)
        # Clean up configuration after sweep
        # only call uf self.param or self.sweep_other_params exist
        
        self.clean_config_after_sweep()

        return data

    
    def analyze(
        self,
        fitfunc=None,
        fitterfunc=None,
        data=None,
        fit=True,
        use_i=None,
        get_hist=True,
        verbose=True,
        inds=None,
        **kwargs,
    ):
        """
        Analyze measurement data by fitting to theoretical models.

        This method:
        1. Fits the data to the specified model function
        2. Determines the best fit parameters and error estimates
        3. Calculates goodness-of-fit metrics (R²)
        4. Optionally scales data based on histogram analysis

        Args:
            fitfunc: Function to fit data to (e.g., exponential decay)
            fitterfunc: Function that performs the fitting
            data: Data dictionary to analyze (uses self.data if None)
            fit: Whether to perform fitting
            use_i: Whether to use I quadrature for fitting (auto-determined if None)
            get_hist: Whether to generate histogram and scale data
            **kwargs: Additional arguments passed to the fitter

        Returns:
            Data dictionary with added fit results
        """
        if data is None:
            data = self.data
        # Remove the first and last points from fit to avoid edge effects

        # Determine which data sets to fit
        ydata_lab = ["amps", "avgi", "avgq"]

        # Scale data based on histogram if requested
        if get_hist:
            self.scale_ge()
            ydata_lab.append("scale_data")

        # Perform fits on each data set (amplitude, I, Q)
        for i, ydata in enumerate(ydata_lab):
            # Use standard curve_fit via fitterfunc
            (
                data["fit_" + ydata],
                data["fit_err_" + ydata],
                data["fit_init_" + ydata],
            ) = fitterfunc(data["xpts"][1:-1], data[ydata][1:-1], **kwargs)

        # Determine which fit is best (I, Q, or amplitude)
        if use_i is None:
            use_i = self.cfg.device.qubit.tuned_up[self.cfg.expt.qubit[0]]
        if use_i:
            # For tuned-up qubits, use I quadrature by default
            i_best = "avgi"
            fit_pars = data["fit_avgi"]
            fit_err = data["fit_err_avgi"]
        else:
            # Otherwise, determine best fit automatically
            fit_pars, fit_err, i_best = fitter.get_best_fit(data, fitfunc)

        # Calculate goodness-of-fit (R²)
        r2 = fitter.get_r2(data["xpts"][1:-1], data[i_best][1:-1], fitfunc, fit_pars)
        data["r2"] = r2
        data["best_fit"] = fit_pars
        i_best = i_best.encode("ascii", "ignore")
        data["i_best"] = i_best

        if inds is None:
            inds = np.arange(len(fit_err))

        fit_err = fit_err[inds]
        # Calculate relative parameter errors
        fit_pars = np.array(fit_pars)
        data["fit_err_par"] = np.sqrt(np.diag(fit_err)) / fit_pars[inds]
        fit_err = np.mean(np.abs(data["fit_err_par"]))
        data["fit_err"] = fit_err

        # Print fit quality metrics
        if verbose:
            # print(f"R2:{r2:.3f}\tFit par error:{fit_err:.3f}\t Best fit:{i_best}")
            pass

        self.get_status()

        return data

    def display(
        self,
        data=None,
        ax=None,
        plot_all=False,
        title="",
        xlabel="",
        fit=True,
        show_hist=False,
        rescale=False,
        fitfunc=None,
        caption_params=[],
        debug=False,
        vlines =None,
        ylim = None,
        **kwargs,
    ):
        """
        Display measurement results with optional fit curves.

        This method creates plots showing the measurement data and optional fit curves.
        It can display:
        - Single quadrature (I) or all quadratures (I, Q, amplitude)
        - Fit curves with parameter values in the legend
        - Histograms of single-shot measurements
        - Rescaled data based on histogram analysis

        Args:
            data: Data dictionary to display (uses self.data if None)
            ax: Matplotlib axis to plot on (creates new figure if None)
            plot_all: Whether to plot all quadratures (I, Q, amplitude)
            title: Plot title
            xlabel: X-axis label
            fit: Whether to show fit curves
            show_hist: Whether to show histogram plot
            rescale: Whether to show rescaled data (0-1 probability)
            fitfunc: Function used for fitting
            caption_params: List of parameters to display in the legend
            debug: Whether to show debug information (initial guess)
            **kwargs: Additional arguments for plotting
        """
        if data is None:
            data = self.data

        # Determine whether to save the figure
        if ax is None:
            save_fig = True
        else:
            save_fig = False

        # Configure plot layout based on what to display
        if plot_all:
            # Create 3-panel figure for amplitude, I, and Q
            fig, ax = plt.subplots(3, 1, figsize=(7, 9.5))
            fig.suptitle(title)
            ylabels = ["Amplitude (ADC units)", "I (ADC units)", "Q (ADC units)"]
            ydata_lab = ["amps", "avgi", "avgq"]
        else:
            # Create single panel figure
            if ax is None:
                fig, a = plt.subplots(1, 1, figsize=(7, 4))
                ax = [a]
            if rescale:
                # Show rescaled data (0-1 probability)
                ylabels = ["Excited State Probability"]
                ydata_lab = ["scale_data"]
            else:
                # Show raw I quadrature
                ylabels = ["I (ADC units)"]
                ydata_lab = ["avgi"]
            ax[0].set_title(title)

        # Plot each data set
        for i, ydata in enumerate(ydata_lab):
            # Plot data points (excluding first and last points)
            ax[i].plot(data["xpts"][1:-1], data[ydata][1:-1], "o-")

            # Add fit curve if requested
            if fit:
                p = data["fit_" + ydata]  # Fit parameters
                pCov = data["fit_err_" + ydata]  # Covariance matrix

                # Create caption with fit parameters
                caption = ""
                for j in range(len(caption_params)):
                    if j > 0:
                        caption += "\n"
                    if isinstance(caption_params[j]["index"], int):
                        # Display parameter value and error
                        ind = caption_params[j]["index"]
                        caption += caption_params[j]["format"].format(
                            val=(p[ind]), err=np.sqrt(pCov[ind, ind])
                        )
                    else:
                        # Display derived parameter
                        var = caption_params[j]["index"]
                        caption += caption_params[j]["format"].format(
                            val=data[var + "_" + ydata]
                        )

                # Plot fit curve
                ax[i].plot(
                    data["xpts"][1:-1], fitfunc(data["xpts"][1:-1], *p), label=caption
                )
                ax[i].legend()
            # Plot vertical lines if provided
            if vlines is not None:
                colors = ['r','g','b','c','m','y']
                for idx, vline in enumerate(vlines):
                    ax[i].axvline(x=vline,label = str(vline), color=colors[idx%len(colors)], linestyle='--')
                ax[i].legend()

            # Set axis labels
            ax[i].set_ylabel(ylabels[i])
            ax[i].set_xlabel(xlabel)
            
            # y limits 
            if ylim is not None:
                ax[i].set_ylim(ylim)

            # Show initial guess if in debug mode
            if debug:
                pinit = data["fit_init_" + ydata]
                # print(pinit)
                ax[i].plot(
                    data["xpts"], fitfunc(data["xpts"], *pinit), label="Initial Guess"
                )

        # Show histogram if requested
        if show_hist:
            fig2, ax = plt.subplots(1, 1, figsize=(3, 3))
            ax.plot(data["bin_centers"], data["hist"], "o-")
            # Try to plot histogram fit if available
            try:
                ax.plot(
                    data["bin_centers"],
                    helpers.two_gaussians_decay(data["bin_centers"], *data["hist_fit"]),
                    label="Fit",
                )
            except:
                pass
            ax.set_xlabel("I [ADC units]")
            ax.set_ylabel("Probability")

        # Save figure if created in this method
        if save_fig:
            imname = self.fname.split("\\")[-1]
            fig.tight_layout()
            fig.savefig(
                self.fname[0 : -len(imname)] + "images\\" + imname[0:-3] + ".png"
            )
            plt.show()

    def make_hist(self, prog, single=True):
        """
        Generate histogram of single-shot measurement results.

        This method collects individual measurement shots and creates a histogram
        of the I quadrature values, which can be used for state discrimination
        and readout fidelity analysis.

        Args:
            prog: QickProgram instance to collect shots from
            single: Whether to collect shots for the entire experiment together, or separately for each point in the sweep

        Returns:
            Tuple of (bin_centers, hist) containing histogram data
        """
        # Get I/Q offset from configuration
        offset = self.soccfg._cfg["readouts"][self.cfg.expt.qubit_chan]["iq_offset"]

        # Collect individual measurement shots
        shots_i, shots_q = prog.collect_shots(offset=offset, single=single)

        # Create histogram with 60 bins
        # sturges_bins = int(np.ceil(np.log2(len(shots_i)) + 1))
        if single:
            hist, bin_edges = np.histogram(shots_i, bins=60, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        else:
            hist = []
            bin_centers = []
            for i in range(len(shots_i)):
                hist0, bin_edges0 = np.histogram(shots_i[i], bins=60, density=True)
                hist.append(hist0)
                bin_centers.append((bin_edges0[:-1] + bin_edges0[1:]) / 2)
        return bin_centers, hist

    def qubit_run(
        self,
        qi=0,
        progress=True,
        analyze=True,
        display=True,
        save=True,
        print=False,
        min_r2=0.1,
        max_err=1,
        disp_kwargs=None,
        **kwargs,
    ):
        # Configure active reset if enabled
        if self.cfg.expt.active_reset:
            self.configure_reset()

        # For untuned qubits, show all data points by default
        if not self.cfg.device.qubit.tuned_up[qi] and disp_kwargs is None:
            disp_kwargs = {"plot_all": True}
            # For untuned qubits, show all data points by default
        if (
            self.cfg.device.readout.rescale[qi]
            or disp_kwargs is not None
            and "rescale" in disp_kwargs
        ):
            disp_kwargs = {"rescale": True}

        # Run the experiment if go=True
        if print:
            self.print()
        else:
            self.run(
                analyze=analyze,
                display=display,
                save=save,
                progress=progress,
                min_r2=min_r2,
                max_err=max_err,
                disp_kwargs=disp_kwargs,
            )

    def run(
        self,
        progress=True,
        analyze=True,
        display=True,
        save=True,
        min_r2=0.1,
        max_err=1,
        disp_kwargs=None,
        **kwargs,
    ):
        """
        Run the complete experiment workflow.

        This method executes the full experiment sequence:
        1. Acquire data
        2. Analyze results
        3. Display plots
        4. Save data to disk
        5. Determine if the experiment was successful

        Args:
            progress: Whether to show progress bar during acquisition
            analyze: Whether to perform data analysis
            display: Whether to display results
            save: Whether to save data to disk
            min_r2: Minimum R² value for acceptable fit
            max_err: Maximum error for acceptable fit
            disp_kwargs: Display options dictionary
            **kwargs: Additional arguments passed to the analyze method
        """

        # Set default values for fit quality thresholds
        if min_r2 is None:
            min_r2 = 0.1
        if max_err is None:
            max_err = 1
        if disp_kwargs is None:
            disp_kwargs = {}
            # These might be rescale, show_hist, plot_all. Eventually, want to put plot_all into the config.

        # Execute experiment workflow
        data = self.acquire(progress)
        if analyze:
            data = self.analyze(data, **kwargs)
        if save:
            self.save_data(data)
        if display:
            self.display(data, **disp_kwargs)

    def save_data(self, data=None, verbose=True):
        """
        Save experiment data to disk.

        Args:
            data: Data dictionary to save (uses self.data if None)
            verbose: Whether to print save confirmation

        Returns:
            Filename where data was saved
        """
        if verbose:
            # print(f"Saving {self.fname}")
            # print(data)
            pass
        super().save_data(data=data)
        # print('Finished saving data')
        return self.fname

    def print(self):
        """
        Print out the experimental config
        """
        for key, value in self.cfg.expt.items():
            # print(f"{key}: {value}")
            pass

    def get_status(self, max_err=1, min_r2=0.1):
        # Determine if experiment was successful based on fit quality
        if (
            "fit_err" in self.data
            and "r2" in self.data
            and self.data["fit_err"] < max_err
            and self.data["r2"] > min_r2
        ):
            self.status = True
        elif "fit_err" not in self.data or "r2" not in self.data:
            # No fit performed, can't determine status
            pass
        else:
            # print("Fit failed")
            self.status = False

    def get_params(self, prog):
        """
        Get swept parameter values from the program.

        This method extracts the values of the parameter being swept in the experiment,
        either a pulse parameter (e.g., amplitude, frequency) or a time parameter
        (e.g., delay, pulse length).
        self.param needs to have fields set:
        - param_type: "pulse" or "time"
        - label: Label of the parameter to extract [listed in the program]
        - param: Name of the parameter to extract (freq, gain, total_length, t)

        Args:
            prog: QickProgram instance to get parameters from

        Returns:
            Array of parameter values
        """
        # if not hasattr(self, "params"):
        #     self.params = {"Null": self.param}
        # if len(self.sweep_param.keys())==1: 
        #     self.params = {"Null": self.param}
        
        xpts_for_all_params = {}
            
        for param_tag, param in self.sweep_param.items():
            
            if param["param_type"] == "pulse":
                # Extract pulse parameter (amplitude, frequency, etc.)
                xpts = prog.get_pulse_param(
                    param["label"], param["param"], as_array=True
                )
            else:
                # Extract time parameter (delay, pulse length, etc.)
                xpts = prog.get_time_param(
                    param["label"], param["param"], as_array=True
                )
            xpts_for_all_params[param_tag] = xpts
        # print(xpts_for_all_params)
            
        return xpts_for_all_params

    # def check_params(self, params_def):
    #     if self._check_params:
    #         unexpected_params = set(self.cfg.expt.keys()) - set(params_def.keys())
    #         if unexpected_params:
    #             print(f"Unexpected parameters found in params: {unexpected_params}")

    def configure_reset(self):
        qi = self.cfg.expt.qubit[0]
        # we may want to put these params in the config.
        params_def = dict(
            threshold_v=self.cfg.device.readout.threshold[qi],
            read_wait=0.1,
            extra_delay=0.2,
        )
        self.cfg.expt = {**params_def, **self.cfg.expt}
        # this number should be changed to be grabbed from soc
        self.cfg.expt["threshold"] = int(
            self.cfg.expt["threshold_v"]
            * self.cfg.device.readout.readout_length[qi]
            / 0.0032552083333333335
        )

    def get_freq(self, fit=True):
        """
        Provide correct frequency if mixers are in use, for LO coming from QICK or external source
        """
        freq_offset = 0
        q = self.cfg.expt.qubit[0]
        if "mixer_freq" in self.cfg.hw.soc.dacs.readout:
            freq_offset += self.cfg.hw.soc.dacs.readout.mixer_freq[q]
        # lo_freq is in readout; used for signal core.
        if "lo_freq" in self.cfg.hw.soc.dacs.readout:
            freq_offset += self.cfg.hw.soc.dacs.readout.lo_freq[q]
        if "lo" in self.cfg.hw.soc and "mixer_freq" in self.cfg.hw.soc.lo:
            freq_offset += self.cfg.hw.soc.lo.mixer_freq[q]

        self.data["freq"] = freq_offset + self.data["xpts"]
        self.data["freq_offset"] = freq_offset
        # if fit:
        #     self.data["freq_fit"] = self.data["fit"]
        #     self.data["freq_init"] = self.data["init"]
        #     self.data["freq_fit"][0] = freq_offset + self.data["fit"][0]
        #     self.data["freq_init"][0] = freq_offset + self.data["init"][0]

    def scale_ge(self):
        """
        Scale g->0 and e->1 based on histogram data"""

        hist = self.data["hist"]
        bin_centers = self.data["bin_centers"]
        v_rng = np.max(bin_centers) - np.min(bin_centers)

        p0 = [
            0.5,
            np.min(bin_centers) + v_rng / 3,
            0.5,
            v_rng / 10,
            np.max(bin_centers) - v_rng / 3,
        ]
        try:
            popt, pcov = curve_fit(helpers.two_gaussians, bin_centers, hist, p0=p0)
            vg = popt[1]
            ve = popt[4]
            dv = ve - vg
            # if (
            #     "tm" in self.cfg.device.readout
            #     and self.cfg.device.readout.tm[self.cfg.expt.qubit[0]] != 0
            # ):
            #     tm = self.cfg.device.readout.tm[self.cfg.expt.qubit[0]]
            #     sigma = self.cfg.device.readout.sigma[self.cfg.expt.qubit[0]]
            #     p0 = [popt[0], vg, ve]
            #     popt, pcov = curve_fit( #@IgnoreException
            #         lambda x, mag_g, vg, ve: helpers.two_gaussians_decay(
            #             x, mag_g, vg, ve, sigma, tm
            #         ),
            #         bin_centers,
            #         hist,
            #         p0=p0,
            #     )
            #     popt = np.concatenate((popt, [sigma, tm]))

            # dv = popt[2] - popt[1]
            self.data["scale_data"] = (self.data["avgi"] - popt[1]) / dv
            self.data["hist_fit"] = popt
        except:
            self.data["scale_data"] = self.data["avgi"]

    def add_swept_parameters_to_data(self, prog, data):
        """
        Add each swept parameter as a separate entry in the data dictionary.

        Args:
            prog: The program instance to extract parameters from.
            data: The data dictionary to update with swept parameters.
        """
        # Get swept parameter values
        xpts_for_all_params = self.get_params(prog)
        for param_name, xpts in xpts_for_all_params.items():
            if len(self.sweep_param.keys()) == 1:
                param_full_name = "xpts"
            else:
                param_full_name = f"xpts_{param_name}"
            data[param_full_name] = xpts
            
    def initialize_sweep_variables(self, params = None):
        """Initialize sweep variables for the experiment."""
        # print('entering initailoize sweep vars ')
        
        for param_name, param_values in self.sweep_param.items():
            # print('param name: ', param_name)
            # if there is a key parent , then it should be 
            # self.cfg.expt[parent][param_name] = QickSweep1D(..)
            if "parent_dict" in param_values:
                # print('-----------------')
                # print(param_values)
                # print(self.cfg.expt)
                parent_name = param_values['parent_dict']
                parent_dict = dict(self.cfg.expt[parent_name]) # since attr dict makes dict objects immutable 
                parent_dict[param_name] = QickSweep1D(
                    param_name, param_values.start, param_values.start + param_values.step * param_values.expts
                )
                # print(parent_dict)
                self.cfg.expt[parent_name] = parent_dict
                # print(self.cfg.expt)
                # print('-----------------')
                # print(param_values)
                # print(self.cfg.expt)
            else:
                # print()
                self.cfg.expt[param_name] = QickSweep1D(
                    param_name, param_values.start, param_values.start + param_values.step * param_values.expts
                )
           
            
        self.cfg.expt.sweep_param = self.sweep_param

            
    def clean_config_after_sweep(self): 
        """
        Remove temporary sweep parameters after the experiment.
        """
        if self.sweep_param:  # Replace sweep_other_param with sweep_param
            for param_name in self.sweep_param.keys():
                param_values = self.sweep_param[param_name]
                if "parent_dict" in param_values:
                    # print('-----------------')
                    # print(param_values)
                    # print(self.cfg.expt)
                    parent_name = param_values['parent_dict']
                    parent_dict = dict(self.cfg.expt[parent_name]) # since attr dict makes dict objects immutable 
                    parent_dict[param_name] = None
                    # print(parent_dict)
                    self.cfg.expt[parent_name] = parent_dict
                    # print(self.cfg.expt)
                    # print('-----------------')
                    # print(param_values)
                    # print(self.cfg.expt)
                else:
                    self.cfg.expt[param_name] = None

    def combine_sweep_params(self, primary_params, additional_params):
        """
        Combine two dictionaries of sweep parameters, ensuring all keys are included.

        Args:
            primary_params (dict): The primary sweep parameters.
            additional_params (dict): Additional sweep parameters to merge.

        Returns:
            dict: Combined dictionary of sweep parameters.
        """
        combined_params = primary_params.copy()
        if additional_params:
            for key, value in additional_params.items():
                if key not in combined_params:
                    combined_params[key] = value
        return combined_params


