from .general_fitting import GeneralFitting
import numpy as np
import matplotlib.pyplot as plt

class Histogram(GeneralFitting):
    def __init__(self, data, span=None, verbose=True, active_reset=False, readout_per_round=None, threshold=None, config=None):
        super().__init__(data, readout_per_round, threshold, config)
        self.span = span
        self.verbose = verbose
        self.active_reset = self.cfg.expt.active_reset 
        self.results = {}

    def analyze(self, plot=True):
        if self.active_reset:
            Ig, Qg = self.filter_data_IQ(self.data['Ig'], self.data['Qg'], self.threshold)
            Ie, Qe = self.filter_data_IQ(self.data['Ie'], self.data['Qe'], self.threshold)
            plot_f = 'If' in self.data.keys()
            if plot_f:
                If, Qf = self.filter_data_IQ(self.data['If'], self.data['Qf'], self.threshold)
        else:
            Ig, Qg = self.data['Ig'], self.data['Qg']
            Ie, Qe = self.data['Ie'], self.data['Qe']
            plot_f = 'If' in self.data.keys()
            if plot_f:
                If, Qf = self.data['If'], self.data['Qf']

        numbins = 200
        xg, yg = np.median(Ig), np.median(Qg)
        xe, ye = np.median(Ie), np.median(Qe)
        if plot_f:
            xf, yf = np.median(If), np.median(Qf)

        if self.verbose:
            print('Unrotated:')
            print(f'Ig {xg} +/- {np.std(Ig)} \t Qg {yg} +/- {np.std(Qg)} \t Amp g {np.abs(xg+1j*yg)}')
            print(f'Ie {xe} +/- {np.std(Ie)} \t Qe {ye} +/- {np.std(Qe)} \t Amp e {np.abs(xe+1j*ye)}')
            if plot_f:
                print(f'If {xf} +/- {np.std(If)} \t Qf {yf} +/- {np.std(Qf)} \t Amp f {np.abs(xf+1j*yf)}')

        theta = -np.arctan2((ye - yg), (xe - xg))
        if plot_f:
            theta = -np.arctan2((ye - yf), (xe - xf))

        Ig_new = Ig * np.cos(theta) - Qg * np.sin(theta)
        Qg_new = Ig * np.sin(theta) + Qg * np.cos(theta)
        Ie_new = Ie * np.cos(theta) - Qe * np.sin(theta)
        Qe_new = Ie * np.sin(theta) + Qe * np.cos(theta)
        self.data['Ig_rot'] = Ig_new
        self.data['Qg_rot'] = Qg_new
        self.data['Ie_rot'] = Ie_new
        self.data['Qe_rot'] = Qe_new

        if plot_f:
            If_new = If * np.cos(theta) - Qf * np.sin(theta)
            Qf_new = If * np.sin(theta) + Qf * np.cos(theta)

        xg, yg = np.median(Ig_new), np.median(Qg_new)
        xe, ye = np.median(Ie_new), np.median(Qe_new)
        if plot_f:
            xf, yf = np.median(If_new), np.median(Qf_new)

        if self.verbose:
            print('Rotated:')
            print(f'Ig {xg} +/- {np.std(Ig_new)} \t Qg {yg} +/- {np.std(Qg_new)} \t Amp g {np.abs(xg+1j*yg)}')
            print(f'Ie {xe} +/- {np.std(Ie_new)} \t Qe {ye} +/- {np.std(Qe_new)} \t Amp e {np.abs(xe+1j*ye)}')
            if plot_f:
                print(f'If {xf} +/- {np.std(If_new)} \t Qf {yf} +/- {np.std(Qf_new)} \t Amp f {np.abs(xf+1j*yf)}')

        if self.span is None:
            self.span = (np.max(np.concatenate((Ie_new, Ig_new))) - np.min(np.concatenate((Ie_new, Ig_new)))) / 2
        midpoint = (np.max(np.concatenate((Ie_new, Ig_new))) + np.min(np.concatenate((Ie_new, Ig_new))))/2
        xlims = [midpoint-self.span, midpoint+self.span]

        ng, binsg = np.histogram(Ig_new, bins=numbins, range=xlims, density=True)
        ne, binse = np.histogram(Ie_new, bins=numbins, range=xlims, density=True)
        if plot_f:
            nf, binsf = np.histogram(If_new, bins=numbins, range=xlims, density=True)

        contrast = np.abs(((np.cumsum(ng) - np.cumsum(ne)) / (0.5 * ng.sum() + 0.5 * ne.sum())))
        tind = contrast.argmax()
        thresholds = [binsg[tind]]
        fids = [contrast[tind]]

        confusion_matrix = [np.cumsum(ng)[tind] / ng.sum(),
                            1 - np.cumsum(ng)[tind] / ng.sum(),
                            np.cumsum(ne)[tind] / ne.sum(),
                            1 - np.cumsum(ne)[tind] / ne.sum()]

        if plot_f:
            contrast = np.abs(((np.cumsum(ng) - np.cumsum(nf)) / (0.5 * ng.sum() + 0.5 * nf.sum())))
            tind = contrast.argmax()
            thresholds.append(binsg[tind])
            fids.append(contrast[tind])

            contrast = np.abs(((np.cumsum(ne) - np.cumsum(nf)) / (0.5 * ne.sum() + 0.5 * nf.sum())))
            tind = contrast.argmax()
            thresholds.append(binsg[tind])
            fids.append(contrast[tind])

        self.results = {
            'fids': fids,
            'thresholds': thresholds,
            'angle': theta * 180 / np.pi,
            'confusion_matrix': confusion_matrix
        }