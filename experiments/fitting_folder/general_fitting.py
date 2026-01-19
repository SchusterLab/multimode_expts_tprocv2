import datetime
import os
from copy import deepcopy

import lmfit
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit, least_squares
from scipy.signal import find_peaks


class GeneralFitting:
    def __init__(self, data, readout_per_round=None, threshold=None, config=None, station = None):
        self.cfg = config
        self.data = data
        if readout_per_round is None:
            readout_per_round = 4
        # if threshold is None:
        #     threshold = self.cfg.device.readout.threshold
        self.readout_per_round = readout_per_round
        self.threshold = threshold
        self.station = station
        
    

    def bin_ss_data(self, conf=True):
        '''
        This function takes config saved single shot parameters, applies the angle correction and threshold to the main data of the experiment
        bins it into counts_g and counts_e
        '''
        temp_data = self.data
        rounds = self.cfg['expt']['rounds']
        reps = self.cfg['expt']['reps']
        expts = self.cfg['expt']['expts']
        threshold = self.cfg.device.readout.threshold[0]
        conf_mat_wn_reset = self.cfg.device.readout.confusion_matrix_without_reset

        try:
            I_data = temp_data['I_data']
            Q_data = temp_data['Q_data']
        except KeyError:
            try:
                I_data = temp_data['idata']
                Q_data = temp_data['qdata']
            except KeyError:
                I_data = temp_data['i0']
                Q_data = temp_data['q0']
            

        # reshape data into (rounds * reps x expts)
        '''
        Averager returns data in (rounds, reps) and if you do for looping 
        returns in (expts, rounds, reps)
        Here I assume you have done looping !
        '''
        I_data = np.reshape(np.transpose(np.reshape(I_data, ( expts,rounds, reps)), (1, 2, 0)), (rounds*reps, expts))
        Q_data = np.reshape(np.transpose(np.reshape(Q_data, ( expts,rounds, reps)), (1, 2, 0)), (rounds*reps, expts))

       
        # threshold data
        shots = np.zeros((rounds*reps, expts))
        print(shots.shape)
        shots[I_data > threshold] = 1

        # average over rounds and reps
        shots_avg = np.mean(shots, axis=0)
        np.shape(shots_avg)

        # fix using confusion matrix 
        ydata = shots_avg
        if conf: 
            P_matrix = np.matrix([[conf_mat_wn_reset[0], conf_mat_wn_reset[2]],[conf_mat_wn_reset[1], conf_mat_wn_reset[3]]])
            for i in range(len(ydata)):
                #ydata_old.append(ydata[i])
                from numpy.linalg import inv
                counts_new = inv(P_matrix)*np.matrix([[1-ydata[i]],[ydata[i]]])
                ydata[i] = counts_new[1,0]
        return ydata


    def bin_ss_data_given_ss(self, conf = True):
        '''
        Assumes that experiment perfroms its own single shot 

        This function takes the single shot data, applies the angle correction and threshold to the main data of the experiment
        '''
        temp_data = self.data
        rounds = self.cfg['expt']['rounds']
        reps = self.cfg['expt']['reps']
        expts = self.cfg['expt']['expts']

        try:
            I_data = temp_data['I_data']
            Q_data = temp_data['Q_data']
        except KeyError:
            I_data = temp_data['idata']
            Q_data = temp_data['qdata']

        # reshape data into (rounds * reps x expts)
        I_data = np.reshape(np.transpose(np.reshape(I_data, (rounds, expts, reps)), (0, 2, 1)), (rounds*reps, expts))
        Q_data = np.reshape(np.transpose(np.reshape(Q_data, (rounds, expts, reps)), (0, 2, 1)), (rounds*reps, expts))

        if 'angle' not in temp_data:
            print('No angle calibration found in data, assuming no rotation')
            temp_data['angle'] = 0
        if 'thresholds' not in temp_data:
            print('No thresholds found in data, using default threshold')
            temp_data['thresholds'] = self.cfg.device.readout.threshold[0]
        if 'confusion_matrix' not in temp_data:
            print('No confusion matrix found in data, using default confusion matrix')
            temp_data['confusion_matrix'] = self.cfg.device.readout.confusion_matrix_without_reset

        # rotate I,Q based on the angle calibration
        # theta = (-1*(float(temp_data['angle'])) - self.cfg['device']['readout']['phase'][0]) * np.pi/180 # to radians
        theta = -1*float(temp_data['angle']) * np.pi/180 # to radians

        print(f'Rotating data by {theta} radians')
        I_data_rot = I_data*np.cos(theta) - Q_data*np.sin(theta)
        Q_data_rot = I_data*np.sin(theta) + Q_data*np.cos(theta)

        # threshold data
        shots = np.zeros((rounds*reps, expts))
        shots[I_data_rot > temp_data['thresholds']] = 1

        # average over rounds and reps
        shots_avg = np.mean(shots, axis=0)
        np.shape(shots_avg)

        # fix using confusion matrix 
        ydata = shots_avg
        if conf: 
            P_matrix = np.matrix([[temp_data['confusion_matrix'][0], temp_data['confusion_matrix'][2]],[temp_data['confusion_matrix'][1], temp_data['confusion_matrix'][3]]])
            for i in range(len(ydata)):
                #ydata_old.append(ydata[i])
                from numpy.linalg import inv
                counts_new = inv(P_matrix)*np.matrix([[1-ydata[i]],[ydata[i]]])
                ydata[i] = counts_new[1,0]
        
        return ydata


    def filter_data_BS(self, a1, a2, a3, threshold, post_selection = False):
        # assume the last one  is experiment data, the last but one is for post selection
        '''
        This is for active reset post selection 

        the post selection parameter DOES not refer to active reset post selection
        a1: from active reset pre selection 
        a2: from actual experiment
        a3: from actual experiment post selection
        '''
        result_1 = []
        result_2 = []

        for k in range(len(a1)):
            if a1[k] < threshold:
                result_1.append(a2[k])
                if post_selection:
                    result_2.append(a3[k])

        return np.array(result_1), np.array(result_2)


    def filter_data_IQ(self, II, IQ, threshold):
        result_Ig = []
        result_Ie = []

        for k in range(len(II) // self.readout_per_round):
            index_4k_plus_2 = self.readout_per_round * k + self.readout_per_round - 2
            index_4k_plus_3 = self.readout_per_round * k + self.readout_per_round - 1

            if index_4k_plus_2 < len(II) and index_4k_plus_3 < len(II):
                if II[index_4k_plus_2] < threshold:
                    result_Ig.append(II[index_4k_plus_3])
                    result_Ie.append(IQ[index_4k_plus_3])

        return np.array(result_Ig), np.array(result_Ie)


    def post_select_raverager_data(self, temp_data):
        read_num = self.readout_per_round

        # Use self.cfg instead of attrs for config values
        rounds = self.cfg.expt.rounds
        reps = self.cfg.expt.reps
        expts = self.cfg.expt.expts
        I_data = np.array(temp_data['idata'])
        Q_data = np.array(temp_data['qdata'])

        I_data = np.reshape(np.transpose(np.reshape(I_data, (rounds, expts, reps, read_num)), (1, 0, 2, 3)), (expts, rounds * reps * read_num))
        Q_data = np.reshape(np.transpose(np.reshape(Q_data, (rounds, expts, reps, read_num)), (1, 0, 2, 3)), (expts, rounds * reps * read_num))

        Ilist = []
        Qlist = []
        # for ii in range(len(I_data) - 1): # why was this done???
        for ii in range(len(I_data)):
            Ig, Qg = self.filter_data_IQ(I_data[ii], Q_data[ii], self.threshold)
            Ilist.append(np.mean(Ig))
            Qlist.append(np.mean(Qg))

        return Ilist, Qlist


    def save_plot(self, fig, filename="plot.png", subdir=None):
        """
        Save a matplotlib figure to the specified folder.
        Optionally append the image path to a markdown file for viewing.

        Parameters:
        - fig: matplotlib.figure.Figure object to save.
        - folder_path: Path to the folder where the plot will be saved.
        - filename: Name of the file (default: "plot.png").
        - markdown_path: Path to a markdown file to append the image (optional).
        """ 
        """
        Save a matplotlib figure using the station's save_plot method.

        Parameters:
        - fig: matplotlib.figure.Figure object to save
        - filename: Name of the file (default: "plot.png")
        - subdir: Optional subdirectory within station's plot_path

        Raises:
        - ValueError: If no station was provided during initialization
        """
        if self.station is None:
            raise ValueError(
                "No station provided to fitting class. "
                "Cannot save plot without a MultimodeStation instance. "
                "Pass station=<your_station> when initializing this fitting class."
            )

        return self.station.save_plot(fig, filename, subdir=subdir)
        # plots_folder_path = "plots"
        # markdown_path = None
        # # print('entering save_plot') 

        # # Extract markdown folder from config if available
        # # if self.cfg and hasattr(self.cfg, "data_management"):
        # markdown_folder = self.station.log_path#getattr(self.cfg.data_management, "plot_and_logs_folder")
        # # print(f"Markdown folder path: {markdown_folder}")
        # plots_folder_path = self.station.plot_path
        # # if markdown_folder:
        # #     os.makedirs(markdown_folder, exist_ok=True)
        # #     today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        # #     markdown_path = os.path.join(markdown_folder, f"{today_str}.md")
        # #     if not os.path.exists(markdown_path):
        # #         with open(markdown_path, "w") as f:
        # #             f.write(f"# Plots for {today_str}\n\n")

        # now = datetime.datetime.now()
        # date_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        # print("supertitle is ", fig._suptitle)
        # if fig._suptitle is not None:
        #     fig._suptitle.set_text(fig._suptitle.get_text() + f" | {date_str} - {filename}")
        # else:
        #     fig.suptitle(f"{date_str} - {filename}", fontsize=16)
        # #get tight layout
        # fig.tight_layout()
        # filename = f"{date_str}_{filename}"
        # os.makedirs(plots_folder_path, exist_ok=True)
        # filepath = os.path.join(plots_folder_path, filename)
        # fig.savefig(filepath)
        # print(f"Plot saved to {filepath}")

        # if markdown_path is not None:
        #     # Use relative path if markdown file is in the same folder or subfolder
        #     rel_path = os.path.relpath(filepath, os.path.dirname(markdown_path))
        #     md_line = f"![Plot]({rel_path})\n"
        #     with open(markdown_path, "a") as md_file:
        #         md_file.write(md_line)
        #     print(f"Plot path appended to {markdown_path}")

