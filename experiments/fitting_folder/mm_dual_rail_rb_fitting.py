from .general_fitting import GeneralFitting
import numpy as np
import matplotlib.pyplot as plt
import os

class MM_DualRailRBFitting(GeneralFitting):
    def __init__(self, filename=None, file_prefix=None, data=None, readout_per_round=2, threshold=-4.0, config=None,
                 prev_data=None, expt_path=None, title='RB', dir_path=None, station=None):
        super().__init__(data, readout_per_round, threshold, config, station)
        self.filename = filename
        self.expt_path = expt_path
        self.prev_data = prev_data
        self.title = title
        self.file_prefix = file_prefix
        self.dir_path = dir_path

    def get_sweep_files(self):
        # Implementation from fit_display_classes.py
        pass

    def plot_rb(self, fids_list, fids_post_list, xlist,
                pop_dict, pop_err_dict, ebars_list, ebars_post_list,
                reset_qubit_after_parity=False, parity_meas=True,
                title='M1-S4 RB Post selection', save_fig=False):
        # Implementation from fit_display_classes.py
        pass

    def show_rb(self, dual_rail_spec=False, skip_spec_state_idx=None, active_reset=False, save_fig=False):
        # Implementation from fit_display_classes.py
        pass

    def RB_extract_postselction_excited(self, temp_data, attrs, active_reset=False, conf_matrix=None):
        # Implementation from fit_display_classes.py
        pass