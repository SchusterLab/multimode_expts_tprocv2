from .general_fitting import GeneralFitting
import numpy as np
import matplotlib.pyplot as plt
import os

class ColorPlot2D(GeneralFitting):
    """
    Class for producing 2D color plots from x, y, and multiple z lists.
    The analyze function finds the maximum response time (x value) for each 2D color plot.
    """
    def __init__(self, xlist, ylist, zlists, config=None, xlabel="X", ylabel="Y", zlabels=None, station=None):
        """
        xlist: 1D array of x values (e.g., time)
        ylist: 1D array of y values (e.g., frequency)
        zlists: list of 2D arrays, each shape (len(ylist), len(xlist)), representing response matrices
        config: optional configuration object
        xlabel: label for x axis
        ylabel: label for y axis
        zlabels: list of labels for each z plot (optional)
        """
        super().__init__(data=None, readout_per_round=2, threshold=-4.0, config=config, station=station)
        self.xlist = np.array(xlist)
        self.ylist = np.array(ylist)
        self.zlists = [np.array(z) for z in zlists]
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.zlabels = zlabels if zlabels is not None else [None] * len(self.zlists)
        self.results = []

    def analyze(self):
        """
        For each zlist (2D response matrix), find the x (time) value where the maximum response occurs.
        Stores a list of dicts with max value and corresponding x, y indices.
        """

        self.results = []
        for idx, z in enumerate(self.zlists):
            max_idx = np.unravel_index(np.argmax(z), z.shape)
            y_idx, x_idx = max_idx  # Corrected the order of indices
            print(max_idx)
            max_val = z[y_idx, x_idx]
            max_x = self.xlist[x_idx]
            max_y = self.ylist[y_idx]
            self.results.append({
                'z_index': idx,
                'max_value': max_val,
                'max_x': max_x,
                'max_y': max_y,
                'x_idx': x_idx,
                'y_idx': y_idx
            })

    def display(self, titles=None, vlines=None, hlines=None, save_fig=False, directory=None):
        """
        Display all 2D color plots with optional vertical/horizontal lines.
        Parameters:
            titles: list of titles for each subplot (optional)
            vlines: list of x values to draw vertical lines (optional)
            hlines: list of y values to draw horizontal lines (optional)
            save_fig: if True, saves the figure(s)
            directory: directory to save figures if save_fig is True
        """
        for idx, z in enumerate(self.zlists):
            plt.figure(figsize=(10, 6))

            # Dynamically adjust shading based on dimensions
            shading_mode = 'auto'
            if z.shape == (len(self.ylist) - 1, len(self.xlist) - 1):
                shading_mode = 'flat'

            plt.pcolormesh(self.xlist, self.ylist, z, shading=shading_mode, cmap='viridis')
            zlabel = self.zlabels[idx] if self.zlabels and idx < len(self.zlabels) else "Response"
            plt.colorbar(label=zlabel)
            title = titles[idx] if titles and idx < len(titles) else f'2D Color Plot {idx+1}'
            plt.title(title)
            plt.xlabel(self.xlabel)
            plt.ylabel(self.ylabel)

            # Add vertical lines and include in legend
            if vlines is not None:
                for v in vlines:
                    plt.axvline(v, color='red', linestyle='--', label=f'VLine: {v}')

            # Add horizontal lines
            if hlines is not None:
                for h in hlines:
                    plt.axhline(h, color='blue', linestyle='--')

            # Mark the maximum point if analysis was run
            if self.results and idx < len(self.results):
                res = self.results[idx]
                plt.plot(res['max_x'], res['max_y'], 'ko', label=f'Max Response ({res["max_x"]:.2f}, {res["max_y"]:.2f}, Z: {res["max_value"]:.2f})')

                # Add dashed lines intersecting at the maximum point
                plt.axvline(res['max_x'], color='green', linestyle='--', label=f'Max X: {res["max_x"]:.2f}')
                plt.axhline(res['max_y'], color='purple', linestyle='--', label=f'Max Y: {res["max_y"]:.2f}')

            plt.legend()
            plt.tight_layout()

            if save_fig:
                if directory:
                    os.makedirs(directory, exist_ok=True)
                    fname = f"colorplot2d_{idx+1}.png"
                    plt.savefig(os.path.join(directory, fname))

            plt.show()