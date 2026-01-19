from .general_fitting import GeneralFitting
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

class ColorPlot3D(GeneralFitting):
    """
    Class for producing 3D color plots from x, y, z, and F(x, y, z).
    The analyze function finds the maximum response value and its corresponding x, y, z indices.
    """
    def __init__(self, xlist, ylist, zlist, F, config=None, xlabel="X", ylabel="Y", zlabel="Z", station=None):
        """
        xlist: 1D array of x values
        ylist: 1D array of y values
        zlist: 1D array of z values
        F: 3D array of shape (len(zlist), len(ylist), len(xlist)), representing response values
        config: optional configuration object
        xlabel: label for x axis
        ylabel: label for y axis
        zlabel: label for z axis
        """
        super().__init__(data=None, readout_per_round=2, threshold=-4.0, config=config, station=station)
        self.xlist = np.array(xlist)
        self.ylist = np.array(ylist)
        self.zlist = np.array(zlist)
        self.F = np.array(F)
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.zlabel = zlabel
        self.results = {}

    def analyze(self):
        """
        Find the maximum response value in F and its corresponding x, y, z indices.
        """
        max_idx = np.unravel_index(np.argmax(self.F), self.F.shape)
        z_idx, y_idx, x_idx = max_idx
        max_val = self.F[z_idx, y_idx, x_idx]
        max_x = self.xlist[x_idx]
        max_y = self.ylist[y_idx]
        max_z = self.zlist[z_idx]

        self.results = {
            'max_value': max_val,
            'max_x': max_x,
            'max_y': max_y,
            'max_z': max_z,
            'x_idx': x_idx,
            'y_idx': y_idx,
            'z_idx': z_idx
        }

    def display(self, title="3D Color Plot", save_fig=False, directory=None):
        """
        Display a 3D color plot of F(x, y, z).

        Parameters:
            title: Title of the plot
            save_fig: If True, saves the figure
            directory: Directory to save the figure if save_fig is True
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Create a meshgrid for plotting
        X, Y, Z = np.meshgrid(self.xlist, self.ylist, self.zlist, indexing='ij')

        # Flatten the arrays for scatter plot
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        Z_flat = Z.flatten()
        F_flat = self.F.flatten()

        # Create the scatter plot
        sc = ax.scatter(X_flat, Y_flat, Z_flat, c=F_flat, cmap='viridis')
        plt.colorbar(sc, label="Response")

        ax.set_title(title)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_zlabel(self.zlabel)

        # Mark the maximum point if analysis was run
        if self.results:
            ax.scatter(
                self.results['max_x'],
                self.results['max_y'],
                self.results['max_z'],
                color='red',
                label=f"Max: {self.results['max_value']:.2f}",
                s=100
            )
            ax.legend()

        if save_fig:
            if directory:
                os.makedirs(directory, exist_ok=True)
                fname = "colorplot3d.png"
                plt.savefig(os.path.join(directory, fname))

        plt.show()