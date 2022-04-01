"""
Classes for visualizing trajectory data.
"""
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp
from tqdm import tqdm

from pyib.md.potentials import Potential2D

# Resource-light and SSH-friendly non-GUI plotting
matplotlib.use('Agg')


class VisualizePotential2D:
    """
    Class defining functions to generate scatter plots and animated trajectories
    of a particle on a 2D potential surface.

    Args:
        potential2D:
        temp:
        xrange:
        yrange:
        contourvals:
        mesh:
        cmap:
    """
    def __init__(self,
                 potential2D: Potential2D,
                 temp: float,
                 xrange: tuple,
                 yrange: tuple,
                 contourvals = None,
                 clip = None,
                 mesh: int = 200,
                 cmap: str = 'jet'):
        self.potential2D = potential2D
        self.kT = 8.3145 / 1000 * temp
        self.xrange = xrange
        self.yrange = yrange
        self.contourvals = contourvals
        self.clip = clip
        self.mesh = mesh
        self.cmap = cmap

    def plot_potential(self):
        """
        Plots the potential within (xrange[0], xrange[1]) and (yrange[0], yrange[1]).
        """
        xx, yy = np.meshgrid(np.linspace(self.xrange[0], self.xrange[1], self.mesh), np.linspace(self.yrange[0], self.yrange[1], self.mesh))
        x = xx.ravel()
        y = yy.ravel()
        v = self.potential2D.potential(x, y)

        if self.clip is not None:
            V = v.reshape(self.mesh, self.mesh) / self.kT
            V = V.clip(max=self.clip)
        else:
            V = v.reshape(self.mesh, self.mesh) / self.kT

        fig, ax = plt.subplots(dpi=150)
        if self.contourvals is not None:
            cs = ax.contourf(xx, yy, V, self.contourvals, cmap=self.cmap)
        else:
            cs = ax.contourf(xx, yy, V, cmap=self.cmap)
        cbar = fig.colorbar(cs)
        cbar.set_label(r"Free energy ($k_B T$)")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        return (fig, ax)

    def plot_projection_x(self):
        """
        Plots the x-projection of potential within (xrange[0], xrange[1])
        and (yrange[0], yrange[1]).
        """
        # Compute 2D free energy profile
        xx, yy = np.meshgrid(np.linspace(self.xrange[0], self.xrange[1], self.mesh), np.linspace(self.yrange[0], self.yrange[1], self.mesh))
        x = xx.ravel()
        y = yy.ravel()
        v = self.potential2D.potential(x, y)
        V = v.reshape(self.mesh, self.mesh) / self.kT

        # Integrate over y-coordinate to get free-energy along x-coordinate
        Fx = -logsumexp(-V, axis=0)
        Fx = Fx - np.min(Fx)
        x = np.linspace(self.xrange[0], self.xrange[1], self.mesh)

        # Plot
        fig, ax = plt.subplots(dpi=150)
        ax.plot(x, Fx)
        ax.set_ylabel(r"Free energy ($k_B T$)")
        ax.set_xlabel("$x$")
        ax.set_ylim([0, None])
        return (fig, ax, x, Fx)

    def scatter_traj(self, traj, outimg, every=1, s=1, c='black'):
        """
        Scatters entire trajectory onto potential energy surface.

        Args:
            traj:
            outimg:
            every:
            s:
            c:
        """
        fig, ax = self.plot_potential()
        ax.scatter(traj[::every, 0], traj[::every, 1], s=s, c=c)
        plt.savefig(outimg)
        plt.close()

    def scatter_traj_projection_x(self, traj, outimg, every=1, s=1, c='black'):
        """
        Scatters x-projection of entire trajectory onto potential energy surface.

        Args:
            traj:
            outimg:
            every:
            s:
            c:
        """
        fig, ax, x, Fx = self.plot_projection_x()
        for i in tqdm(range(0, traj.shape[0], every)):
            xpt = traj[i, 0]
            yloc = np.argmin((x - xpt)**2)
            ypt = Fx[yloc]
            ax.scatter(xpt, ypt, s=s, c=c)
        plt.savefig(outimg)
        plt.close()

    def animate_traj(self, traj, outdir, every=1, s=3, c='black', call_ffmpeg: bool = True):
        """
        Plots positions at timesteps defined by interval `every` on potential
        energy surface and stitches together plots using ffmpeg to make a movie.

        Args:
            traj (iterable):
            outdir (str):
            every (int):
            s (int):
            c (str):
            call_ffmpeg (bool):
        """
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        for t, frame in enumerate(tqdm(traj[::every])):
            fig, ax = self.plot_potential()
            ax.scatter(frame[0], frame[1], s=s, c=c)
            plt.savefig("{}/traj.{:05d}.png".format(outdir, t))
            plt.close()

        if call_ffmpeg:
            os.system("ffmpeg -r 25 -i {}/traj.%5d.png -vb 20M {}/traj.mp4".format(outdir, outdir))

    def animate_traj_projection_x(self, traj, outdir, every=1, s=3, c='black',
                                  call_ffmpeg: bool = True):
        """
        Plots positions at timesteps defined by interval `every` on the x-projection of the
        potential energy surface and stitches together plots using ffmpeg to make a movie.

        Args:
            traj (iterable):
            outdir (str):
            every (int):
            s (int):
            c (str):
            call_ffmpeg (bool):
        """
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        for t, frame in enumerate(tqdm(traj[::every])):
            fig, ax, x, Fx = self.plot_projection_x()
            xpt = frame[0]
            yloc = np.argmin((x - xpt)**2)
            ypt = Fx[yloc]
            ax.scatter(xpt, ypt, s=s, c=c)
            plt.savefig("{}/traj_x.{:05d}.png".format(outdir, t))
            plt.close()

        if call_ffmpeg:
            os.system("ffmpeg -r 25 -i {}/traj_x.%5d.png -vb 20M {}/traj_x.mp4".format(outdir, outdir))
