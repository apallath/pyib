"""
Tests for visualization of different potentials.

Note:
    Tests are self-explanatory and are hence not documented.
"""
import inspect
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

from pyib.md.potentials import SlipBondPotential2D, CatchBondPotential2D, SzaboBerezhkovskiiPotential as SBPotential, MullerBrownPotential
from pyib.md.visualization import VisualizePotential2D


def test_vis_SlipBondPotential2D():
    pot = SlipBondPotential2D()
    temp = 300
    vis = VisualizePotential2D(pot, temp=temp,
                               xrange=[-12, 15], yrange=[-6, 8],
                               contourvals=10, clip=15)

    if not os.path.exists("./tmp/"):
        os.makedirs("./tmp/")

    # 2D surface
    fig, ax = vis.plot_potential()
    plt.savefig("tmp/SlipBondPotential2D.png")

    # 1D projection
    fig, ax, _, _ = vis.plot_projection_x()
    plt.savefig("tmp/SlipBondPotential2D_x.png")


def test_vis_CatchBondPotential():
    pot = CatchBondPotential2D()
    temp = 300
    vis = VisualizePotential2D(pot, temp=temp,
                               xrange=[-12, 15], yrange=[-6, 8],
                               contourvals=10, clip=15)

    if not os.path.exists("./tmp/"):
        os.makedirs("./tmp/")

    # 2D surface
    fig, ax = vis.plot_potential()
    plt.savefig("tmp/CatchBondPotential2D.png")

    # 1D projection
    fig, ax, _, _ = vis.plot_projection_x()
    plt.savefig("tmp/CatchBondPotential2D_x.png")


def test_vis_SBPotential():
    pot = SBPotential()
    temp = 300
    vis = VisualizePotential2D(pot, temp=temp,
                               xrange=[-7.5, 7.5], yrange=[-7.5, 7.5],
                               contourvals=[-2, -1, 0, 1, 2, 5, 8, 10])

    if not os.path.exists("./tmp/"):
        os.makedirs("./tmp/")

    # 2D surface
    fig, ax = vis.plot_potential()
    plt.savefig("tmp/SBPotential.png")

    # 1D projection
    fig, ax, _, _ = vis.plot_projection_x()
    plt.savefig("tmp/SBPotential_x.png")


def test_vis_MullerBrownPotential():
    pot = MullerBrownPotential()
    temp = 300
    vis = VisualizePotential2D(pot, temp=temp,
                               xrange=[-1.5, 1.2], yrange=[-0.2, 2],
                               contourvals=50, clip=100)

    if not os.path.exists("./tmp/"):
        os.makedirs("./tmp/")

    # 2D surface
    fig, ax = vis.plot_potential()
    plt.savefig("tmp/MullerBrownPotential.png")

    # 1D projection
    fig, ax, _, _ = vis.plot_projection_x()
    plt.savefig("tmp/MullerBrownPotential_x.png")


if __name__ == "__main__":
    all_objects = inspect.getmembers(sys.modules[__name__])
    for obj in all_objects:
        if re.match("^test_+", obj[0]):
            print("Running " + obj[0] + " ...")
            obj[1]()
            print("Done")
