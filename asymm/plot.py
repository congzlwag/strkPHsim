import numpy as np
import h5py
from matplotlib import pyplot as plt
from matplotlib import colors as mcolor
plt.rcParams["image.cmap"] = 'bwr'

tag = "1psTwide"
with h5py.File(f"{tag}S.h5") as fh:
    streakd = np.squeeze(fh["densities/A0p1"][...])
    
with h5py.File(f"{tag}UnS.h5") as fh:
    unstreakd = np.squeeze(fh["densities/A0p0"][...])

diff = (streakd.mean(axis=-1) - unstreakd).T

plt.pcolor(diff, norm=mcolor.CenteredNorm())
plt.colorbar()
plt.title("Tw=(-4,5)fs")
plt.show()