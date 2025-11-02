#!/usr/bin/env python3

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import sys

def unwrap_phase_per_tile_pol(phases):
    """
    Unwrap each tile and pol of the three-dimensional phase array (tile, freq, pol) along the frequency axis.
    Safely handle NaN.

    Parameters
    ----------
    phases : np.ndarray
        A phase array with shapes of (ntile, nfreq, npol) in radians.

    Returns
    -------
    unwrapped : np.ndarray
        Same dimensional array, containing unwrapped phases and retaining NaN.
    """
    if phases.ndim != 3:
        raise ValueError(f"Input must be 3D (tile, freq, pol), got shape {phases.shape}")

    ntile, nfreq, npol = phases.shape
    unwrapped = np.full_like(phases, np.nan, dtype=float)

    for i_tile in range(ntile):
        for i_pol in range(npol):
            ph = phases[i_tile, :, i_pol]
            mask = np.isfinite(ph)
            if np.count_nonzero(mask) > 1:
                unwrapped_tile = np.full_like(ph, np.nan)
                unwrapped_tile[mask] = np.unwrap(ph[mask])
                unwrapped[i_tile, :, i_pol] = unwrapped_tile
            else:
                # Keep the original value when all NaN values are present or when there is only one value
                unwrapped[i_tile, :, i_pol] = ph

    return unwrapped

# =============================
# Parameter Settings
# =============================
if len(sys.argv) != 2 or sys.argv[1] in ["-h", "--help"]:
    print(f"Usage: {sys.argv[0]} [filename.fits]")
    sys.exit(1)
filename = sys.argv[1]
# =============================
# Open FITS file
# =============================
f = fits.open(filename)
data = f["SOLUTIONS"].data
num_tiles = data.shape[1]
num_tiles_per_row = num_tiles // 16
name_tiles=f['TILES'].data

# Extract frequency axis (assuming it is in the SOLUTIONS header or inferred)
# Usually it can be obtained from another extension table or header; If not, use index instead
if "FREQS" in f:
    freqs = f["FREQS"].data
else:
    # fallback: Replace with channel index (in MHz)
    n_chan = data.shape[2]  # since there is re/im
    #n_chan = data.shape[2] // 2  # since there is re/im
    freqs = np.linspace(0, n_chan-1, n_chan)  # Occupation frequency, can be replaced with real MHz

# =============================
# Select a time block
# =============================
i_timeblock = 0
data = data[i_timeblock, :, :, ::2] + data[i_timeblock, :, :, 1::2] * 1j

# =============================
# For amplitude part
# =============================
amps = np.abs(data)
fig_amp, ax_amp = plt.subplots(num_tiles_per_row, 16, sharex=True, sharey=True, figsize=(18, 10))
ax_amp = ax_amp.flatten()
smoothness_rms_amp = []

for i in range(num_tiles):
    xx = amps[i, :, 0].flatten()
    yy = amps[i, :, 3].flatten()
    xy = amps[i, :, 1].flatten()
    yx = amps[i, :, 2].flatten()
    ax_amp[i].plot(freqs, xx, color="blue", alpha=0.6)
    ax_amp[i].plot(freqs, yy, color="green", alpha=0.6)
    ax_amp[i].plot(freqs, xy, color="blue", alpha=0.2)
    ax_amp[i].plot(freqs, yx, color="green", alpha=0.2)
    ax_amp[i].set_xlabel('{0}:{1}'.format(i,name_tiles[i][2]))
fig_amp.suptitle("Amplitude Solutions vs Frequency", fontsize=14)
plt.tight_layout()
plt.savefig('./Amplitude Solutions vs Frequency.png', dpi=300, bbox_inches = 'tight')

# =============================
# For phase part (consider ± 180° jump)
# =============================

phases = np.angle(data)
unwrap_phases = unwrap_phase_per_tile_pol(phases)
fig_phase, ax_phase = plt.subplots(num_tiles_per_row, 16, sharex=True, sharey=True, figsize=(18, 10))
ax_phase = ax_phase.flatten()
smoothness_rms_phase = []

for i in range(num_tiles):
    xx = phases[i, :, 0].flatten()
    yy = phases[i, :, 3].flatten()
    xy = phases[i, :, 1].flatten()
    yx = phases[i, :, 2].flatten()
    ax_phase[i].plot(freqs, xx, color="blue", alpha=0.6)
    ax_phase[i].plot(freqs, yy, color="green", alpha=0.6)
    ax_phase[i].plot(freqs, xy, color="blue", alpha=0.2)
    ax_phase[i].plot(freqs, yx, color="green", alpha=0.2)
    ax_phase[i].set_xlabel('{0}:{1}'.format(i,name_tiles[i][2]))
fig_phase.suptitle("Phase Solutions vs Frequency", fontsize=14)
plt.tight_layout()
plt.savefig('./Phase Solutions vs Frequency.png', dpi=300, bbox_inches = 'tight')
