#!/usr/bin/env python3
"""
Flag Occupancy Visualization Tool

This script plots flag occupancy (fraction of flagged visibilities) versus time
and frequency for visibility datasets stored either in:
1. Measurement Set (MS) format (using casacore)
2. UVFITS or other formats supported by pyuvdata

Author: Flag occupancy plot codes originally written by Dev Null,
        modified and integrated by SHAO EOR Group and Teal Team

Dependencies: casacore, pyuvdata, numpy, matplotlib, astropy, pyyaml
Last updated: 2025-11-08
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from casacore.tables import table
from astropy.io import fits
import yaml

try:
    from pyuvdata import UVData
    PYUVDATA_AVAILABLE = True
except ImportError:
    PYUVDATA_AVAILABLE = False
    print("[WARNING] pyuvdata not available - UVFITS/pyuvdata-supported formats will not work")


def plot_ms_flags_casacore(ms_path, obsname=None, cmap="viridis", outdir=None, show=False):
    """
    Plot flag occupancy for MS format using casacore.

    Parameters
    ----------
    ms_path : str
        Path to Measurement Set
    obsname : str, optional
        Observation name for plot title
    cmap : str, optional
        Colormap to use (default: 'viridis')
    outdir : str, optional
        Output directory for saving the plot
    show : bool, optional
        If True, display plot; if False, save to file

    Returns
    -------
    str or None
        Path to saved file if saved, None if displayed
    """
    # Read MAIN table
    t = table(ms_path, readonly=True)
    flags = t.getcol('FLAG')            # shape: (nrow, nchan, ncorr)
    times = t.getcol('TIME')            # shape: (nrow,)

    # Row-level flags (if available)
    if 'FLAG_ROW' in t.colnames():
        flag_row = t.getcol('FLAG_ROW').astype(bool)  # (nrow,)
        flags[flag_row, :, :] = True

    # Get frequencies from SPECTRAL_WINDOW table
    spw_tab = table(f"{ms_path}::SPECTRAL_WINDOW", readonly=True)
    chan_freq = spw_tab.getcol('CHAN_FREQ')          # [nspw, nchan_spw]

    # For simplicity: only plot DDID=0 if multiple SPWs present
    if chan_freq.shape[0] != 1:
        print(f"[WARNING] Detected {chan_freq.shape[0]} SPW/DDID, plotting only DDID=0.")

    freqs_hz = chan_freq[0]
    freqs_mhz = freqs_hz / 1e6

    # Aggregate rows with same TIME
    utimes, inv = np.unique(times, return_inverse=True)
    nt = utimes.size

    occ_counts = np.zeros((nt, freqs_mhz.size), dtype=np.float64)

    # Sum over polarization dimension (flags True=1, False=0)
    per_row = flags.sum(axis=-1)  # (nrow, nchan)

    for row_idx, tbin in enumerate(inv):
        occ_counts[tbin] += per_row[row_idx]

    rows_per_time = np.bincount(inv)              # (nt,)
    ncorr = flags.shape[-1]
    full_occupancy_value = rows_per_time[:, None] * ncorr

    occupancy = occ_counts / full_occupancy_value
    full_mask = (occ_counts == full_occupancy_value)
    occupancy[full_mask] = np.nan

    max_occupancy = np.nanmax(occupancy) if np.any(np.isfinite(occupancy)) else 0
    print(f"max_occupancy={max_occupancy} full_occupancy_value varies per-time")

    # Close tables
    spw_tab.close()
    t.close()

    # Create plot
    plt.figure()
    title = f"{obsname or ms_path} Flag Occupancy"
    plt.suptitle(title)

    plt.imshow(
        occupancy,
        aspect="auto",
        interpolation="none",
        cmap=cmap,
        extent=[freqs_mhz.min(), freqs_mhz.max(), utimes.max(), utimes.min()],
    )

    cbar = plt.colorbar()
    cbar.set_label("Flag Occupancy")
    plt.ylabel("TIME (s, MJD*86400-like)")
    plt.xlabel("Frequency [MHz]")
    plt.gcf().set_size_inches(16, 9)
    plt.tight_layout()

    if show:
        plt.show()
        return None

    if outdir is None:
        outdir = "."
    os.makedirs(outdir, exist_ok=True)
    obs_label = obsname or os.path.basename(ms_path.rstrip("/"))
    outfile = os.path.join(outdir, f"flag_occupancy_{obs_label}.png")
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    return outfile


def get_gps_times(uv):
    """Convert UVData time_array (MJD) to GPS seconds."""
    if not hasattr(uv, "time_array"):
        raise RuntimeError("UVData object has no time_array attribute")
    times_mjd = uv.time_array
    return (times_mjd - 44244.0) * 86400.0


def plot_uv_flags_pyuvdata(input_file, obsname=None, cmap="Reds", outdir=None, show=False):
    """
    Plot flag occupancy for UVFITS (or other pyuvdata-supported formats).

    Parameters
    ----------
    input_file : str
        Path to UVFITS/UVData-supported file
    obsname : str, optional
        Observation name for plot title
    cmap : str, optional
        Colormap to use (default: 'Reds')
    outdir : str, optional
        Output directory for saving the plot
    show : bool, optional
        If True, display plot; if False, save to file

    Returns
    -------
    str or None
        Path to saved file if saved, None if displayed
    """
    if not PYUVDATA_AVAILABLE:
        raise RuntimeError("pyuvdata is required for UVFITS flag plotting but not available")

    uv = UVData()
    uv.read(input_file, read_data=True)

    try:
        pols = uv.get_pols()
    except Exception:
        if hasattr(uv, 'polarization_array'):
            pols = [str(p) for p in uv.polarization_array]
        else:
            pols = ['Unknown']

    gps_times = get_gps_times(uv)

    if uv.freq_array is None:
        raise RuntimeError("UVData object has no freq_array attribute")

    if uv.freq_array.ndim == 2:
        freqs_mhz = uv.freq_array[0] / 1e6
    else:
        freqs_mhz = uv.freq_array / 1e6

    if not hasattr(uv, 'flag_array') or uv.flag_array is None:
        raise RuntimeError("No flag_array found in UVData. Cannot plot flag occupancy.")

    flag_array = uv.flag_array
    unique_times = np.unique(uv.time_array)
    n_times = len(unique_times)

    if hasattr(uv, 'baseline_array'):
        unique_baselines = np.unique(uv.baseline_array)
        n_bls = len(unique_baselines)
    else:
        n_bls = flag_array.shape[0] // n_times if n_times > 0 else flag_array.shape[0]

    n_spws = getattr(uv, 'Nspws', 1)

    if flag_array.shape[0] == n_times * n_bls * n_spws:
        flag_reshaped = flag_array.reshape(n_times, n_bls, n_spws, uv.Nfreqs, len(pols))
        occupancy = np.nansum(flag_reshaped, axis=(1, 2, 4)).astype(np.float64)
        full_occupancy_value = n_bls * n_spws * len(pols)
    else:
        time_indices = np.searchsorted(unique_times, uv.time_array)
        occupancy = np.zeros((n_times, uv.Nfreqs), dtype=np.float64)

        for t_idx in range(n_times):
            mask = (time_indices == t_idx)
            if np.any(mask):
                occupancy[t_idx, :] = np.nansum(flag_array[mask, :, :], axis=(0, 2))

        n_blts_per_time = flag_array.shape[0] // n_times if n_times > 0 else flag_array.shape[0]
        full_occupancy_value = n_blts_per_time * len(pols)

    occupancy[occupancy == full_occupancy_value] = np.nan
    max_occupancy = np.nanmax(occupancy) if np.any(np.isfinite(occupancy)) else 0
    print(f"max_occupancy={max_occupancy} full_occupancy_value={full_occupancy_value}")

    occupancy /= full_occupancy_value

    plt.figure()
    title = f"{obsname or os.path.basename(input_file)} Flag Occupancy {pols[0] if len(pols) == 1 else ''}"
    plt.suptitle(title)

    unique_times_mjd = np.unique(uv.time_array)
    gps_times_unique = (unique_times_mjd - 44244.0) * 86400.0

    if occupancy.shape[0] != len(gps_times_unique):
        if occupancy.shape[0] < len(gps_times_unique):
            repeat_factor = len(gps_times_unique) // occupancy.shape[0] + 1
            occupancy = np.repeat(occupancy, repeat_factor, axis=0)[:len(gps_times_unique)]
        else:
            occupancy = occupancy[:len(gps_times_unique)]

    plt.imshow(
        occupancy,
        aspect="auto",
        interpolation="none",
        cmap=cmap,
        extent=[
            np.min(freqs_mhz),
            np.max(freqs_mhz),
            np.max(gps_times_unique),
            np.min(gps_times_unique),
        ],
    )

    cbar = plt.colorbar()
    cbar.set_label("Flag Occupancy")
    plt.ylabel("GPS Time [s]")
    plt.xlabel("Frequency [MHz]")
    plt.gcf().set_size_inches(16, np.min([9, 4 * len(pols)]))
    plt.tight_layout()

    if show:
        plt.show()
        return None

    if outdir is None:
        outdir = "."
    os.makedirs(outdir, exist_ok=True)
    obs_label = obsname or os.path.splitext(os.path.basename(input_file))[0]
    outfile = os.path.join(outdir, f"flag_occupancy_{obs_label}.png")
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    return outfile


def determine_file_format(input_file, format_type="auto"):
    """Determine whether the file is MS or UVFITS."""
    if format_type != "auto":
        return format_type

    if input_file.endswith('.ms') or os.path.isdir(input_file):
        try:
            with table(f"{input_file}/POLARIZATION", readonly=True, ack=False):
                return 'ms'
        except (RuntimeError, OSError):
            pass

    if input_file.lower().endswith(('.fits', '.uvfits')):
        return 'uvfits'

    # Try detecting via pyuvdata if available
    if PYUVDATA_AVAILABLE:
        try:
            uv = UVData()
            uv.read(input_file, read_data=False)
            return 'uvfits'
        except Exception:
            pass

    raise ValueError("Cannot determine file format. Use --format to specify ('ms' or 'uvfits').")


def run_flag_plotting(config_file='config.yaml', input_file=None, format_type=None,
                      outdir=None, show=None, cmap=None):
    """
    Main processing function for plotting flag occupancy.

    Parameters
    ----------
    config_file : str, optional
        Path to configuration YAML file (default: 'config.yaml')
    input_file : str, optional
        Path to input data file (MS or UVFITS format). If None, read from config_file.
    format_type : str, optional
        Force input format ('auto', 'ms', 'uvfits'). If None, read from config_file.
    outdir : str, optional
        Output directory for plots. If None, read from config_file or default to '.'
    show : bool, optional
        If True, show plots instead of saving them (default: False).
    cmap : str, optional
        Colormap to use for plotting. Defaults to 'viridis' for MS and 'Reds' for UVFITS.

    Returns
    -------
    str or None
        Path to saved plot (if show=False), otherwise None.
    """
    # Load configuration
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"[WARNING] Config file '{config_file}' not found. Using defaults.")
        config = {}
    except Exception as e:
        print(f"[WARNING] Error reading config file '{config_file}': {e}. Using defaults.")
        config = {}

    if input_file is None:
        input_file = config.get('input_file', '')
        if not input_file:
            raise ValueError("Input file must be specified either via argument or config.yaml")

    if format_type is None:
        format_type = config.get('format', 'auto')

    if outdir is None:
        outdir = config.get('outdir', '.')

    if show is None:
        show = config.get('show', False)

    # Determine format
    file_format = determine_file_format(input_file, format_type)

    obsname = os.path.splitext(os.path.basename(input_file.rstrip("/"))) if os.path.isdir(input_file) else os.path.basename(input_file)
    obsname = obsname[0] if isinstance(obsname, tuple) else obsname

    if file_format == 'ms':
        cmap_used = cmap or config.get('flag_cmap_ms', 'viridis')
        print(f"[INFO] Plotting MS flag occupancy (cmap={cmap_used})")
        return plot_ms_flags_casacore(input_file, obsname=obsname, cmap=cmap_used, outdir=outdir, show=show)
    elif file_format == 'uvfits':
        cmap_used = cmap or config.get('flag_cmap_uv', 'Reds')
        print(f"[INFO] Plotting UVFITS flag occupancy (cmap={cmap_used})")
        return plot_uv_flags_pyuvdata(input_file, obsname=obsname, cmap=cmap_used, outdir=outdir, show=show)
    else:
        raise ValueError(f"Unsupported format: {file_format}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot flag occupancy vs time/frequency.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use config.yaml (all parameters from config file)
  python plot_flags.py --config config.yaml

  # Override specific parameters from config.yaml
  python plot_flags.py --config config.yaml --format uvfits --cmap plasma

  # Specify all parameters via command line (no config file)
  python plot_flags.py input_file.ms --format ms --outdir ./plots

  # Show plot interactively
  python plot_flags.py input_file.uvfits --show
        """
    )
    parser.add_argument("input_file", nargs='?', default=None,
                        help="Input file (MS directory or UVFITS file). Optional if using --config")
    parser.add_argument("--config", default='config.yaml',
                        help="Path to configuration YAML file (default: 'config.yaml')")
    parser.add_argument("--format", choices=['auto', 'ms', 'uvfits'],
                        help="Force input format (default: auto-detect)")
    parser.add_argument("--outdir", help="Output directory for plots. Overrides config.yaml")
    parser.add_argument("--cmap", help="Colormap to use for the plot. Overrides defaults/config.")

    def str_to_bool(v):
        if v is None:
            return True
        if isinstance(v, bool):
            return v
        v_lower = str(v).lower()
        if v_lower in ('yes', 'true', 't', 'y', '1'):
            return True
        if v_lower in ('no', 'false', 'f', 'n', '0'):
            return False
        raise argparse.ArgumentTypeError(f"Boolean value expected, got {v}")

    parser.add_argument("--show", nargs='?', const=True, default=None, type=str_to_bool,
                        help="Show plot interactively instead of saving PNG. "
                             "Use --show (True) or --show False. If not specified, uses config.yaml value.")

    args = parser.parse_args()

    run_flag_plotting(
        config_file=args.config,
        input_file=args.input_file,
        format_type=args.format,
        outdir=args.outdir,
        show=args.show,
        cmap=args.cmap
    )


if __name__ == "__main__":
    main()

