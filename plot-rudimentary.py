#!/usr/bin/env python3
"""
pyuv_mwa_vis_qc.py (fixed version)

Read MWA (or other pyuvdata-supported) visibility with pyuvdata,
compute time/freq statistics for a given baseline and polarization,
and save amplitude/phase vs time/freq plots.

Author: Wulinhui (patched by ChatGPT)
Dependencies: pyuvdata, numpy, matplotlib
Last updated: 2025-10-19
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pyuvdata import UVData

# ----------------------
# Utilities / helpers
# ----------------------
def robust_nanmedian(a, axis=None):
    return np.nanmedian(a, axis=axis)

def robust_nanmad_over_med(x):
    x = np.array(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return 1.4826 * mad / med if np.isfinite(med) and med != 0 else np.nan

def unwrap_safe(ph):
    ph = np.array(ph, dtype=float)
    unwrapped = np.full_like(ph, np.nan, dtype=float)
    mask = np.isfinite(ph)
    if np.count_nonzero(mask) > 1:
        unwrapped_freq = np.full_like(ph, np.nan)
        unwrapped_freq[mask] = np.unwrap(ph[mask])
        unwrapped = unwrapped_freq
    else:
        unwrapped = ph
    return unwrapped

def find_pol_index(uv: UVData, want_pol: str):
    want = want_pol.upper()
    if hasattr(uv, "polarization_array") and uv.polarization_array is not None:
        try:
            pol_strings = uv.get_pols()
            for idx, ps in enumerate(pol_strings):
                if ps.upper() == want:
                    return idx, pol_strings
        except Exception:
            pass

    if hasattr(uv, "polarization_array") and uv.polarization_array is not None:
        try:
            pols = uv.polarization_array
            if isinstance(pols[0], str):
                pol_strings = list(pols)
                if want in pol_strings:
                    return pol_strings.index(want), pol_strings
        except Exception:
            pass

    common = ['XX', 'YY', 'XY', 'YX', 'RR', 'LL', 'RL', 'LR']
    if want not in common:
        try:
            pol_strings = uv.get_pols()
        except Exception:
            pol_strings = common
    else:
        pol_strings = common

    if want in pol_strings:
        return pol_strings.index(want), pol_strings

    raise RuntimeError(f"Could not map requested polarization '{want}' to data polarizations. Available pols: {pol_strings}")

def antenna_name_to_index(uv: UVData, name_or_index):
    try:
        index = int(name_or_index)
        if hasattr(uv, "Nants_telescope") and index >= uv.Nants_telescope:
            print(f"[WARNING] Antenna index {index} may be out of range (Nants_telescope={uv.Nants_telescope})")
        return index
    except (ValueError, TypeError):
        pass

    if hasattr(uv, "antenna_names") and uv.antenna_names is not None:
        names = np.array(uv.antenna_names)
        where = np.where(names == name_or_index)[0]
        if where.size == 0:
            available_names = list(names[:min(20, len(names))])
            raise RuntimeError(f"Antenna name '{name_or_index}' not found. Available: {available_names} ...")
        return int(where[0])
    else:
        raise RuntimeError("UVData has no 'antenna_names' attribute to map from name.")

def load_uvdata(path, read_data=True, file_type=None, verbose=True):
    uv = UVData()
    if verbose:
        print(f"[INFO] Reading file: {path} (read_data={read_data}, file_type={file_type})")
    uv.read(path, read_data=read_data, file_type=file_type)
    if verbose:
        print(f"[INFO] Read UVData: Nblts={uv.Nblts}, Nfreqs={uv.Nfreqs}, Npols={uv.Npols}")
    return uv

def select_baseline(uv: UVData, ant1, ant2):
    a1 = int(ant1); a2 = int(ant2)
    ant1_arr = uv.ant_1_array
    ant2_arr = uv.ant_2_array
    mask = ((ant1_arr == a1) & (ant2_arr == a2)) | ((ant1_arr == a2) & (ant2_arr == a1))
    blt_inds = np.where(mask)[0]
    if blt_inds.size == 0:
        raise RuntimeError(f"No baseline found for antenna pair {a1}-{a2}.")
    return blt_inds

def freq_axis_mhz(uv: UVData):
    fa = uv.freq_array
    if fa is None:
        raise RuntimeError("UVData has no freq_array")
    fa = np.array(fa)
    if fa.ndim == 2:
        if fa.shape[0] == 1:
            freqs = fa[0]
        else:
            freqs = fa.flatten()
    else:
        freqs = fa
    return freqs * 1e-6

def process_baseline_uv(uv, blt_inds, pol_index, timebin=1, chanbin=1, use_weights=True):
    data = uv.data_array
    flag = getattr(uv, "flag_array", None)
    times = uv.time_array
    weight = None

    if use_weights:
        if hasattr(uv, "weight_array"):
            weight = uv.weight_array
        elif hasattr(uv, "extra_keywords") and "WEIGHT_SPECTRUM" in uv.extra_keywords:
            weight = uv.extra_keywords["WEIGHT_SPECTRUM"]
        elif hasattr(uv, "WEIGHT_SPECTRUM"):
            weight = uv.WEIGHT_SPECTRUM

    blt_inds = np.array(blt_inds, dtype=int)
    v = data[blt_inds, :, pol_index]
    f = flag[blt_inds, :, pol_index] if flag is not None else np.zeros_like(v, dtype=bool)
    w = weight[blt_inds, :, pol_index] if (weight is not None and weight.shape == data.shape) else np.ones_like(v)

    if chanbin > 1:
        nchan = v.shape[1]
        new_nchan = (nchan // chanbin) * chanbin
        if new_nchan < chanbin:
            raise RuntimeError("chanbin too large for available channels")
        v = v[:, :new_nchan].reshape(v.shape[0], -1, chanbin).mean(axis=2)
        f = f[:, :new_nchan].reshape(f.shape[0], -1, chanbin).any(axis=2)
        w = w[:, :new_nchan].reshape(w.shape[0], -1, chanbin).mean(axis=2)

    v_masked = np.where(~f, v, np.nan + 1j * np.nan)
    amp_row = np.nanmedian(np.abs(v_masked), axis=1)
    phs_row = np.nanmedian(np.angle(v_masked), axis=1)

    times_sel = times[blt_inds]
    if timebin > 1:
        k = (amp_row.shape[0] // timebin) * timebin
        if k > 0:
            amp_row = amp_row[:k].reshape(-1, timebin).mean(axis=1)
            phs_row = phs_row[:k].reshape(-1, timebin).mean(axis=1)
            times_sel = times_sel[:k].reshape(-1, timebin).mean(axis=1)

    if use_weights and weight is not None:
        amp_sum = np.nansum(np.abs(v_masked) * w, axis=0)
        wsum = np.nansum(w, axis=0)
        amp_freq = amp_sum / np.where(wsum > 0, wsum, np.nan)

        unit = v_masked / np.abs(v_masked)
        unit[~np.isfinite(unit.real)] = np.nan + 1j * np.nan
        phs_vec_sum = np.nansum(unit * w, axis=0)
        phs_wsum = np.nansum(w, axis=0)
        phs_freq = np.angle(phs_vec_sum / np.where(phs_wsum > 0, phs_wsum, np.nan))
    else:
        amp_freq = np.nanmean(np.abs(v_masked), axis=0)
        unit = v_masked / np.abs(v_masked)
        unit[~np.isfinite(unit.real)] = np.nan + 1j * np.nan
        phs_vec_sum = np.nansum(unit, axis=0)
        phs_wsum = np.sum(np.isfinite(unit.real), axis=0)
        phs_freq = np.angle(phs_vec_sum / np.where(phs_wsum > 0, phs_wsum, np.nan))

    phs_time_un = unwrap_safe(phs_row)
    phs_freq_un = unwrap_safe(phs_freq)

    amp_time_scatter = robust_nanmad_over_med(amp_row)
    phs_time_rms = np.nanstd(phs_time_un)
    amp_freq_scatter = robust_nanmad_over_med(amp_freq)

    finite_mask = np.isfinite(phs_freq_un)
    if np.count_nonzero(finite_mask) > 2:
        freq_mhz = freq_axis_mhz(uv)[finite_mask]
        phs_valid = phs_freq_un[finite_mask]
        A = np.vstack([freq_mhz, np.ones_like(freq_mhz)]).T
        coeff, _, _, _ = np.linalg.lstsq(A, phs_valid, rcond=None)
        phs_fit = A @ coeff
        phs_resid = phs_valid - phs_fit
        phs_freq_rms = np.nanstd(phs_resid)
    else:
        phs_freq_rms = np.nan

    out = {
        "times": times_sel,
        "time_hr": (times_sel - np.nanmin(times_sel)) / 3600.0 if len(times_sel) > 0 else np.array([]),
        "amp_time": amp_row,
        "phs_time": phs_time_un,
        "amp_freq": amp_freq,
        "phs_freq": phs_freq_un,
        "amp_time_scatter": amp_time_scatter,
        "phs_time_rms": phs_time_rms,
        "amp_freq_scatter": amp_freq_scatter,
        "phs_freq_rms": phs_freq_rms
    }

    return out

def save_time_freq_plots(out, freq_mhz, ant1_name, ant2_name, pol_label, col_label, outdir):
    tag = f"{ant1_name}-{ant2_name}_{pol_label}_{col_label}"
    os.makedirs(outdir, exist_ok=True)

    f1 = os.path.join(outdir, f"vis_amp_time_{tag}.png")
    plt.figure()
    plt.plot(out["time_hr"] * 3600 * 1e5, out["amp_time"], ".", ms=2)
    plt.xlabel("Time [s]")
    plt.ylabel(f"Amplitude ({tag})")
    plt.title("Amplitude vs Time")
    plt.tight_layout()
    plt.savefig(f1, dpi=150)
    plt.close()

    f2 = os.path.join(outdir, f"vis_phs_time_{tag}.png")
    plt.figure()
    plt.plot(out["time_hr"] * 3600 * 1e5, out["phs_time"], ".", ms=2)
    plt.xlabel("Time [s]")
    plt.ylabel("Phase (rad)")
    plt.title("Phase vs Time (unwrapped)")
    plt.tight_layout()
    plt.savefig(f2, dpi=150)
    plt.close()

    f3 = os.path.join(outdir, f"vis_amp_freq_{tag}.png")
    plt.figure()
    plt.plot(freq_mhz, out["amp_freq"], ".", ms=2)
    plt.xlabel("Frequency [MHz]")
    plt.ylabel(f"Amplitude ({tag})")
    plt.title("Amplitude vs Frequency (time-avg)")
    plt.tight_layout()
    plt.savefig(f3, dpi=150)
    plt.close()

    f4 = os.path.join(outdir, f"vis_phs_freq_{tag}.png")
    plt.figure()
    plt.plot(freq_mhz, out["phs_freq"], ".", ms=2)
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Phase (rad)")
    plt.title("Phase vs Frequency (unwrapped, circular mean)")
    plt.tight_layout()
    plt.savefig(f4, dpi=150)
    plt.close()

    return f1, f2, f3, f4

def main():
    parser = argparse.ArgumentParser(
        description="Plot vis amp/phase vs time/freq using pyuvdata (MWA).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use config.yaml (all parameters from config file)
  python plot-rudimentary.py --config config.yaml

  # Override specific parameters from config.yaml
  python plot-rudimentary.py --config config.yaml --ant1 3 --ant2 7

  # Use command line arguments only (no config file)
  python plot-rudimentary.py --file input_file.uvfits --ant1 2 --ant2 6 --pol XX --outdir ./plots
        """
    )
    parser.add_argument("--file", default=None, help="Path to visibility file (MWA corr fits, uvh5, ms, miriad...). Optional if using --config")
    parser.add_argument("--config", default=None, help="Path to configuration YAML file.")
    parser.add_argument("--ant1", help="Antenna 1 (index or name). Overrides config.yaml")
    parser.add_argument("--ant2", help="Antenna 2 (index or name). Overrides config.yaml")
    parser.add_argument("--pol", help="Polarization to plot (e.g. XX, XY, YY). Overrides config.yaml")
    parser.add_argument("--col", help="Column label (informational only for tag). Overrides config.yaml")
    parser.add_argument("--outdir", help="Output directory. Overrides config.yaml")
    parser.add_argument("--timebin", type=int, help="Time binning factor. Overrides config.yaml")
    parser.add_argument("--chanbin", type=int, help="Channel binning factor. Overrides config.yaml")
    parser.add_argument("--use_weights", action="store_true", help="Use weights for time averaging. Overrides config.yaml")
    parser.add_argument("--file_type", default=None, help="Optional file_type for pyuvdata.read (e.g. 'mwa_corr_fits', 'uvh5', 'ms').")

    # Extra parameters for plot-rudimentary
    parser.add_argument("--ddid", type=int, help="[MS ONLY] DATA_DESC_ID override. Overrides config.yaml")
    parser.add_argument("--list_baselines", action="store_true", help="[MS ONLY] List baselines and exit. Overrides config.yaml")
    parser.add_argument("--ignore_flags", action="store_true", help="Ignore flags during processing. Overrides config.yaml")
    parser.add_argument("--vmin", type=float, help="Color min for |FFT|/|FFT|^2 pages. Overrides config.yaml")
    parser.add_argument("--vmax", type=float, help="Color max for |FFT|/|FFT|^2 pages. Overrides config.yaml")
    parser.add_argument("--no_log", action="store_true", help="Disable log color for |FFT| and |FFT|^2 pages. Overrides config.yaml")
    parser.add_argument("--plot_title_base", help="Base title for plots. Overrides config.yaml")
    parser.add_argument("--out_file", help="Output filename for PDF. Overrides config.yaml")
    parser.add_argument("--show", action="store_true", help="Display plots inline. Overrides config.yaml")

    args = parser.parse_args()

    # Load configuration from config.yaml
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"[WARNING] Config file '{args.config}' not found. Using defaults.")
        config = {}
    except Exception as e:
        print(f"[WARNING] Error reading config file '{args.config}': {e}. Using defaults.")
        config = {}

    # Get parameters: command line args override config.yaml, config.yaml overrides defaults
    # Common parameters
    input_file = args.file if args.file is not None else config.get('input_file', '')
    if not input_file:
        parser.error("Input file must be specified either via --file or in config.yaml as 'input_file'")

    ant1 = args.ant1 if args.ant1 is not None else config.get('ant1', '2')
    ant2 = args.ant2 if args.ant2 is not None else config.get('ant2', '6')
    pol = args.pol if args.pol is not None else config.get('corr', 'XX')
    col = args.col if args.col is not None else config.get('col', 'DATA')
    outdir = args.outdir if args.outdir is not None else config.get('outdir', '.')
    timebin = args.timebin if args.timebin is not None else config.get('timebin', 1)
    chanbin = args.chanbin if args.chanbin is not None else config.get('chanbin', 1)
    # For boolean flags, if provided via command line, use it; otherwise use config
    use_weights = args.use_weights if args.use_weights else config.get('use_weights', False)

    # Extra parameters for plot-rudimentary
    ddid = args.ddid if args.ddid is not None else config.get('ddid', None)
    list_baselines = args.list_baselines if args.list_baselines else config.get('list_baselines', False)
    ignore_flags = args.ignore_flags if args.ignore_flags else config.get('ignore_flags', False)
    vmin = args.vmin if args.vmin is not None else config.get('vmin', None)
    vmax = args.vmax if args.vmax is not None else config.get('vmax', None)
    no_log = args.no_log if args.no_log else config.get('no_log', False)
    plot_title_base = args.plot_title_base if args.plot_title_base is not None else config.get('plot_title_base', "Delay–Time Waterfall")
    out_file = args.out_file if args.out_file is not None else config.get('out_file', "delay_waterfall_full.pdf")
    show = args.show if args.show else config.get('show', True)

    # Prepend outdir to output file path if relative
    if not os.path.isabs(out_file):
        out_file = os.path.join(outdir, out_file)

    uv = load_uvdata(input_file, read_data=True, file_type=args.file_type)

    try:
        # Convert ant1 and ant2 to int if possible, otherwise keep as string
        try:
            ant1_val = int(ant1)
        except ValueError:
            ant1_val = ant1
        try:
            ant2_val = int(ant2)
        except ValueError:
            ant2_val = ant2

        a1_idx = antenna_name_to_index(uv, ant1_val)
        a2_idx = antenna_name_to_index(uv, ant2_val)
    except Exception as e:
        print("[ERROR] antenna parsing:", e)
        sys.exit(1)

    blt_inds = select_baseline(uv, a1_idx, a2_idx)

    try:
        pol_idx, pol_list = find_pol_index(uv, pol)
    except Exception as e:
        print("[ERROR] polarization mapping:", e)
        print("Available pols (best guess):", getattr(uv, 'get_pols', lambda: 'unknown')())
        sys.exit(1)

    freq_mhz = freq_axis_mhz(uv)

    out = process_baseline_uv(uv, blt_inds, pol_idx, timebin=timebin, chanbin=chanbin, use_weights=use_weights)

    print("\n==== Quick QA ====")
    if hasattr(uv, "antenna_names") and uv.antenna_names is not None:
        a1_name, a2_name = uv.antenna_names[a1_idx], uv.antenna_names[a2_idx]
    else:
        print("[WARNING] No antenna_names found in file; using numeric indices.")
        a1_name, a2_name = str(a1_idx), str(a2_idx)

    print(f"Baseline: {a1_name}-{a2_name}  pol: {pol} col: {col}")
    print("Points (time):", out["time_hr"].size, " Channels:", freq_mhz.size)
    print("AMP time scatter (1.4826*MAD/med): {:.3f}".format(out["amp_time_scatter"]))
    print("PHASE time RMS (rad): {:.3f}".format(out["phs_time_rms"]))
    print("AMP freq scatter (1.4826*MAD/med): {:.3f}".format(out["amp_freq_scatter"]))
    print("PHASE freq RMS (rad) after linear detrend: {:.3f}".format(out["phs_freq_rms"]))
    print("===================\n")

    f1, f2, f3, f4 = save_time_freq_plots(out, freq_mhz, a1_name, a2_name, pol, col, outdir)
    print("Saved plots:")
    for f in (f1, f2, f3, f4):
        print(" ", f)

if __name__ == "__main__":
    main()
