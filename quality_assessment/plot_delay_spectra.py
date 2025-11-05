#!/usr/bin/env python3
"""
Delay–time dynamic spectrum (waterfall) with three outputs in a single PDF:
  Page 1: Raw FFT (two subplots) -> log(|FFT|) and Phase ∠FFT
  Page 2: |FFT|
  Page 3: |FFT|²

Also prints delay spectrum statistics (max, mean, median, total power, peak delay).

Supports:
  - Measurement Sets (.MS) via casacore.tables
  - UVFITS via pyuvdata

Authors:
  - Shao EoR Group and Teal Team
"""

import os
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages
from casacore.tables import table

try:
    from pyuvdata import UVData
    PYUVDATA_AVAILABLE = True
except ImportError:
    PYUVDATA_AVAILABLE = False


# ----------------------------
# Utility helpers
# ----------------------------
def robust_percentiles(vals, low=5, high=95, floor=1e-12):
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return floor, floor * 10.0
    vmin = max(np.percentile(vals, low), floor)
    vmax = np.percentile(vals, high)
    if vmax <= vmin:
        vmax = vmin * 10.0
    return vmin, vmax


def to_edges(x):
    x = np.asarray(x, float)
    if x.size < 2:
        dx = 1.0
        return np.array([x[0]-0.5*dx, x[0]+0.5*dx])
    dx = np.diff(x)
    dx = np.r_[dx[:1], dx]
    return np.r_[x - dx/2, x[-1] + dx[-1]/2]


# ----------------------------
# MS helpers
# ----------------------------
CORR_MAP_NUM2STR = {5: 'RR', 6: 'RL', 7: 'LR', 8: 'LL', 9: 'XX', 10: 'XY', 11: 'YX', 12: 'YY'}


def ms_get_corr_index(ms_path: str, want_corr: str) -> int:
    pol = table(f"{ms_path}/POLARIZATION", readonly=True, ack=False)
    corr_types = pol.getcol('CORR_TYPE')[0]
    pol.close()
    labels = [CORR_MAP_NUM2STR.get(int(x), str(x)) for x in list(corr_types)]
    want = want_corr.upper()
    if want not in labels:
        raise RuntimeError(f"Requested correlation {want} not found. Available: {labels}")
    return labels.index(want)


def ms_get_freq_axis_hz(ms_path: str, ddid: int) -> np.ndarray:
    dd = table(f"{ms_path}/DATA_DESCRIPTION", readonly=True, ack=False)
    spw_id = int(dd.getcell('SPECTRAL_WINDOW_ID', ddid))
    dd.close()
    spw = table(f"{ms_path}/SPECTRAL_WINDOW", readonly=True, ack=False)
    chan_freq_hz = spw.getcell('CHAN_FREQ', spw_id)
    spw.close()
    return np.asarray(chan_freq_hz, dtype=float)


def ms_ant_to_index(ms_path: str, name_or_index) -> int:
    names = table(f"{ms_path}/ANTENNA", readonly=True, ack=False).getcol('NAME')
    s = str(name_or_index)
    if s.isdigit():
        idx = int(s)
        if idx < 0 or idx >= len(names):
            raise RuntimeError(f"Antenna index {idx} out of range [0,{len(names)-1}]")
        return idx
    where = np.where(names == s)[0]
    if where.size == 0:
        raise RuntimeError(f"Antenna '{name_or_index}' not found. Available: {list(names)}")
    return int(where[0])


def ms_list_baselines(ms_path: str):
    t = table(ms_path, readonly=True, ack=False)
    ants = table(f"{ms_path}/ANTENNA", readonly=True, ack=False).getcol('NAME')
    a1 = t.getcol('ANTENNA1')
    a2 = t.getcol('ANTENNA2')
    bls = np.unique(np.vstack([a1, a2]).T, axis=0)
    print("\nAntenna indices & names:")
    for i, nm in enumerate(ants):
        print(f"  {i}: {nm}")
    print("\nBaselines (ANT1-ANT2):")
    for u, v in bls:
        print(f"  {u}-{v}  ({ants[u]} - {ants[v]})")
    print(f"\nTotal unique baselines: {len(bls)}\n")
    return bls, ants


# ----------------------------
# Smart polarization resolver (UVFITS)
# ----------------------------
def resolve_polarization(requested: str, available: list[str]) -> str:
    req = requested.strip().lower()
    avail_lower = [p.lower() for p in available]
    if req in avail_lower:
        return available[avail_lower.index(req)]

    equivalents = {
        'xx': ['xx', 'nn'],
        'yy': ['yy', 'ee'],
        'xy': ['xy', 'ne'],
        'yx': ['yx', 'en'],
        'rr': ['rr'],
        'll': ['ll'],
    }
    alt_map = {'nn': 'xx', 'ee': 'yy', 'ne': 'xy', 'en': 'yx'}
    key = alt_map.get(req, req)
    if key in equivalents:
        for candidate in equivalents[key]:
            if candidate in avail_lower:
                chosen = available[avail_lower.index(candidate)]
                if candidate != req:
                    print(f"[INFO] Requested polarization '{requested}' not found, using fallback '{chosen}'")
                return chosen
    raise RuntimeError(f"Polarization {requested} not found among {available}")


# ----------------------------
# FFT over frequency
# ----------------------------
def fft_delay_complex(vis_tf: np.ndarray, df_hz: float):
    F = np.fft.fft(vis_tf, axis=1)
    F = np.fft.fftshift(F, axes=1)
    tau_s = np.fft.fftshift(np.fft.fftfreq(vis_tf.shape[1], d=df_hz))
    return tau_s, F


# ----------------------------
# Load MS or UVFITS → delay waterfall
# ----------------------------
def load_ms_delay_waterfall(ms_path, ant1, ant2, corr, col="DATA",
                            ddid_override=None, timebin=1, chanbin=1,
                            ignore_flags=False):
    t = table(ms_path, readonly=True, ack=False)
    ants = table(f"{ms_path}/ANTENNA", readonly=True, ack=False).getcol('NAME')
    a1 = ms_ant_to_index(ms_path, ant1)
    a2 = ms_ant_to_index(ms_path, ant2)
    lo, hi = sorted([a1, a2])
    q = t.query(f"ANTENNA1=={lo} && ANTENNA2=={hi}")
    if q.nrows() == 0:
        raise RuntimeError(f"No rows for baseline {lo}-{hi} ({ants[lo]} - {ants[hi]}).")

    ddids = np.unique(q.getcol('DATA_DESC_ID'))
    ddid = ddid_override if ddid_override is not None else int(ddids[0])
    freqs_hz_full = ms_get_freq_axis_hz(ms_path, ddid)
    dnu = np.diff(freqs_hz_full)
    df_nom = np.median(dnu)
    corr_idx = ms_get_corr_index(ms_path, corr)
    have_flag = "FLAG" in t.colnames()

    total, flagged = 0, 0
    times_list, F_rows = [], []
    step = 100000
    for start in range(0, q.nrows(), step):
        nr = min(step, q.nrows() - start)
        data = q.getcol(col, start, nr)[:, :, corr_idx]
        flags = q.getcol('FLAG', start, nr)[:, :, corr_idx] if have_flag else np.zeros_like(data, bool)
        time = q.getcol('TIME', start, nr)
        total += flags.size
        flagged += np.sum(flags)

        # Channel binning
        if chanbin > 1:
            nch2 = (data.shape[1] // chanbin) * chanbin
            data = data[:, :nch2].reshape(nr, -1, chanbin).mean(axis=2)
            flags = flags[:, :nch2].reshape(nr, -1, chanbin).any(axis=2)
            freqs_hz = freqs_hz_full[:nch2].reshape(-1, chanbin).mean(axis=1)
            df_eff = np.median(np.diff(freqs_hz))
        else:
            freqs_hz, df_eff = freqs_hz_full, df_nom

        # Time binning
        if timebin > 1 and data.shape[0] >= timebin:
            k = (data.shape[0] // timebin) * timebin
            data = data[:k].reshape(-1, timebin, data.shape[1]).mean(axis=1)
            flags = flags[:k].reshape(-1, timebin, flags.shape[1]).any(axis=1)
            time  = time[:k].reshape(-1, timebin).mean(axis=1)

        vis_block = data if ignore_flags else np.where(~flags, data, 0.0+0.0j)
        tau_s, F_block = fft_delay_complex(vis_block, df_eff)
        F_rows.append(F_block)
        times_list.append(time)

    q.close(); t.close()
    flag_pct = flagged / total * 100 if total > 0 else 0
    print(f"[INFO] Flagged samples: {flag_pct:.2f}% ({'ignored' if ignore_flags else 'zeroed'})")

    F_t_tau = np.vstack(F_rows)
    times_all = np.concatenate(times_list)
    time_sec = times_all - np.nanmin(times_all)
    delay_us = tau_s * 1e6
    return time_sec, delay_us, F_t_tau, f"{lo}-{hi}", corr.upper()


def load_uvfits_delay_waterfall(uvfits_path, ant1, ant2, corr,
                                timebin=1, chanbin=1, ignore_flags=False):
    if not PYUVDATA_AVAILABLE:
        raise RuntimeError("pyuvdata required for UVFITS input.")
    uv = UVData(); uv.read(uvfits_path)

    def uv_ant_to_index(uvo, a):
        try:
            return int(a)
        except Exception:
            names = np.array(uvo.antenna_names)
            where = np.where(names == a)[0]
            if where.size == 0:
                raise RuntimeError(f"Antenna '{a}' not found in UVFITS.")
            return int(where[0])

    a1 = uv_ant_to_index(uv, ant1)
    a2 = uv_ant_to_index(uv, ant2)
    pols_available = uv.get_pols()
    chosen_pol_str = resolve_polarization(corr, pols_available)
    pol_code = uv.polarization_array[list(pols_available).index(chosen_pol_str)]

    vis = uv.get_data((a1, a2, pol_code))
    flags = uv.get_flags((a1, a2, pol_code))
    freqs_hz_full = np.asarray(uv.freq_array[0] if uv.freq_array.ndim == 2 else uv.freq_array, float)

    if chanbin > 1:
        nch2 = (vis.shape[1] // chanbin) * chanbin
        vis = vis[:, :nch2].reshape(vis.shape[0], -1, chanbin).mean(axis=2)
        flags = flags[:, :nch2].reshape(flags.shape[0], -1, chanbin).any(axis=2)
        freqs_hz = freqs_hz_full[:nch2].reshape(-1, chanbin).mean(axis=1)
    else:
        freqs_hz = freqs_hz_full

    times_jd = uv.get_times((a1, a2, pol_code))
    if timebin > 1 and vis.shape[0] >= timebin:
        k = (vis.shape[0] // timebin) * timebin
        vis = vis[:k].reshape(-1, timebin, vis.shape[1]).mean(axis=1)
        flags = flags[:k].reshape(-1, timebin, flags.shape[1]).any(axis=1)
        times_jd = times_jd[:k].reshape(-1, timebin).mean(axis=1)

    vis_proc = vis if ignore_flags else np.where(~flags, vis, 0.0+0.0j)
    df_nom = np.median(np.diff(freqs_hz))
    tau_s, F_t_tau = fft_delay_complex(vis_proc, df_nom)
    delay_us = tau_s * 1e6
    time_sec = (times_jd - np.min(times_jd)) * 86400.0
    polname = chosen_pol_str.upper()
    if polname != corr.upper():
        print(f"[INFO] Using polarization '{polname}' as fallback for requested '{corr}'")
    return time_sec, delay_us, F_t_tau, f"{a1}-{a2}", polname


# ----------------------------
# Print delay-spectra statistics
# ----------------------------
def print_delay_statistics(delay_us, F_t_tau, input_file, data_type, bl, pol, timebin, chanbin):
    AMP = np.abs(F_t_tau)
    POWER = AMP**2
    amp_max = np.nanmax(AMP)
    amp_mean = np.nanmean(AMP)
    amp_med = np.nanmedian(AMP)
    power_total = np.nansum(POWER)
    avg_amp_delay = np.nanmean(AMP, axis=0)
    peak_delay_us = delay_us[np.nanargmax(avg_amp_delay)]

    print("\n[SUMMARY]")
    print(f"  Input file     : {input_file}")
    print(f"  Data type      : {data_type}")
    print(f"  Baseline       : {bl}")
    print(f"  Polarization   : {pol}")
    print(f"  Time binning   : {timebin}")
    print(f"  Channel binning: {chanbin}\n")
    print("[DELAY SPECTRUM STATISTICS]")
    print(f"  |FFT| Max      = {amp_max:.3e}")
    print(f"  |FFT| Mean     = {amp_mean:.3e}")
    print(f"  |FFT| Median   = {amp_med:.3e}")
    print(f"  Σ|FFT|²        = {power_total:.3e}")
    print(f"  Peak delay (µs)= {peak_delay_us:.3f}\n")


# ----------------------------
# Plotting
# ----------------------------
def plot_three_page_pdf(time_sec, delay_us, F_t_tau, title_base, out_pdf,
                        vmin=None, vmax=None, log=True, show=False):
    AMP = np.abs(F_t_tau)
    POWER = AMP**2
    PHASE = np.angle(F_t_tau)
    amp_vmin, amp_vmax = robust_percentiles(AMP)
    pow_vmin, pow_vmax = robust_percentiles(POWER)
    if vmin is not None:
        amp_vmin = pow_vmin = max(vmin, 1e-12)
    if vmax is not None:
        amp_vmax = pow_vmax = max(vmax, amp_vmin*1.01)
    t_edges, d_edges = to_edges(time_sec), to_edges(delay_us)

    if not show:
        with PdfPages(out_pdf) as pdf:
            # Page 1: log|FFT| & Phase
            fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.8, 8.4), sharex=True)
            mesh1 = ax1.pcolormesh(t_edges, d_edges, AMP.T, shading="auto",
                                norm=LogNorm(vmin=max(amp_vmin,1e-12), vmax=amp_vmax),
                                cmap="Spectral_r")
            plt.colorbar(mesh1, ax=ax1, label="log |FFT| (arb.)")
            ax1.set_ylabel("Delay [µs]")
            ax1.set_title(f"{title_base} — log(|FFT|) & Phase")
            mesh2 = ax2.pcolormesh(t_edges, d_edges, PHASE.T, shading="auto",
                                cmap="twilight", vmin=-np.pi, vmax=np.pi)
            plt.colorbar(mesh2, ax=ax2, label="Phase ∠FFT [rad]")
            ax2.set_xlabel("Time [s]")
            ax2.set_ylabel("Delay [µs]")
            fig1.tight_layout()
            pdf.savefig(fig1); plt.close(fig1)

            # Page 2: |FFT|
            fig2 = plt.figure(figsize=(7.8, 5.8))
            ax = fig2.add_subplot(111)
            norm = LogNorm(vmin=max(amp_vmin,1e-12), vmax=amp_vmax) if log else None
            mesh = ax.pcolormesh(t_edges, d_edges, AMP.T, shading="auto", norm=norm, cmap="Spectral_r")
            plt.colorbar(mesh, ax=ax, label="|FFT| (arb.)")
            ax.set_xlabel("Time [s]"); ax.set_ylabel("Delay [µs]")
            ax.set_title(f"{title_base} — |FFT|")
            fig2.tight_layout(); pdf.savefig(fig2); plt.close(fig2)

            # Page 3: |FFT|²
            fig3 = plt.figure(figsize=(7.8, 5.8))
            ax = fig3.add_subplot(111)
            norm = LogNorm(vmin=max(pow_vmin,1e-12), vmax=pow_vmax) if log else None
            mesh = ax.pcolormesh(t_edges, d_edges, POWER.T, shading="auto", norm=norm, cmap="Spectral_r")
            plt.colorbar(mesh, ax=ax, label="|FFT|² (arb.)")
            ax.set_xlabel("Time [s]"); ax.set_ylabel("Delay [µs]")
            ax.set_title(f"{title_base} — |FFT|²")
            fig3.tight_layout(); pdf.savefig(fig3); plt.close(fig3)

        print(f"[INFO] Saved 3-page PDF: {out_pdf}")
    else:
        # Display interactively, one page at a time
        # Enable interactive mode for non-blocking display
        plt.ion()
        
        # Page 1: log|FFT| & Phase
        fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.8, 8.4), sharex=True)
        mesh1 = ax1.pcolormesh(t_edges, d_edges, AMP.T, shading="auto",
                            norm=LogNorm(vmin=max(amp_vmin,1e-12), vmax=amp_vmax),
                            cmap="Spectral_r")
        plt.colorbar(mesh1, ax=ax1, label="log |FFT| (arb.)")
        ax1.set_ylabel("Delay [µs]")
        ax1.set_title(f"{title_base} — log(|FFT|) & Phase")
        mesh2 = ax2.pcolormesh(t_edges, d_edges, PHASE.T, shading="auto",
                            cmap="twilight", vmin=-np.pi, vmax=np.pi)
        plt.colorbar(mesh2, ax=ax2, label="Phase ∠FFT [rad]")
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Delay [µs]")
        fig1.tight_layout()
        plt.show(block=False)  # Non-blocking, show first plot
        #plt.close(fig1)

        # Page 2: |FFT|
        fig2 = plt.figure(figsize=(7.8, 5.8))
        ax = fig2.add_subplot(111)
        norm = LogNorm(vmin=max(amp_vmin,1e-12), vmax=amp_vmax) if log else None
        mesh = ax.pcolormesh(t_edges, d_edges, AMP.T, shading="auto", norm=norm, cmap="Spectral_r")                                                             
        plt.colorbar(mesh, ax=ax, label="|FFT| (arb.)")
        ax.set_xlabel("Time [s]"); ax.set_ylabel("Delay [µs]")
        ax.set_title(f"{title_base} — |FFT|")
        fig2.tight_layout()
        plt.show(block=False)  # Non-blocking, show second plot
       # plt.close(fig2)

        # Page 3: |FFT|²
        fig3 = plt.figure(figsize=(7.8, 5.8))
        ax = fig3.add_subplot(111)
        norm = LogNorm(vmin=max(pow_vmin,1e-12), vmax=pow_vmax) if log else None
        mesh = ax.pcolormesh(t_edges, d_edges, POWER.T, shading="auto", norm=norm, cmap="Spectral_r")
        plt.colorbar(mesh, ax=ax, label="|FFT|² (arb.)")
        ax.set_xlabel("Time [s]"); ax.set_ylabel("Delay [µs]")
        ax.set_title(f"{title_base} — |FFT|²")
        fig3.tight_layout()
        plt.show(block=True)  # Block until all windows are closed
        print("[INFO] Displayed 3 plots interactively. Close the windows to continue.")
        # plt.close(fig3)


# ----------------------------
# Config-based execution function
# ----------------------------
def execute_delay_time_waterfall(
    config_file="config.yaml",
    input_file=None,
    ant1=None,
    ant2=None,
    corr=None,
    col=None,
    ddid=None,
    list_baselines=None,
    ignore_flags=None,
    timebin=None,
    chanbin=None,
    vmin=None,
    vmax=None,
    no_log=None,
    plot_title_base=None,
    out_file=None,
    show=None,
):
    """
    Executes the delay–time waterfall plotting for MS or UVFITS files.

    Args:
        config_file (str): Path to YAML config file (default: "config.yaml").
            If given, all config values loaded from file unless individually overridden by keyword args.                                                        
        Other keyword arguments override values from config when not None.
    """
    # --- Load config file and parameters on each function execution ---
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"[ERROR] Unable to read configuration file: {config_file}: {e}")
        return

    # Helper: get override if specified, else from config, else default
    def get_arg(arg, key, fallback=None):
        return arg if arg is not None else config.get(key, fallback)

    # Assign parameters using overrides, config, fallback
    input_file      = get_arg(input_file,      "input_file",      "")
    ant1            = get_arg(ant1,            "ant1",            "2")
    ant2            = get_arg(ant2,            "ant2",            "6")
    corr            = get_arg(corr,            "corr",            "XX")
    col             = get_arg(col,             "col",             "DATA")
    ddid            = get_arg(ddid,            "ddid",            None)
    list_baselines  = get_arg(list_baselines,  "list_baselines",  False)
    ignore_flags    = get_arg(ignore_flags,    "ignore_flags",    False)
    timebin         = get_arg(timebin,         "timebin",         1)
    chanbin         = get_arg(chanbin,         "chanbin",         1)
    vmin            = get_arg(vmin,            "vmin",            None)
    vmax            = get_arg(vmax,            "vmax",            None)
    no_log          = get_arg(no_log,          "no_log",          False)
    plot_title_base = get_arg(plot_title_base, "plot_title_base", "Delay–Time Waterfall")                                                                       
    out_file        = get_arg(out_file,        "out_file",        "delay_waterfall_full.pdf")                                                                   
    show            = get_arg(show,            "show",            True)

    # For unexposed config params (not passed in function signature), still use config                                                                          
    outdir   = config.get("outdir", "./")

    # Prepend outdir to output file path if relative
    if not os.path.isabs(out_file):
        out_file = os.path.join(outdir, out_file)

    try:
        # MS or UVFITS?
        if os.path.isdir(input_file):
            if list_baselines:
                print(f"Listing baselines for {input_file}...")
                ms_list_baselines(input_file)
            else:
                print(f"Loading MS: {input_file}")
                time_sec, delay_us, F_t_tau, bl, pol = load_ms_delay_waterfall(
                    input_file, ant1, ant2, corr,
                    col=col, ddid_override=ddid,
                    timebin=timebin, chanbin=chanbin,
                    ignore_flags=ignore_flags
                )
                data_type = "MeasurementSet (MS)"
                
                # Statistics
                print_delay_statistics(delay_us, F_t_tau, input_file, data_type,
                                     bl, pol, timebin, chanbin)

                out_pdf = out_file if out_file.endswith(".pdf") else out_file + ".pdf"                                                                          
                base_title = plot_title_base if plot_title_base else f"Delay–Time Waterfall: {bl}, {pol}"                                                       

                plot_three_page_pdf(
                    time_sec, delay_us, F_t_tau, base_title, out_pdf,
                    vmin=vmin, vmax=vmax,
                    log=(not no_log), show=show
                )

        elif os.path.isfile(input_file):
            print(f"Loading UVFITS: {input_file}")
            time_sec, delay_us, F_t_tau, bl, pol = load_uvfits_delay_waterfall(
                input_file, ant1, ant2, corr,
                timebin=timebin, chanbin=chanbin,
                ignore_flags=ignore_flags
            )
            data_type = "UVFITS"
            
            # Statistics
            print_delay_statistics(delay_us, F_t_tau, input_file, data_type,
                                 bl, pol, timebin, chanbin)

            out_pdf = out_file if out_file.endswith(".pdf") else out_file + ".pdf"
            base_title = plot_title_base if plot_title_base else f"Delay–Time Waterfall: {bl}, {pol}"

            # Only output PDF if out_pdf is defined and non-empty (not None, not blank/whitespace)
            plot_three_page_pdf(
                time_sec, delay_us, F_t_tau, base_title, out_pdf,
                vmin=vmin, vmax=vmax,
                log=(not no_log), show=show
            )
        else:
            if not list_baselines:
                print(f"[ERROR] Input file not found or is not a directory (MS) or file (UVFITS): {input_file}")                                                

    except (RuntimeError, ImportError, FileNotFoundError, NameError) as e:
        print(f"\n[ERROR] An error occurred: {e}")
        print("Please check your parameters, file paths, and required libraries (casacore, pyuvdata).")


# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Delay–time waterfall (PDF + statistics). "
                    "Parameters can be provided via YAML config file (--config) or command-line arguments. "
                    "Command-line arguments override config file values."
    )
    ap.add_argument("--config", default="config.yaml", 
                    help="Path to YAML config file containing parameters (default: config.yaml). "
                         "All parameters can be specified in the config file and overridden individually via command-line arguments.")
    ap.add_argument("input_file", nargs='?', help="MS directory or UVFITS file (overrides YAML if given)")
    ap.add_argument("--ant1", help="Antenna 1 (overrides YAML)")
    ap.add_argument("--ant2", help="Antenna 2 (overrides YAML)")
    ap.add_argument("--corr", help="Polarization/correlation (overrides YAML)")
    ap.add_argument("--col", help="MS data column (overrides YAML)")
    ap.add_argument("--ddid", type=int, help="Data Desc ID (overrides YAML)")
    ap.add_argument("--ignore-flags", action="store_true", help="Ignore flags (overrides YAML)")
    ap.add_argument("--timebin", type=int, help="Time bin (overrides YAML)")
    ap.add_argument("--chanbin", type=int, help="Channel bin (overrides YAML)")
    ap.add_argument("--vmin", type=float, help="Color minimum (overrides YAML)")
    ap.add_argument("--vmax", type=float, help="Color maximum (overrides YAML)")
    ap.add_argument("--no-log", action="store_true", help="Disable log scale (overrides YAML)")
    ap.add_argument("--title", help="Plot title (overrides YAML)")
    ap.add_argument("--out", help="PDF output name (overrides YAML)")
    args = ap.parse_args()

    # Load config if it exists
    config = {}
    if args.config and os.path.isfile(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

    # Build options, prioritizing command line args over config.yaml
    def get_parm(key, default=None, always_strip=False):
        argval = getattr(args, key, None)
        if argval is not None:
            if always_strip and isinstance(argval, str):
                return argval.strip()
            return argval
        yamlval = config.get(key)
        if yamlval is not None:
            return yamlval
        return default

    # Support both possible ways input_file might be positioned
    input_file = args.input_file if args.input_file else config.get("input_file")
    ant1 = get_parm("ant1")
    ant2 = get_parm("ant2")
    corr = get_parm("corr", "XX")
    col = get_parm("col", "DATA")
    ddid = get_parm("ddid", None)
    ignore_flags = args.ignore_flags or config.get("ignore_flags", False)
    timebin = get_parm("timebin", 1)
    chanbin = get_parm("chanbin", 1)
    vmin = get_parm("vmin")
    vmax = get_parm("vmax")
    no_log = args.no_log or config.get("no_log", False)
    plot_title_base = get_parm("title", config.get("plot_title_base", "Delay–Time Waterfall"))
    # Check for "out" (CLI arg name) or "out_file" (config key name)
    out_file = get_parm("out")
    if out_file is None:
        out_file = config.get("out_file", "delay_waterfall.pdf")
    # Execute core function
    execute_delay_time_waterfall(
        config_file=args.config,
        input_file=input_file,
        ant1=ant1,
        ant2=ant2,
        corr=corr,
        col=col,
        ddid=ddid,
        list_baselines=None,  # Can add --list-baselines arg if needed
        ignore_flags=ignore_flags,
        timebin=timebin,
        chanbin=chanbin,
        vmin=vmin,
        vmax=vmax,
        no_log=no_log,
        plot_title_base=plot_title_base,
        out_file=out_file,
        show=None
    )


if __name__ == "__main__":
    main()
