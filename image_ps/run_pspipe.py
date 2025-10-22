#!/usr/bin/env python3
import os
import shutil
import click
import yaml
import subprocess
from pathlib import Path
from casacore import tables

# === Constants ===
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pspipe_templates')


def run_cmd(cmd):
    """Run a shell command and raise an error if it fails."""
    click.echo(f"Running: {cmd}")
    subprocess.check_call(cmd, shell=True)


def get_ms_freqs(ms_file):
    """Read frequencies and channel widths from the MS SPECTRAL_WINDOW table."""
    with tables.table(os.path.join(ms_file, 'SPECTRAL_WINDOW'), readonly=True, ack=False) as t_spec_win:
        freqs = t_spec_win.getcol('CHAN_FREQ').reshape(-1)
        chan_widths = t_spec_win.getcol('CHAN_WIDTH').reshape(-1)
    return freqs, chan_widths


def get_channel_range(ms_path, freq_range):
    """Return (c_start, c_end) indices for the requested freq_range (MHz)."""
    import numpy as np
    freqs, _ = get_ms_freqs(str(ms_path))
    freqs_mhz = freqs / 1e6
    fmin, fmax = freq_range
    c_start = int((np.abs(freqs_mhz - fmin)).argmin())
    c_end   = int((np.abs(freqs_mhz - fmax)).argmin()) + 1
    if c_start > c_end:
        c_start, c_end = c_end, c_start
    print(c_start, c_end, len(freqs))
    return c_start, c_end


def load_config(config_file):
    """Load and validate the YAML config file."""
    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)
    required = ["datafolder", "datafile", "instrument", "freq_range", "data_col", "pol"]
    for key in required:
        if key not in cfg:
            raise click.ClickException(f"Missing required key in config: {key}")
    return cfg


def make_run_dir():
    """Create run$i directory with increment until not existing, and cd into it."""
    i = 1
    while True:
        run_dir = Path(f"pspipe_workdir_{i}")
        if not run_dir.exists():
            run_dir.mkdir()
            os.chdir(run_dir)
            return run_dir
        i += 1


@click.command()
@click.argument("config_file", type=click.Path(exists=True))
def main(config_file):
    """Produce power spectra from an MS using psdb/pspipe, driven by a YAML config."""
    # derive root from config file location
    config_path = Path(config_file).resolve()
    root = config_path.parent
    cfg = load_config(config_path)

    datafolder = (root / cfg["datafolder"]).resolve()
    datafile   = cfg["datafile"]
    dataset    = cfg["instrument"]
    freq_range = cfg["freq_range"]
    data_col   = cfg["data_col"]
    pol        = cfg["pol"]

    ms_path = (datafolder / datafile).resolve()
    if not ms_path.exists():
        raise click.ClickException(f"MS not found: {ms_path}")
    if not ms_path.is_dir():
        raise click.ClickException(f"Expected MS directory, got a file: {ms_path}")

    # create run$i dir and work inside
    run_dir = make_run_dir()
    click.echo(f"Working in {run_dir}")

    # copy config file into run dir for record-keeping
    # print(config_path, run_dir / Path(config_file).name)
    # shutil.copy(config_path, run_dir / Path(config_file).name)

    obs_id = ms_path.stem
    template_toml = f"{TEMPLATE_DIR}/default_{dataset.lower()}.toml"
    rev_toml = f"default_{dataset.lower()}.toml"

    # Step 1: clone with modifiers
    c_start, c_end = get_channel_range(ms_path, freq_range)
    pol_clean = "".join(pol.replace(" ", "").split(","))  # e.g. "I,V" -> "IV"
    modifiers = [
        f"\"image.data_col='{data_col}'\"",
        f"\"image.wsclean_args.channel-range='{c_start} {c_end}'\"",
        f"\"image.stokes='{pol_clean}'\""
    ]
    mods_str = " ".join(modifiers)
    run_cmd(f"psdb clone {template_toml} data_dir {mods_str}")
    # Step 2: create ms_list entry
    Path("ms_lists").mkdir(exist_ok=True)
    with open(f"ms_lists/{obs_id}", "w") as f:
        f.write(str(ms_path) + "\n")

    # Step 3: add obs
    run_cmd(f"psdb add_all_obs {rev_toml} ms_lists/{obs_id}")

    # Step 4: run pspipe
    run_cmd(f"pspipe image,gen_vis_cube {rev_toml} {obs_id}")

    # Step 5: call make_ps.py
    make_ps = Path(__file__).resolve().parent / "make_ps.py"
    fmin, fmax = freq_range
    run_cmd(f"{make_ps} {rev_toml} {obs_id} --fmin {fmin} --fmax {fmax} --pol {pol}")


if __name__ == "__main__":
    main()
