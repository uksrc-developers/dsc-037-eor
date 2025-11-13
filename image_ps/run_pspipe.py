#!/usr/bin/env python3
# **pspipe_run**: Automated psdb/pspipe workflow for power spectrum generation
#
# **Author:** Florent Mertens and the Teal team
#
# **Documentation:**
# - DSC description page: https://confluence.skatelescope.org/x/0rs6F
# - Chronological walkthrough: https://confluence.skatelescope.org/x/osw6F
# - Implementation: https://confluence.skatelescope.org/x/n8LMF
# - GitHub repo: https://github.com/uksrc-developers/dsc-037-eor
# - pspipe: https://gitlab.com/flomertens/pspipe/
#
# **Summary:**  
# This script automates the generation of power spectra from Measurement Sets (MS)
# using `pspipe`. It reads a YAML configuration file specifying data location, 
# frequency range, data column, and polarisation, detects the telescope
# from the MS metadata, clones and modifies a TOML template, and executes the
# corresponding pspipe workflow.
#
# The script:
# - Loads configuration from YAML  
# - Detects telescope name and selects the correct template  
# - Creates a working directory (`pspipe_workdir_i/`)  
# - Runs `psdb clone`, `psdb add_all_obs`, and `pspipe image,gen_vis_cube`  
# - Calls `make_ps.py` to compute power spectra  
#
# **Tickets:** TEAL-1128, TEAL-1155
#

import os
import click
import yaml
import subprocess
from pathlib import Path
from casacore import tables


TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pspipe_templates')


def run_cmd(cmd):
    """Execute a shell command with click output and error propagation."""
    click.echo(f"Running: {cmd}")
    subprocess.check_call(cmd, shell=True)


def get_telescope_name(ms_path):
    """Return telescope name from the OBSERVATION table of a Measurement Set."""
    with tables.table(f"{ms_path}/OBSERVATION", readonly=True) as t:
        return t.getcol('TELESCOPE_NAME')[0]


def get_ms_freqs(ms_file):
    """Return frequency array and channel widths from an MS SPECTRAL_WINDOW table."""
    with tables.table(os.path.join(ms_file, 'SPECTRAL_WINDOW'), readonly=True, ack=False) as t_spec_win:
        freqs = t_spec_win.getcol('CHAN_FREQ').reshape(-1)
        chan_widths = t_spec_win.getcol('CHAN_WIDTH').reshape(-1)
    return freqs, chan_widths


def get_channel_range(ms_path, freq_range):
    """Compute start/end channel indices for a given frequency range (MHz)."""
    import numpy as np
    freqs, _ = get_ms_freqs(str(ms_path))
    freqs_mhz = freqs / 1e6
    fmin, fmax = freq_range
    c_start = int((np.abs(freqs_mhz - fmin)).argmin())
    c_end   = int((np.abs(freqs_mhz - fmax)).argmin()) + 1
    if c_start > c_end:
        c_start, c_end = c_end, c_start
    return c_start, c_end


def load_config(config_file):
    """Load YAML configuration and check required keys."""
    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)
    required = ["datafolder", "datafile", "freq_range", "data_col", "pol"]
    for key in required:
        if key not in cfg:
            raise click.ClickException(f"Missing required key in config: {key}")
    return cfg


def make_run_dir(config_path, workdir=None):
    """Create or use a working directory based on the config filename.
    If a directory already exists, append _i automatically."""
    if workdir:
        run_dir = Path(workdir)
    else:
        base_name = Path(config_path).stem
        run_dir = Path(f"{base_name}")
        i = 1
        while run_dir.exists():
            run_dir = Path(f"{base_name}_{i}")
            i += 1
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(run_dir)
    click.echo(f"Working directory set to: {run_dir}")
    return run_dir


@click.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--workdir", "-w", type=click.Path(), help="Optional working directory.")
def main(config_file, workdir):
    """Run the pspipe workflow for the dataset described in CONFIG_FILE."""
    config_path = Path(config_file).resolve()
    root = config_path.parent
    cfg = load_config(config_path)

    datafolder = (root / cfg["datafolder"]).resolve()
    datafile   = cfg["datafile"]
    freq_range = cfg["freq_range"]
    data_col   = cfg["data_col"]
    pol        = cfg["pol"]

    ms_path = (datafolder / datafile).resolve()
    if not ms_path.exists() or not ms_path.is_dir():
        raise click.ClickException(f"Invalid Measurement Set path: {ms_path}")

    telescope = get_telescope_name(ms_path)
    run_dir = make_run_dir(config_path, workdir)
    click.echo(f"Working in {run_dir}")

    obs_id = ms_path.stem
    template_toml = f"{TEMPLATE_DIR}/default_{telescope.lower()}.toml"
    rev_toml = f"default_{telescope.lower()}.toml"

    # Apply modifiers
    c_start, c_end = freq_range
    freqs, _ = get_ms_freqs(str(ms_path))
    freqs_mhz = freqs / 1e6

    if c_end == 0:
        c_end = len(freqs) - 1
    c_out = c_end - c_start

    fmin = freqs_mhz[c_start]
    fmax = freqs_mhz[c_end]

    if isinstance(pol, str):
        pol = pol.replace(" ", "").split(",")

    pol_clean = ",".join(pol)

    # --- Print main parameters ---
    click.echo("\n--- Main parameters ---")
    click.echo(f"Measurement Set path  : {ms_path}")
    click.echo(f"Telescope             : {telescope.lower()}")
    click.echo(f"Observation ID        : {obs_id}")
    click.echo(f"Polarisation(s)       : {pol_clean}")
    click.echo(f"Frequency range (MHz) : {fmin:.2f} – {fmax:.2f}")
    click.echo(f"Channel range         : {c_start} – {c_end}  ({c_out} channels)")
    click.echo("------------------------\n")

    modifiers = [
        f"\"image.data_col='{data_col}'\"",
        f"\"image.wsclean_args.channel-range='{c_start} {c_end}'\"",
        f"\"image.channels_out='{c_out}'\"",
        f"\"image.stokes='{pol_clean.replace(',', '')}'\"",
    ]
    run_cmd(f"psdb clone {template_toml} data_dir {' '.join(modifiers)}")

    Path("ms_lists").mkdir(exist_ok=True)
    with open(f"ms_lists/{obs_id}", "w") as f:
        f.write(str(ms_path) + "\n")

    run_cmd(f"psdb add_all_obs {rev_toml} ms_lists/{obs_id}")
    run_cmd(f"pspipe image,gen_vis_cube {rev_toml} {obs_id}")

    make_ps = Path(__file__).resolve().parent / "make_ps.py"
    run_cmd(f"{make_ps} {rev_toml} {obs_id} --fmin {fmin} --fmax {fmax} --pol {pol_clean}")


if __name__ == "__main__":
    main()
