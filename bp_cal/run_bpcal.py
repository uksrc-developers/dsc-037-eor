#!/usr/bin/env python3
import yaml
import subprocess
from pathlib import Path
import shutil
import sys


# -----------------------------------------------------------
# Utility: run shell command with logging (tee style)
# -----------------------------------------------------------
def run_cmd(cmd, log_file, cwd=None):
    """Run command, print output live, and tee into log file."""
    print(f"[RUN] {' '.join(cmd)}   (log → {log_file})")

    with open(log_file, "w") as lf:
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in process.stdout:
            print(line, end="")
            lf.write(line)

    ret = process.wait()
    if ret != 0:
        raise RuntimeError(f"Command failed with exit code {ret}. See log: {log_file}")


# -----------------------------------------------------------
# Utility: write DP3 parset
# -----------------------------------------------------------
def write_parset(path, lines_dict):
    with open(path, "w") as f:
        for k, v in lines_dict.items():
            f.write(f"{k} = {v}\n")
    print(f"[WRITE] {path}")


# -----------------------------------------------------------
# Create run directory next to YAML, with _i suffix if exists
# -----------------------------------------------------------
def make_run_dir(yaml_file):
    config_path = Path(yaml_file).resolve()
    base_name = config_path.with_suffix("").name
    base_dir = config_path.parent

    run_dir = base_dir / f"{base_name}_run"
    i = 1
    while run_dir.exists():
        run_dir = base_dir / f"{base_name}_run_{i}"
        i += 1

    run_dir.mkdir()
    print(f"[INFO] Created run directory: {run_dir}")
    return run_dir


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
def main(yaml_file):

    cfg = yaml.safe_load(open(yaml_file))

    datafolder = Path(cfg["datafolder"]).expanduser().resolve()
    datafile   = cfg["datafile"]
    input_col  = cfg.get("input_datacol", "DATA")
    output_col = cfg.get("output_datacol", "BP_CORRECTED")

    skymodel   = cfg.get("skymodelfile", None)
    max_bl     = float(cfg.get("max_bl_len", 0))
    min_bl     = float(cfg.get("min_bl_len", 0))
    avg_ts     = int(cfg.get("avg_time_step", 1))
    solint     = int(cfg.get("time_sol_int", 450))

    if skymodel is None:
        raise ValueError("Missing required key: skymodelfile")

    ms_path = datafolder / datafile
    if not ms_path.exists():
        raise FileNotFoundError(f"MS not found: {ms_path}")

    # ----------------------------------------------
    # Create run directory
    # ----------------------------------------------
    run_dir = make_run_dir(yaml_file)

    # Temporary MS
    temp_ms = run_dir / "temp_output.MS"

    # Parmdb
    parmdb_name = "bandpass-gaincal.h5"
    parmdb_path = run_dir / parmdb_name

    # ----------------------------------------------
    # Build DI_bandpass_cal.parset
    # ----------------------------------------------
    bandpass_parset = run_dir / "DI_bandpass_cal.parset"
    bandpass_lines = {
        "numthreads": 200,
        "msin": str(ms_path),
        "msin.datacolumn": input_col,
        "msout": str(temp_ms),
        "msout.overwrite": "true",

        "steps": "[averager,gaincal]",
        "averager.timestep": avg_ts,

        "gaincal.usemodelcolumn": "false",
        "gaincal.usebeammodel": "true",
        "gaincal.beammode": "full",
        "gaincal.beamproximitylimit": 600,
        "gaincal.caltype": "diagonal",
        "gaincal.solint": solint,
        "gaincal.nchan": 1,
        "gaincal.maxiter": 50,
        "gaincal.propagatesolutions": "true",

        "gaincal.sourcedb": str(Path(skymodel).resolve()),
        "gaincal.parmdb": parmdb_name,
    }

    if min_bl != max_bl:
        bandpass_lines["gaincal.uvlambdamin"] = min_bl
        bandpass_lines["gaincal.uvlambdamax"] = max_bl
    else:
        print("[INFO] min_bl == max_bl → no baseline selection applied.")

    write_parset(bandpass_parset, bandpass_lines)

    # ----------------------------------------------
    # Run DP3 bandpass calibration
    # ----------------------------------------------
    log_bp = run_dir / "bandpass_cal.log"
    print("\n[STEP] Running DI_bandpass_cal…")
    run_cmd(["DP3", str(bandpass_parset)], log_bp, cwd=run_dir)

    print(f"[INFO] Parmdb generated: {parmdb_path}")

    # ----------------------------------------------
    # Build DI_applycal.parset
    # ----------------------------------------------
    apply_parset = run_dir / "DI_applycal.parset"
    apply_lines = {
        "msin": str(ms_path),
        "msin.datacolumn": input_col,
        "msout": str(ms_path),
        "msout.datacolumn": output_col,

        "steps": "[applycal]",
        "applycal.steps": "[amp,phase]",
        "applycal.amp.correction": "amplitude000",
        "applycal.phase.correction": "phase000",
        "applycal.parmdb": parmdb_name,
    }

    write_parset(apply_parset, apply_lines)

    # ----------------------------------------------
    # Run DP3 applycal
    # ----------------------------------------------
    log_apply = run_dir / "applycal.log"
    print("\n[STEP] Running DI_applycal…")
    run_cmd(["DP3", str(apply_parset)], log_apply, cwd=run_dir)

    # ----------------------------------------------
    # CLEANUP temp MS
    # ----------------------------------------------
    if temp_ms.exists():
        shutil.rmtree(temp_ms)
        print(f"[CLEANUP] Removed temporary MS: {temp_ms}")

    print("\n✓ All steps completed successfully.")
    print(f"[INFO] All outputs stored in: {run_dir}")


# -----------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python dsc037_run_bpcal.py config.yaml")
        sys.exit(1)
    main(sys.argv[1])

