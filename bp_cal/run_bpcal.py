#!/usr/bin/env python3
import sys
import shutil
import subprocess
from pathlib import Path


def run_cmd(cmd, log_file, cwd=None):
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


def write_parset(path, lines_dict):
    with open(path, "w") as f:
        for k, v in lines_dict.items():
            f.write(f"{k} = {v}\n")
    print(f"[WRITE] {path}")


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


def load_yaml_config(path):
    cfg = {}
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue

            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()

            if (val.startswith('"') and val.endswith('"')) or \
               (val.startswith("'") and val.endswith("'")):
                val = val[1:-1]

            cfg[key] = val
    return cfg


def main(yaml_file):
    cfg = load_yaml_config(yaml_file)

    datafolder = Path(cfg["datafolder"]).expanduser().resolve()
    datafile   = cfg["datafile"]
    input_col  = cfg.get("input_datacol", "DATA")
    output_col = cfg.get("output_datacol", "BP_CORRECTED")

    skymodel   = cfg["skymodelfile"]
    max_bl     = float(cfg.get("max_bl_len", 0))
    min_bl     = float(cfg.get("min_bl_len", 0))
    avg_ts     = int(cfg.get("avg_time_step", 1))
    solint     = int(cfg.get("time_sol_int", 450))

    ms_path = datafolder / datafile
    if not ms_path.exists():
        raise FileNotFoundError(f"MS not found: {ms_path}")

    run_dir = make_run_dir(yaml_file)

    parmdb_name = "bandpass-gaincal.h5"
    parmdb_path = run_dir / parmdb_name
    avg_ms      = run_dir / "temp_avg.MS"

    gain_parset = run_dir / "DI_gaincal.parset"
    gain_lines = {
        "numthreads": 200,
        "msin": str(ms_path),
        "msin.datacolumn": input_col,
        "msout": str(avg_ms),
        "msout.overwrite": "true",
        "steps": "[averager, gaincal]",
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

    write_parset(gain_parset, gain_lines)

    log_gain = run_dir / "gaincal.log"
    run_cmd(["DP3", str(gain_parset)], log_gain, cwd=run_dir)

    # ================================================================
    # STEP 3 — APPLYCAL on original MS
    # ================================================================
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

    log_apply = run_dir / "applycal.log"
    run_cmd(["DP3", str(apply_parset)], log_apply, cwd=run_dir)

    # ================================================================
    # CLEANUP
    # ================================================================
    if avg_ms.exists():
        shutil.rmtree(avg_ms)
        print(f"[CLEANUP] Removed temporary averaged MS: {avg_ms}")

    print("\n✓ All steps completed successfully.")
    print(f"[INFO] All outputs stored in: {run_dir}")


# -----------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_bpcal.py config.yaml")
        sys.exit(1)
    main(sys.argv[1])

