# DSC-037: Cable Reflection Systematics for EoR Science

This repository contains the implementation code for **DSC037**, including notebooks for quality assessment and scripts for visibility-based and image-based power spectra computation.

## Quality Assessment Notebooks

| Notebook | Purpose | Description |
|----------|---------|-------------|
| `plot-vis.ipynb` | step 3 of the WT: visibility inspection | Loads visibilities from LOFAR Measurement Sets (via `casacore`) or UVFITS/pyuvdata-compatible files and plots amplitude and phase versus time and frequency for individual baselines. Interactive controls let you choose polarisations, antennas, and plotting axes to verify temporal and spectral smoothness. |
| `plot-calibration-solutions.ipynb` | step 3  of the WT: calibration-solution QA | Opens complex gain solutions saved as FITS tables and plots the per-antenna, per-polarisation amplitude and phase versus frequency. Provides optional phase unwrapping and smoothness metrics to highlight jumps that may indicate cable reflections or calibration pathologies. |
| `plot-rudimentary.ipynb` | step 3  of the WT: quick delay-spevctra inspection | Computes and plots rudimentary delay power spectra (FFT → absolute value → square). |

## Visibility-Based Delay Power Spectra

A script uses **hera_pspec** (the HERA power-spectra pipeline) to compute visibility-based delay power spectra (step 4 of the WT).  
It creates:

- a time-averaged delay PS per antenna (averaging all baselines including a specific antenna), and  
- a time-averaged delay power spectrum across all baselines.  

## Image-Based Power Spectra

A second script uses **pspipe** (the LOFAR power-spectra pipeline) to compute image-based cylindrically averaged power spectra  (step 4 of the WT).  

It creates:
- a Cylndrically averaged power-spectra  figure.
- a Angular power-spectra figure.

## Configuration

Both scripts take a YAML file as input (with the same format).  
The file specifies dataset location and input parameters such as instrument, data column, polarisation, and frequency/time ranges.  

### Example YAML
```yaml
datafolder : '../data/'
datafile   : 'hyp_1184702048_ionosub_ssins_30l_src8k_300it_8s_80kHz_i1000.uvfits'
instrument : 'MWA'
beamfile   : None
data_col   : None

# The polarisations to include when reading data into the object
pol : 'XX'

# The time range in Julian Date to include when reading data into the object, must be length 2. 
# If None, all times in datafile are used.
time_range : None

# The range of frequencies to include when reading data into the object (min and max only), in MHz.
freq_range : [100, 200]

# The antenna numbers to include when reading data into the object. 
# If None, all antennas are considered.
antenna_nums : None
```

## Bandpass Calibration

This script performs a direction-independent correction of bandpass effects, including the removal of spectral structures produced by cable reflections.

### Bandpass Calibration Script: `run_bpcal.py`

The script `run_bpcal.py` automates a two-step DI calibration:

1. Bandpass gaincal using DP3 (steps `averager` and `gaincal`)
2. Applycal using DP3 (writing corrected visibilities to a new data column)

It also creates a clean working directory (`*_run/`) for each execution, generates all DP3 parset files based on the YAML configuration, logs all DP3 output to both screen and log files, and removes temporary intermediate Measurement Sets.

#### Example usage

```
python run_bpcal.py config_bpcal.yaml
```

This produces:

```
config_bpcal_1/
    DI_bandpass_cal.parset
    DI_applycal.parset
    bandpass-gaincal.h5
    bandpass_cal.log
    applycal.log
```

### Configuration file format

`run_bpcal.py` takes as input a YAML configuration file describing the Measurement Set location and calibration parameters.

#### Example `config_bpcal.yaml`

```
datafolder: "../data/"
datafile: "L253456_SAP000_002_time1.flagged.5ch8s.dical.MS"

input_datacol: "DATA"
output_datacol: "BP_CORRECTED"

skymodelfile: "../skymodels/L253456_DI_10mJy.txt"

min_bl_len: 50.
max_bl_len: 5000.

avg_time_step: 4
time_sol_int: 450
```

The script uses this to generate `DI_bandpass_cal.parset` and `DI_applycal.parset` and executes DP3 accordingly.

---

## Bandpass Calibration Quality-Assessment Notebooks

Two notebooks are included to validate the calibration solutions and to compare the impact of the calibration on the final power spectra.

### `check_bp_cal.ipynb` — Inspect DI bandpass calibration solutions

This notebook loads H5Parm gain solutions and visualises:

- per-antenna bandpass amplitude  
- delay-domain FFT of the bandpass  
- antenna-averaged responses  
- optional sinusoidal fitting of cable-reflection signatures  

It is designed to assess the smoothness and quality of the DI bandpass calibration and to check for residual cable-reflection terms.

### `compare_ps.ipynb` — Compare power spectra before/after calibration

This notebook compares two power-spectra products generated by `image_ps/run_pspipe.py`. It loads two output working directories (e.g. before and after DI bandpass calibration) and plots:

- 2D cylindrical power spectra for each dataset  
- the 2D power spectrum of the visibility-difference  
- 1D k∥ spectra for each dataset  

This allows quick inspection of whether the bandpass calibration successfully suppresses spectral.

## Dataset prerequisites
Each notebook assumes access to calibrated visibility data from either an MWA or
LOFAR observation. The example datasets below are the ones used during the
challenge and provide a good starting point when reproducing the analysis.

### MWA example dataset (~2 GB, UVFITS)

```bash
wget https://projects.pawsey.org.au/high0.uvfits/hyp_1184702048_ionosub_ssins_30l_src8k_300it_8s_80kHz_i1000.uvfits
```

### LOFAR example dataset (~22 GB, Measurement Set)

```bash
wget -O L253456_SAP000_002_time1.flagged.5ch8s.dical.MS.zip 'https://share.obspm.fr/s/Cek959sM3KRb4BQ/download'
unzip L253456_SAP000_002_time1.flagged.5ch8s.dical.MS.zip
```

## Software requirements

All notebooks rely on a standard scientific Python stack plus packages commonly used
for radio-interferometric data analysis:

- `numpy`
- `matplotlib`
- `astropy`
- `pyuvdata`
- `hera_pspec`
- `python-casacore` (required only when reading Measurement Sets)

For the image-based power-spectra, a container is available at https://share.obspm.fr/s/nj3eB2bA9z9oBLd

For the bandpass calibratuon, DP3 is required which is available in the same container.

## Additional resources

The following challenge documentation provides broader context and detailed walkthroughs:

- Chronological walkthrough: https://confluence.skatelescope.org/x/osw6F
- Implementation notes and software design: https://confluence.skatelescope.org/x/n8LMF
- Source repository on GitHub: https://github.com/uksrc-developers/dsc-037-eor
- Jira tickets: [TEAL-1128](https://jira.skatelescope.org/browse/TEAL-1128),
  [TEAL-1129](https://jira.skatelescope.org/browse/TEAL-1129)

Questions about the notebooks can be directed to the Teal team members listed within
the notebook headers.
