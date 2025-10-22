# DSC-037: Cable Reflection Systematics for EoR Science

This repository contains the implementation code for **DSC037**, including notebooks for quality assessment and scripts for visibility-based and image-based power spectra computation.

## Quality Assessment Notebooks

| Notebook | Purpose | Description |
|----------|---------|-------------|
| `plot-vis.ipynb` | step 3 of the WT: visibility inspection | Loads visibilities from LOFAR Measurement Sets (via `casacore`) or UVFITS/pyuvdata-compatible files and plots amplitude and phase versus time and frequency for individual baselines. Interactive controls let you choose polarisations, antennas, and plotting axes to verify temporal and spectral smoothness. |
| `plot-calibration-solutions.ipynb` | step 3  of the WT: calibration-solution QA | Opens complex gain solutions saved as FITS tables and plots the per-antenna, per-polarisation amplitude and phase versus frequency. Provides optional phase unwrapping and smoothness metrics to highlight jumps that may indicate cable reflections or calibration pathologies. |
| `plot-rudimentary.ipynb` | step 3  of the WT: quick delay-spevctra inspection | Computes and plots rudimentary delay power spectra (FFT → absolute value → square). |

---

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


## Additional resources

The following challenge documentation provides broader context and detailed walkthroughs:

- Chronological walkthrough: https://confluence.skatelescope.org/x/osw6F
- Implementation notes and software design: https://confluence.skatelescope.org/x/n8LMF
- Source repository on GitHub: https://github.com/uksrc-developers/dsc-037-eor
- Jira tickets: [TEAL-1128](https://jira.skatelescope.org/browse/TEAL-1128),
  [TEAL-1129](https://jira.skatelescope.org/browse/TEAL-1129)

Questions about the notebooks can be directed to the Teal team members listed within
the notebook headers.
