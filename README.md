# DSC-037: Cable Reflection Systematics for EoR Science

This repository hosts three exploratory Jupyter notebooks produced for the
[DSC-037](https://confluence.skatelescope.org/x/0rs6F) data-science challenge.  The
material concentrates on steps 3 and 4 of the challenge workflow, where the goal is
to assess the spectral smoothness of calibrated visibilities and to derive delay power
spectra that highlight cable-reflection systematics in SKA-Low pathfinder data.

## Notebook overview

| Notebook | Focus | Key capabilities |
| --- | --- | --- |
| `plot-vis.ipynb` | Challenge step 3: visibility inspection | Loads visibilities from LOFAR Measurement Sets (via `casacore`) or UVFITS/pyuvdata-compatible files and plots amplitude and phase versus time and frequency for individual baselines. Interactive controls let you choose polarisations, antennas, and plotting axes to verify temporal and spectral smoothness. |
| `plot-calibration-solutions.ipynb` | Challenge step 3: calibration-solution QA | Opens complex gain solutions saved as FITS tables and plots the per-antenna, per-polarisation amplitude and phase versus frequency. Provides optional phase unwrapping and smoothness metrics to highlight jumps that may indicate cable reflections or calibration pathologies. |
 `plot-rudimentary.ipynb` | Challenge step 3: calibration-solution QA |  compute and plot rudimentary (FFT, absolute value, then square) delay power spectra. |
| `bl-avg_delayps_per_antenna.ipynb` | Challenge step 4: delay power spectra per antenna | Builds time- and redundancy-averaged delay power spectra for all baselines that include a selected antenna. Uses `pyuvdata` to read visibilities and `hera_pspec` to form delay transforms, with options to filter by polarisation, time range, and maximum baseline length. |
| `time-avg_delayps_across_blens.ipynb` | Challenge step 4: delay spectra across baseline lengths | Aggregates time-averaged delay power spectra across all baselines shorter than a configurable threshold, enabling cylindrical averaging by baseline-length bin. Mirrors the configuration controls of the per-antenna notebook but emphasises exploring different redundant groups. |

Each notebook begins with dataset metadata inspection, followed by configuration
cells that let you select time/frequency windows, polarisation products, target
antennas or baseline-length limits, and plotting preferences.  Subsequent sections
load the visibilities, prepare them for analysis, and render the diagnostic plots
inline.

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


## Additional resources

The following challenge documentation provides broader context and detailed walkthroughs:

- Chronological walkthrough: https://confluence.skatelescope.org/x/osw6F
- Implementation notes and software design: https://confluence.skatelescope.org/x/n8LMF
- Source repository on GitHub: https://github.com/uksrc-developers/dsc-037-eor
- Jira tickets: [TEAL-1128](https://jira.skatelescope.org/browse/TEAL-1128),
  [TEAL-1129](https://jira.skatelescope.org/browse/TEAL-1129)

Questions about the notebooks can be directed to the Teal team members listed within
the notebook headers.
