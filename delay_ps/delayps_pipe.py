# **DSC-037**: Cable reflection systematics for EoR science
# 
# **Author:** Adélie Gorce and Teal team 
# 
# **Documentation on confluence:** 
# - DSC description page: https://confluence.skatelescope.org/x/0rs6F
# - Chronological walkthrough: https://confluence.skatelescope.org/x/osw6F
# - Implementation: https://confluence.skatelescope.org/x/n8LMF
# - GitHub repo: https://github.com/uksrc-developers/dsc-037-eor
# 
# **Summary:**  This notebook is a first implementation of the step 4 of DSC-037 (see chronological walkthrough above) to calculate delay power spectra for individual baselines and then cylindrically averaged power spectra for a user-specified set of frequencies, times, and polarisations.
# In this notebook, we
# - Load the visibilities of a given dataset
# - Compute a delay power spectra for checking cable reflections in EoR data, using the `pyuvdata` and `hera_pspec` packages
#
# **Ticket:** TEAL-1129 https://jira.skatelescope.org/browse/TEAL-1129

# import required packages
import numpy as np
from astropy.time import Time
from tqdm import tqdm
import os
import click
import matplotlib.pyplot as plt
from matplotlib import colors
from pathlib import Path
import yaml

import hera_pspec as hp
import pyuvdata


def load_config(config_file, verbose=False):
    """
    Load analysis choices from yaml file.

    Parameters
    ----------
        config_file: str
            Name of configuration file. Must be a yaml file.
        verbose: bool
            If True, print out loaded configuration.
    Returns
    -------
        Dictionary containing relevant information in appropriate format.
    """

    # Open and read config file
    if not os.path.exists(config_file):
        raise ValueError("The configuration file does not exist.")
    with open(config_file, 'r') as cfile:
        try:
            cfg = yaml.load(cfile, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            raise(exc)

    # Replace entries
    replace(cfg)
    # beam
    if cfg['beamfile'] is not None:
        raise NotImplementedError("Custom beamfile reading not implemented yet.")
    # turn string pol to int
    if isinstance(cfg['pol'], str):
        cfg['pol'] = pyuvdata.utils.str2polnum(cfg['pol'])
    # for measurement set, specify data column
    if cfg['data_col'] is None:
        if os.path.splitext(cfg['datafile']) in ['.ms', '.MS']:
            cfg['data_col'] = 'DATA'
            if verbose:
                print(f'No data column specified; using default column: {cfg["data_col"]}.')
        else:
            cfg['data_col'] = None
    # check data file
    uvd_meta = pyuvdata.UVData()
    uvd_meta.read(os.path.join(cfg['datafolder'], cfg['datafile']), read_data=False)
    # find indices to select frequency range (use in hp.pspec)
    cfg['spw_ranges'] = [tuple([list(uvd_meta.freq_array).index(cfg['freq_range'][i]*1e6) for i in (0, 1)])]
    # format antenna list
    if cfg['antenna_nums'] is None:
        cfg['antenna_nums'] = np.unique(uvd_meta.ant_1_array)
        if verbose:
            print(f'No antennas specified; using all antennas in data ({cfg["antenna_nums"].size}).')
    cfg['antenna_nums'] = np.atleast_1d(cfg['antenna_nums'])
    # format time range
    if cfg['time_range'] is not None:
        uvd_meta.select(time_range=cfg['time_range'])
        cfg['Ntimes'] = uvd_meta.Ntimes
    else:
        cfg['Ntimes'] = uvd_meta.Ntimes
        if verbose:
            print(f'No times specified; using all timestamps in data ({cfg["Ntimes"]}).')
    # format frequency range
    cosmo = hp.conversions.Cosmo_Conversions()
    cfg['avg_z'] = cosmo.f2z(np.mean(cfg['freq_range'])*1e6)

    # Print out loaded configuration
    if verbose:
        print(f'Loaded {cfg["instrument"]} dataset with required configuration.')
        print('Data description:')
        print(f' Number of baselines: {uvd_meta.Nbls}')
        print(f' Number of times: {uvd_meta.Ntimes}')
        print(f' Number of frequencies: {uvd_meta.Nfreqs}')
        print(f' Number of polarizations: {uvd_meta.Npols} ({uvd_meta.polarization_array})')
        print('Analysis choices:')
        print(f' Selected frequency range: {cfg["freq_range"]} MHz,'
              f' corresponding to average redshift of {cfg["avg_z"]:.1f}.')
        print(f' Selected polarization: {cfg["pol"]} ({pyuvdata.utils.polnum2str(cfg["pol"])})') 

    return cfg


def replace(d):
    if isinstance(d, dict):
        for k in d.keys():
            # 'None' and '' turn into None
            if d[k] == 'None':
                d[k] = None
            # list of lists turn into lists of tuples
            if isinstance(d[k], list) and np.all([isinstance(i, (list, tuple)) for i in d[k]]):
                d[k] = [tuple(i) for i in d[k]]
            elif isinstance(d[k], dict):
                replace(d[k])


def bl_avg_delayps_per_antenna(dic, fig_folder):
    """
    Compute a delay power spectrum for a single antenna (and all associated baselines) using the `pyuvdata` and `hera_pspec` packages

    Parameters
    ----------
        dic: dict
            Dictionary containing relevant information about data and analysis choices.
        fig_folder: Path
            Path to folder where to save figures.

    """
    if dic['beamfile'] is None:
        uvb = None

    # Build delay power spectra, but only for baselines including a specific antenna.
    # loop over antennas to build delay power spectra
    data_time_avg = np.zeros((len(dic['antenna_nums']), np.diff(dic['spw_ranges'][0])[0]//2-1))
    data_per_antenna = np.zeros((len(dic['antenna_nums']), dic['Ntimes'], np.diff(dic['spw_ranges'][0])[0]//2-1))
    for u, antenna_num in enumerate(tqdm(dic['antenna_nums'])):
        # create UVData object and read in data
        uvd = pyuvdata.UVData()
        uvd.read(
            os.path.join(dic['datafolder'], dic['datafile']),
            polarizations=[dic['pol']],
            time_range=dic['time_range'],
            # freq_chans=np.arange(spw_ranges[0][0], spw_ranges[0][1]),
            keep_all_metadata=False,
            read_data=True,
            data_column=dic['data_col'],
        )
        # select data for a single antenna
        uvd.select(ant_str=f'{antenna_num}')
        # average over all the baselines including said antenna
        uvd.compress_by_redundancy(tol=100000., use_grid_alg=True)
        # Create a new PSpecData object which will be used to compute the delay PS
        ds = hp.PSpecData(dsets=[uvd, uvd], wgts=[None, None], beam=uvb)
        # in the baseline-averaged dataset, there is only one baseline left (the first one)
        bl = uvd.baseline_to_antnums(uvd.baseline_array[0])
        # build time-averaged delay ps from pairing the baseline with itself
        uvp = ds.pspec(
            [bl], [bl],  # select the baselines to cross (here, with itself)
            dsets=(0, 1),  # select which datasets to use within ds
            pols=[(dic['pol'], dic['pol'])],  # select the polarisation channels to cross
            spw_ranges=dic['spw_ranges'],  # select a smaller bandwidth
            verbose=False
        )
        # save time array for per-antenna figure
        if u == 0:
            time_array = Time(uvp.time_avg_array, format='jd')
        # fold spectrum over the delay axis
        hp.grouping.fold_spectra(uvp)
        # save delay power spectrum per antenna, as a function of time and delay
        data_per_antenna[u] = np.abs(uvp.data_array[0][:, -uvp.get_dlys(0).size:, 0])
        # take time average of the data
        uvp.average_spectra(time_avg=True, inplace=True)
        # save time-averaged delay power spectrum per antenna
        data_time_avg[u] = np.abs(uvp.data_array[0][0, -uvp.get_dlys(0).size:, 0])
    # define time array for per-antenna figure
    t_ref = Time(uvp.time_avg_array.min(), format='jd')
    time_array = (time_array - t_ref).to('s').value

    # Gather the results in a figure presenting the time-averaged delay PS per antenna ($x$ axis) to identify which antennas are the most impacted by cable reflections

    vmin = np.percentile(np.abs(data_time_avg), 2)
    vmax = np.percentile(np.abs(data_time_avg), 98)

    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    im = ax.pcolormesh(
        dic['antenna_nums'],
        uvp.get_dlys(0)*1e6,
        np.abs(data_time_avg).T,
        cmap='Purples',
        norm=colors.Normalize(vmin=vmin, vmax=vmax),
    )
    fig.colorbar(im, ax=ax, label=r'Power [Jy$^2$]')
    ax.set_ylabel(r'Delay [$\mu$s]')
    ax.set_xlabel('Antenna number')
    fig.tight_layout()
    fig.savefig(fig_folder / f'delay_ps_per_antenna_{dic["instrument"]}.png', dpi=300) 

    # Gather the results in a figure showing the delay power spectrum as a function of time for each antenna, in order to identify which antenna is most impacted by cable reflections
    ncol = 10
    nrow = np.ceil(len(dic['antenna_nums'])/ncol).astype(int)

    vmin = np.percentile(np.abs(data_per_antenna), 2)
    vmax = np.percentile(np.abs(data_per_antenna), 98)

    fig, axes = plt.subplots(nrow, ncol, figsize=(20, 2*nrow), sharex=True, sharey=True)
    for i, antenna_num in enumerate(dic['antenna_nums']):
        ax = axes.flatten()[i]
        im = ax.pcolormesh(
            time_array,
            uvp.get_dlys(0)*1e6,
            np.abs(data_per_antenna[i]).T,
            cmap='Purples',
            norm=colors.Normalize(vmin=vmin, vmax=vmax),
        )
        ax.set_title(f'A{antenna_num}')
        if i % ncol == 0:
            ax.set_ylabel(r'Delay [$\mu$s]')
        if i >= (nrow-1)*ncol:
            ax.set_xlabel('Time [s]')
    for i in np.arange(ncol * nrow % len(dic['antenna_nums'])):
        axes.flatten()[-i-1].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_folder / f'delay_ps_per_antenna_vs_time_{dic["instrument"]}.png', dpi=300)


def time_average_delayps_across_blens(dic, fig_folder, max_bl_len=30., bl_tol=1., verbose=False):
    """
    Compute a time-averaged delay power spectrum across all baselines using the `pyuvdata` and `hera_pspec` packages

    Parameters
    ----------
        dic: dict
            Dictionary containing relevant information about data and analysis choices.
        fig_folder: Path
            Path to folder where to save figures.
        max_bl_len: float
            Maximum baseline length to consider [m].
        bl_tol: float
            Baseline tolerance to use when grouping redundant baselines [m].

    """
    if dic['beamfile'] is None:
        uvb = None

    uvd = pyuvdata.UVData()
    uvd.read(
        os.path.join(dic['datafolder'], dic['datafile']),
        polarizations=[dic['pol']],
        time_range=dic['time_range'],
        keep_all_metadata=False,
        read_data=True,
        data_column=dic['data_col'],
    )

    # Coherent time average of visibilities
    uvd.downsample_in_time(n_times_to_avg=dic['Ntimes'])

    # Use hera_cal.redcal to get matching, redundant baseline-pair groups within the specified baseline tolerance, not including flagged ants.
    bls1, bls2, _, _, _, red_groups, red_lens, _ = hp.utils.calc_blpair_reds(
        uvd, uvd,
        exclude_auto_bls=False, exclude_cross_bls=True, # get auto-correlations only, ie pairs [(ant1, ant2), (ant1, ant2)]
        exclude_permutations=True,
        include_autocorrs=False, include_crosscorrs=True,
        bl_tol=bl_tol,
        bl_len_range=(0, max_bl_len),
        extra_info=True
    )
    if verbose:
        print(f'There are {len(bls1)} redundant groups of auto-baselines with length < {max_bl_len} m.')

    # Create a new PSpecData object which will be used to compute the delay PS
    ds = hp.PSpecData(dsets=[uvd, uvd], wgts=[None, None], beam=uvb)
    # build delay ps from baseline pairs constructed above
    uvp = ds.pspec(
        bls1, bls2,
        dsets=(0, 1),
        pols=[(dic['pol'], dic['pol'])],
        spw_ranges=dic['spw_ranges'],  # select a smaller bandwidth
        verbose=False
    )

    # plot
    dat = np.copy(uvp.data_array[0][:, :, 0])
    vmin = np.percentile(abs(dat), 2)
    vmax = np.percentile(abs(dat), 98)

    fig, ax = plt.subplots(1, 1)
    im = ax.pcolormesh(
        uvp.dly_array*1e6,
        np.take(red_lens, red_groups),
        np.abs(dat),
        cmap='Purples',
        norm=colors.LogNorm(vmin=vmin, vmax=vmax),
    )
    fig.colorbar(im, ax=ax, label=f'Delay PS')
    ax.set_xlabel(r'Delay [$\mu$s]')
    ax.set_ylabel('Baseline length [m]')
    fig.tight_layout()
    fig.savefig(fig_folder / f'bl-avg_delay_ps_across_blens_{dic["instrument"]}.png', dpi=300)


@click.command()
@click.argument("config_file", type=click.Path(exists=True))
def main(config_file):

    # derive root from config file location
    config_path = Path(config_file).resolve()
    root = config_path.parent

    # ## Load configuration
    # The antenna(s), time range, frequency range, polarization to analyse are specified in a configuration file. 
    # The configuration file also specifies the datafile to consider and its location.
    # The information is then loaded with the `load_config` method.
    dic = load_config(config_path, verbose=True)

    # Compute a delay power spectrum for a single antenna (and all associated baselines) 
    # Produce corresponding figures
    bl_avg_delayps_per_antenna(dic, root)

    # Compute a time-averaged delay power spectrum across redundant baselines
    time_average_delayps_across_blens(dic, root, max_bl_len=30., bl_tol=1., verbose=True)


if __name__ == "__main__":
    main()
