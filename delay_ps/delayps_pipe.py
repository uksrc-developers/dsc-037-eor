# **DSC-037**: Cable reflection systematics for EoR science
#
# **Author:** Adélie Gorce, Florent Mertens, and the Teal team
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
# **Tickets:** 
# - TEAL-1129 https://jira.skatelescope.org/browse/TEAL-1129
# - TEAL-1158 https://jira.skatelescope.org/browse/TEAL-1158

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
from hera_cal.frf import FRFilter
import pyuvdata
from casacore.tables import table
from astropy.coordinates import EarthLocation

# CASA Stokes enum mapping
STOKES_MAP = {
    1: "I", 2: "Q", 3: "U", 4: "V",
    5: "RR", 6: "RL", 7: "LR", 8: "LL",
    9: "XX", 10: "XY", 11: "YX", 12: "YY",
}


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
    # check if there are several polarisations selected
    cfg['pol'] = np.atleast_1d(cfg['pol'])
    # turn string pol to int
    for i, p in enumerate(cfg['pol']):
        if isinstance(p, str):
            cfg['pol'][i] = pyuvdata.utils.polstr2num(p)
    # for measurement set, specify data column
    cfg['data_format'] = os.path.splitext(cfg['datafile'])[-1][1:]
    if cfg['data_col'] is None:
        if os.path.splitext(cfg['datafile'])[-1] in ['.ms', '.MS']:
            cfg['data_col'] = 'DATA'
            if verbose:
                print(f'No data column specified; using default column: {cfg["data_col"]}.')
        else:
            cfg['data_col'] = None
    # check data file
    uvd_meta = load_data(cfg, read_data=False)
    cfg.update({"instrument": uvd_meta.telescope.name})
    # format antenna list
    if cfg['antenna_nums'] is None:
        cfg['antenna_nums'] = uvd_meta.get_ants()
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
    frequencies = np.array([uvd_meta.freq_array[cfg['freq_range'][i]] for i in [0, 1]])/1e6  # MHz
    cfg['avg_z'] = cosmo.f2z(np.mean(frequencies)*1e6)

    # Print out loaded configuration
    if verbose:
        print(f'\nLoaded {cfg["instrument"]} dataset with required configuration.')
        print('Data description:')
        print(f' Number of baselines: {uvd_meta.Nbls}')
        print(f' Number of times: {uvd_meta.Ntimes}')
        print(f' Number of frequencies: {uvd_meta.Nfreqs}')
        print(f' Number of polarizations: {uvd_meta.Npols} ({uvd_meta.polarization_array})')
        print(f' Number of antennas: {cfg["antenna_nums"].size}')
        print('Required configuration:')
        print(f' Selected frequency range: {frequencies} MHz,'
              f' corresponding to average redshift of {cfg["avg_z"]:.1f}.')
        print(f' Selected polarization: {cfg["pol"]} ({[pyuvdata.utils.polnum2str(p) for p in cfg["pol"]]})')
        print(f' Selected antennas: {cfg["antenna_nums"]}')
        print(f'Performing {cfg['analysis_type']} analysis..')

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


def telescope_from_ms(ms_path):
    """
    Extract telescope metadata from MS and return a pyuvdata.Telescope.
    
    Parameters
    ----------
        ms_path: str or Path
            Path to measurement set folder.

    Returns
    -------
        tel: pyuvdata.Telescope
            pyuvdata.Telescope object containing the information
            extracted from the MS.

    """
    # ANTENNA subtable
    ant_tab = table(ms_path + "/ANTENNA", readonly=True)
    ant_names = ant_tab.getcol("NAME")
    ant_positions = ant_tab.getcol("POSITION")  # ECEF meters
    ant_tab.close()

    # OBSERVATION subtable
    obs_tab = table(ms_path + "/OBSERVATION", readonly=True)
    telescope_name = obs_tab.getcol("TELESCOPE_NAME")[0]
    obs_tab.close()

    # Use first antenna as reference
    loc_array = ant_positions[0]
    telescope_location = EarthLocation.from_geocentric(*loc_array, unit="m")

    ant_numbers = np.arange(len(ant_names))

    tel = pyuvdata.Telescope.new(
        name=telescope_name,
        location=telescope_location,
        antenna_positions=ant_positions,
        antenna_names=ant_names,
        antenna_numbers=ant_numbers.tolist(),
        instrument=telescope_name,
    )
    return tel


def uvd_meta_from_ms(ms_path):
    """
    Build a UVData metadata-only object from MS metadata.
    
    Parameters
    ----------
        ms_path: str or Path
            Path to measurement set folder.

    Returns
    -------
        uvd: pyuvdata.UVData object
            pyuvdata.UVData object containing the information
            extracted from the MS.
    
    """
    # Telescope
    tel = telescope_from_ms(ms_path)

    # Frequencies (Hz)
    spw_tab = table(ms_path + "/SPECTRAL_WINDOW", readonly=True)
    freqs = spw_tab.getcol("CHAN_FREQ").flatten()
    spw_tab.close()
    freq_array = freqs

    # Polarizations
    pol_tab = table(ms_path + "/POLARIZATION", readonly=True)
    corr_types = pol_tab.getcol("CORR_TYPE")[0]
    pol_tab.close()
    polarization_array = [STOKES_MAP.get(c, f"corr{c}") for c in corr_types]

    # Times (MS stores seconds from MJD=0)
    main_tab = table(ms_path, readonly=True)
    times_sec = main_tab.getcol("TIME")  # seconds
    main_tab.close()
    times_jd = np.unique(times_sec) / 86400.0 + 2400000.5

    # Build UVData metadata-only object
    uvd = pyuvdata.UVData.new(
        freq_array=freq_array,
        polarization_array=polarization_array,
        times=times_jd,
        telescope=tel,
    )
    return uvd


def load_data(dic, read_data=True):
    """
    Load UVData object from file.

    Parameters
    ----------
        dic: dict
            Dictionary containing relevant information about data and analysis choices.
    Returns
    ----------
        uvd: UVData object
            UVData object containing (meta)data.
        read_data: boolean
            Whether to read data (True) or only metadata (False).
            Default is True.

    """
    uvd = pyuvdata.UVData()
    if dic['data_format'] in ['ms', 'MS']:
        if read_data:
            uvd.read_ms(
                os.path.join(dic['datafolder'], dic['datafile']),
                data_column=dic['data_col'],
                run_check=False,
            )
            # check UVW convention
            temp_obj = uvd.copy(metadata_only=True)
            temp_obj.set_uvws_from_antenna_positions()
            if np.allclose(uvd.uvw_array, -temp_obj.uvw_array, atol=5.):
                print(
                    "UVW orientation appears to be flipped, attempting to "
                    "fix by changing conjugation of baselines."
                )
                uvd.uvw_array *= -1
                uvd.data_array = np.conj(uvd.data_array)
        else:
            uvd = uvd_meta_from_ms(os.path.join(dic['datafolder'], dic['datafile']))
    else:
        uvd.read(
            os.path.join(dic['datafolder'], dic['datafile']),
            keep_all_metadata=False,
            read_data=read_data,
        )
    uvd.check()
    uvd.select(
        polarizations=dic['pol'],
        time_range=dic['time_range'],
        inplace=True
    )
    return uvd


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

    # load whole dataset
    # create UVData object and read in data
    uvd = load_data(dic, read_data=True)
    # ignore auto-correlation visibilities
    uvd.select(ant_str='cross')

    # Build delay power spectra, but only for baselines including a specific antenna.
    # loop over antennas to build delay power spectra
    data_time_avg = np.zeros((len(dic['pol']), len(dic['antenna_nums']), np.diff(dic['freq_range'])[0]//2-1))
    data_per_antenna = np.zeros((len(dic['pol']), len(dic['antenna_nums']), dic['Ntimes'], np.diff(dic['freq_range'])[0]//2-1))
    for ip, pol in enumerate(dic['pol']):
        for u, antenna_num in enumerate(tqdm(dic['antenna_nums'])):

            # selection when reading data not available for MS
            # select data for a single antenna
            uvd_loc = uvd.select(
                ant_str=f'{antenna_num}',
                inplace=False
            )
            # average over all the baselines including said antenna
            uvd_loc.compress_by_redundancy(tol=100000., use_grid_alg=True)
            # Create a new PSpecData object which will be used to compute the delay PS
            ds = hp.PSpecData(dsets=[uvd_loc, uvd_loc], wgts=[None, None], beam=None)
            # in the baseline-averaged dataset, there is only one baseline left (the first one)
            bl = uvd_loc.baseline_to_antnums(uvd_loc.baseline_array[0])
            # build time-averaged delay ps from pairing the baseline with itself
            uvp = ds.pspec(
                [bl], [bl],  # select the baselines to cross (here, with itself)
                dsets=(0, 1),  # select which datasets to use within ds
                pols=[(pol, pol)],  # select the polarisation channels to cross
                spw_ranges=[tuple(dic['freq_range'])],  # select a smaller bandwidth
                verbose=False
            )
            # save time array for per-antenna figure
            if u == 0:
                time_array = Time(uvp.time_avg_array, format='jd')
            # fold spectrum over the delay axis
            hp.grouping.fold_spectra(uvp)
            # save delay power spectrum per antenna, as a function of time and delay
            data_per_antenna[ip, u] = np.abs(uvp.data_array[0][:, -uvp.get_dlys(0).size:, 0])
            # take time average of the data
            uvp.average_spectra(time_avg=True, inplace=True)
            # save time-averaged delay power spectrum per antenna
            data_time_avg[ip, u] = np.abs(uvp.data_array[0][0, -uvp.get_dlys(0).size:, 0])
    # define time array for per-antenna figure
    t_ref = Time(uvp.time_avg_array.min(), format='jd')
    time_array = (time_array - t_ref).to('s').value

    # Gather the results in a figure presenting the time-averaged delay PS per antenna ($x$ axis) to identify which antennas are the most impacted by cable reflections

    vmin = np.percentile(np.abs(data_time_avg), 2)
    vmax = np.percentile(np.abs(data_time_avg), 98)

    for ip, pol in enumerate(dic['pol']):
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        im = ax.pcolormesh(
            dic['antenna_nums'],
            uvp.get_dlys(0)*1e6,
            np.abs(data_time_avg[ip]).T,
            cmap='Purples',
            norm=colors.Normalize(vmin=vmin, vmax=vmax),
        )
        fig.colorbar(im, ax=ax, label=r'Power [Jy$^2$]')
        ax.set_ylabel(r'Delay [$\mu$s]')
        ax.set_xlabel('Antenna number')
        fig.tight_layout()
        fig_name1 = f'delay_ps_per_antenna_{dic["instrument"]}_{pyuvdata.utils.polnum2str(pol)}.png'
        fig.savefig(fig_folder / fig_name1, dpi=300) 

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
                np.abs(data_per_antenna[ip, i]).T,
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
        fig_name2 = f'delay_ps_per_antenna_vs_time_{dic["instrument"]}_{pyuvdata.utils.polnum2str(pol)}.png'
        fig.savefig(fig_folder / fig_name2, dpi=300)


def time_average_delayps_across_blens(dic, fig_folder, bl_tol=1., verbose=False):
    """
    Compute a time-averaged delay power spectrum across all baselines using the `pyuvdata` and `hera_pspec` packages

    Parameters
    ----------
        dic: dict
            Dictionary containing relevant information about data and analysis choices.
        fig_folder: Path
            Path to folder where to save figures.
        bl_tol: float
            Baseline tolerance to use when grouping redundant baselines [m].

    """

    # load whole dataset
    # create UVData object and read in data
    uvd = load_data(dic, read_data=True)
    # ignore auto-correlation visibilities
    uvd.select(ant_str='cross')

    # Coherent time average of visibilities
    # TODO: temporary fix whilst waiting for pyuvdata issue to be resolved
    # will perform incoherent time averaging instead for MS
    if dic['data_format'] not in ['ms', 'MS']:
        uvd.downsample_in_time(n_times_to_avg=dic['Ntimes'])

    # Use hera_cal.redcal to get matching, redundant baseline-pair groups within the specified baseline tolerance, not including flagged ants.
    bls1, bls2, _, _, _, red_groups, red_lens, _ = hp.utils.calc_blpair_reds(
        uvd, uvd,
        exclude_auto_bls=False, exclude_cross_bls=True, # get auto-correlations only, ie pairs [(ant1, ant2), (ant1, ant2)]
        exclude_permutations=True,
        include_autocorrs=False, include_crosscorrs=True,
        bl_tol=bl_tol,
        bl_len_range=(0, dic['max_bl_len']),
        extra_info=True
    )
    if verbose:
        print(f'There are {len(bls1)} redundant groups of auto-baselines with length < {dic["max_bl_len"]} m.')

    # Create a new PSpecData object which will be used to compute the delay PS
    ds = hp.PSpecData(dsets=[uvd, uvd], wgts=[None, None], beam=None)
    # build delay ps from baseline pairs constructed above
    for ip, pol in enumerate(dic['pol']):
        uvp = ds.pspec(
            bls1, bls2,
            dsets=(0, 1),
            pols=[(pol, pol)],
            spw_ranges=[tuple(dic['freq_range'])],  # select a smaller bandwidth
            verbose=False
        )
        if dic['data_format'] in ['ms', 'MS']:
            uvp.average_spectra(time_avg=True, inplace=True)
        dat = np.copy(uvp.data_array[0][:, :, 0])

        # plot
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
        fig_name = f'bl-avg_delay_ps_across_blens_{dic["instrument"]}_{pyuvdata.utils.polnum2str(pol)}.png'
        fig.savefig(fig_folder / fig_name, dpi=300)


def autocorr_visibilities_per_antenna(dic, fig_folder):
    """
    Compute time-averaged delay auto-correlation visibilities
    using the `pyuvdata` and `hera_cal` packages

    Parameters
    ----------
        dic: dict
            Dictionary containing relevant information about data and analysis choices.
        fig_folder: Path
            Path to folder where to save figures.

    """

    # load whole dataset
    # create UVData object and read in data
    uvd = load_data(dic, read_data=True)

    # select auto-correlation data
    uvd.select(ant_str='auto')

    # only select required antennas in dataset
    if len(dic['antenna_nums']) < len(uvd.get_ants()):
        ant_str = ''
        for i, an in enumerate(dic['antenna_nums']):
            ant_str += str(an)
            if i < len(dic['antenna_nums']) -1:
                ant_str += ','
        uvd.select(ant_str=ant_str)

    # define a filter object from hera_cal
    F = FRFilter(uvd)
    # Coherent time average of visibilities
    F.timeavg_data(F.data, F.times, F.lsts, flags=F.flags, t_avg=1e10, overwrite=True)
    # Fourier transform the data to get delay visibilities
    F.fft_data(
        data=F.avg_data, flags=F.avg_flags, assign='avg_fft', ax='freq', window='bh', 
        # edgecut_low=ecf_low, edgecut_hi=ecf_high, 
        overwrite=True
    )

    # Gather the results in a figure
    ncol = 10
    nrow = np.ceil(len(dic['antenna_nums'])/ncol).astype(int)
    fig, axes = plt.subplots(nrow, ncol, figsize=(20, 2*nrow), sharex=True, sharey=True)
    for i, ant in enumerate(uvd.get_ants()):
        ax = axes.flatten()[i]
        for ip, pol in enumerate(uvd.get_pols()):
            abl = (ant, ant, pol)
            ax.semilogy(
                F.delays, np.abs(np.squeeze(F.avg_fft[abl])),
                label=pol, color=f'C{ip}',
                lw=1., alpha=.6
        )
        ax.set_title(f'({ant}, {ant})')
        ax.grid()
        if i % ncol == 0:
            ax.set_xlabel(r'Delay [ns]')
        if i >= (nrow-1)*ncol:
            ax.set_ylabel(r'$\vert \widetilde{V}_{ii} \vert$  [Jy Hz]')
    for i in np.arange(ncol * nrow % len(dic['antenna_nums'])):
        axes.flatten()[-i-1].set_visible(False)
    axes.flatten()[0].legend()
    fig.tight_layout()
    fig_name = f'autocorr_visibilities_{dic["instrument"]}.png'
    fig.savefig(fig_folder / fig_name, dpi=300)


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

    if dic['analysis_type'] == 'cross':
        # Compute a delay power spectrum for a single antenna (and all associated baselines) 
        # Produce corresponding figures
        bl_avg_delayps_per_antenna(dic, root)

        # Compute a time-averaged delay power spectrum across redundant baselines
        time_average_delayps_across_blens(dic, root, bl_tol=1., verbose=True)

    elif dic['analysis_type'] in ['auto', 'autos']:
        # Compute time-averaged delay visibilities for auto-correlations
        autocorr_visibilities_per_antenna(dic, root)

    else:
        raise ValueError('Analysis type can only be auto or cross.')


if __name__ == "__main__":
    main()
