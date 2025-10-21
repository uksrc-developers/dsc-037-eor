import yaml
import os
import pyuvdata
import numpy as np
import hera_pspec as hp


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
    # turn string pol to int
    if isinstance(cfg['pol'], str):
        cfg['pol'] = pyuvdata.utils.str2polnum(cfg['pol'])
    # for measurement set, specify data column
    if cfg['data_col'] is None:
        cfg['data_col'] = 'DATA'
        if verbose and os.path.splitext(cfg['datafile']) in ['.ms', '.MS']:
            print(f'No data column specified; using default column: {cfg["data_col"]}.')
    # check data file
    uvd_meta = pyuvdata.UVData()
    uvd_meta.read(os.path.join(cfg['datafolder'], cfg['datafile']), read_data=False)
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
        print(f'Selected frequency range: {cfg["freq_range"]} MHz,'
              f' corresponding to average redshift of {cfg["avg_z"]:.1f}.')
        print(f'Selected polarization: {cfg["pol"]} ({pyuvdata.utils.polnum2str(cfg["pol"])})') 

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
