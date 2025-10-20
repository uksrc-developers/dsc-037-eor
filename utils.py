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
    if isinstance(cfg['pol'], str):
        cfg['pol'] = pyuvdata.utils.str2polnum(cfg['pol'])

    # Print out loaded configuration
    if verbose:
        print(f'Loaded {cfg["dataset"]} dataset with required configuration.')
        cosmo = hp.conversions.Cosmo_Conversions()
        avg_z = cosmo.f2z(np.mean(cfg['freq_range'])*1e6)
        print(f'Selected frequency range: {cfg["freq_range"]} MHz,'
              f' corresponding to average redshift of {avg_z:.2f}.')

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
