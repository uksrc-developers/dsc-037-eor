#!/usr/bin/env python3
"""
Calibration Solutions Plotting Tool

This script plots calibration solutions (amplitude and phase) vs frequency for each tile.

Authors: Shao EoR Group and Teal Team
Dependencies: numpy, matplotlib, astropy
Last updated: 2025-01-XX
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import yaml

# ----------------------
# Utilities / helpers
# ----------------------
def unwrap_phase_per_tile_pol(phases):
    """
    Unwrap each tile and pol of the three-dimensional phase array (tile, freq, pol) along the frequency axis.
    Safely handle NaN.

    Parameters
    ----------
    phases : np.ndarray
        A phase array with shapes of (ntile, nfreq, npol) in radians.

    Returns
    -------
    unwrapped : np.ndarray
        Same dimensional array, containing unwrapped phases and retaining NaN.
    """
    if phases.ndim != 3:
        raise ValueError(f"Input must be 3D (tile, freq, pol), got shape {phases.shape}")

    ntile, nfreq, npol = phases.shape
    unwrapped = np.full_like(phases, np.nan, dtype=float)

    for i_tile in range(ntile):
        for i_pol in range(npol):
            ph = phases[i_tile, :, i_pol]
            mask = np.isfinite(ph)
            if np.count_nonzero(mask) > 1:
                unwrapped_tile = np.full_like(ph, np.nan)
                unwrapped_tile[mask] = np.unwrap(ph[mask])
                unwrapped[i_tile, :, i_pol] = unwrapped_tile
            else:
                # Keep the original value when all NaN values are present or when there is only one value
                unwrapped[i_tile, :, i_pol] = ph

    return unwrapped

# ----------------------
# Core processing
# ----------------------
def load_calibration_fits(filename):
    """
    Load calibration solutions from FITS file.
    
    Parameters
    ----------
    filename : str
        Path to FITS file containing calibration solutions
        
    Returns
    -------
    dict
        Dictionary containing:
        - data: Full solution data array
        - name_tiles: Tile names
        - freqs: Frequency array (if available)
        - num_tiles: Number of tiles
        - f: FITS file handle (needs to be closed by caller)
    """
    f = fits.open(filename)
    
    if "SOLUTIONS" not in f:
        raise ValueError(f"FITS file '{filename}' does not contain 'SOLUTIONS' extension")
    
    data = f["SOLUTIONS"].data
    num_tiles = data.shape[1]
    
    name_tiles = None
    if "TILES" in f:
        name_tiles = f['TILES'].data
    
    # Extract frequency axis
    if "FREQS" in f:
        freqs = f["FREQS"].data
    else:
        # Fallback: use channel index
        n_chan = data.shape[2]  # since there is re/im
        freqs = np.linspace(0, n_chan-1, n_chan)
    
    return {
        'data': data,
        'name_tiles': name_tiles,
        'freqs': freqs,
        'num_tiles': num_tiles,
        'f': f
    }

def extract_timeblock(data, i_timeblock):
    """
    Extract complex data from a specific time block.
    
    Parameters
    ----------
    data : np.ndarray
        Solution data array with shape (ntime, ntile, nfreq, nval)
    i_timeblock : int
        Time block index
        
    Returns
    -------
    np.ndarray
        Complex data array with shape (ntile, nfreq, npol)
    """
    return data[i_timeblock, :, :, ::2] + data[i_timeblock, :, :, 1::2] * 1j

# ----------------------
# Plot functions
# ----------------------
def plot_amplitude_solutions(data, freqs, name_tiles, num_tiles, num_tiles_per_row, 
                             figsize, colors, alpha_main, alpha_cross, output_file, dpi, show=False):
    """
    Plot amplitude solutions vs frequency for all tiles.
    
    Parameters
    ----------
    data : np.ndarray
        Complex data array with shape (ntile, nfreq, npol)
    freqs : np.ndarray
        Frequency array
    name_tiles : array-like or None
        Tile names array
    num_tiles : int
        Number of tiles
    num_tiles_per_row : int
        Number of tiles per row in the subplot grid
    figsize : tuple
        Figure size (width, height)
    colors : dict
        Color dictionary for polarizations {'xx': 'blue', 'yy': 'green', ...}
    alpha_main : float
        Alpha value for main polarizations (XX, YY)
    alpha_cross : float
        Alpha value for cross polarizations (XY, YX)
    output_file : str
        Output file path for saving the plot
    dpi : int
        DPI for saved figure
    show : bool
        If True, display plot interactively; if False, save to file
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    amps = np.abs(data)
    fig_amp, ax_amp = plt.subplots(num_tiles_per_row, 16, sharex=True, sharey=True, figsize=figsize)
    ax_amp = ax_amp.flatten()
    
    for i in range(num_tiles):
        xx = amps[i, :, 0].flatten()
        yy = amps[i, :, 3].flatten()
        xy = amps[i, :, 1].flatten()
        yx = amps[i, :, 2].flatten()
        ax_amp[i].plot(freqs, xx, color=colors.get('xx', 'blue'), alpha=alpha_main, label='XX' if i == 0 else '')
        ax_amp[i].plot(freqs, yy, color=colors.get('yy', 'green'), alpha=alpha_main, label='YY' if i == 0 else '')
        ax_amp[i].plot(freqs, xy, color=colors.get('xy', 'blue'), alpha=alpha_cross, label='XY' if i == 0 else '')
        ax_amp[i].plot(freqs, yx, color=colors.get('yx', 'green'), alpha=alpha_cross, label='YX' if i == 0 else '')
        
        if name_tiles is not None and i < len(name_tiles):
            tile_label = name_tiles[i][2] if len(name_tiles[i]) > 2 else str(i)
        else:
            tile_label = str(i)
        ax_amp[i].set_xlabel(f'{i}:{tile_label}')
        if i == 0:
            ax_amp[i].legend(loc='best', fontsize=8)
    
    fig_amp.suptitle("Amplitude Solutions vs Frequency", fontsize=14)
    plt.tight_layout()
    
    if not show:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        plt.close(fig_amp)
    
    return fig_amp

def plot_phase_solutions(data, freqs, name_tiles, num_tiles, num_tiles_per_row,
                        figsize, colors, alpha_main, alpha_cross, output_file, dpi, show=False, unwrap=True):
    """
    Plot phase solutions vs frequency for all tiles.
    
    Parameters
    ----------
    data : np.ndarray
        Complex data array with shape (ntile, nfreq, npol)
    freqs : np.ndarray
        Frequency array
    name_tiles : array-like or None
        Tile names array
    num_tiles : int
        Number of tiles
    num_tiles_per_row : int
        Number of tiles per row in the subplot grid
    figsize : tuple
        Figure size (width, height)
    colors : dict
        Color dictionary for polarizations {'xx': 'blue', 'yy': 'green', ...}
    alpha_main : float
        Alpha value for main polarizations (XX, YY)
    alpha_cross : float
        Alpha value for cross polarizations (XY, YX)
    output_file : str
        Output file path for saving the plot
    dpi : int
        DPI for saved figure
    show : bool
        If True, display plot interactively; if False, save to file
    unwrap : bool
        If True, unwrap phases (but still plot original wrapped phases)
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    phases = np.angle(data)
    if unwrap:
        # Unwrap phases for analysis, but we'll plot the original wrapped phases
        unwrap_phases = unwrap_phase_per_tile_pol(phases)
    
    fig_phase, ax_phase = plt.subplots(num_tiles_per_row, 16, sharex=True, sharey=True, figsize=figsize)
    ax_phase = ax_phase.flatten()
    
    for i in range(num_tiles):
        # Note: plotting original 'phases', not 'unwrap_phases'
        xx = phases[i, :, 0].flatten()
        yy = phases[i, :, 3].flatten()
        xy = phases[i, :, 1].flatten()
        yx = phases[i, :, 2].flatten()
        ax_phase[i].plot(freqs, xx, color=colors.get('xx', 'blue'), alpha=alpha_main, label='XX' if i == 0 else '')
        ax_phase[i].plot(freqs, yy, color=colors.get('yy', 'green'), alpha=alpha_main, label='YY' if i == 0 else '')
        ax_phase[i].plot(freqs, xy, color=colors.get('xy', 'blue'), alpha=alpha_cross, label='XY' if i == 0 else '')
        ax_phase[i].plot(freqs, yx, color=colors.get('yx', 'green'), alpha=alpha_cross, label='YX' if i == 0 else '')
        
        if name_tiles is not None and i < len(name_tiles):
            tile_label = name_tiles[i][2] if len(name_tiles[i]) > 2 else str(i)
        else:
            tile_label = str(i)
        ax_phase[i].set_xlabel(f'{i}:{tile_label}')
        if i == 0:
            ax_phase[i].legend(loc='best', fontsize=8)
    
    fig_phase.suptitle("Phase Solutions vs Frequency", fontsize=14)
    plt.tight_layout()
    
    if not show:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        plt.close(fig_phase)
    
    return fig_phase

# ----------------------
# Main processing function
# ----------------------
def run_calibration_plotting(config_file='config.yaml', input_file=None, i_timeblock=None,
                             figsize=None, output_amp=None, output_phase=None, dpi=None,
                             colors=None, alpha_main=None, alpha_cross=None, show=None):
    """
    Process calibration solutions and generate plots based on configuration parameters.
    
    Common parameters are read from config.yaml by default. All parameters can be overridden
    by passing them as function arguments.
    
    Parameters
    ----------
    config_file : str, optional
        Path to configuration YAML file (default: 'config.yaml')
    input_file : str, optional
        Path to input FITS file. If None, read from config_file.
    i_timeblock : int, optional
        Time block index (default: 0). If None, read from config_file.
    figsize : tuple, optional
        Figure size (width, height). If None, read from config_file or use default (18, 10).
    output_amp : str, optional
        Output file path for amplitude plot. If None, read from config_file or use default.
    output_phase : str, optional
        Output file path for phase plot. If None, read from config_file or use default.
    dpi : int, optional
        DPI for saved figures. If None, read from config_file or use default 300.
    colors : dict, optional
        Color dictionary for polarizations. If None, read from config_file or use defaults.
    alpha_main : float, optional
        Alpha value for main polarizations. If None, read from config_file or use default 0.6.
    alpha_cross : float, optional
        Alpha value for cross polarizations. If None, read from config_file or use default 0.2.
    show : bool, optional
        If True, show plots instead of saving them (default: False).
        
    Returns
    -------
    dict
        Dictionary containing loaded data and metadata
    """
    
    # Load configuration from config.yaml
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"[WARNING] Config file '{config_file}' not found. Using defaults.")
        config = {}
    except Exception as e:
        print(f"[WARNING] Error reading config file '{config_file}': {e}. Using defaults.")
        config = {}
    
    # Use config defaults if parameters not provided
    if input_file is None:
        input_file = config.get('calibration_input_file', config.get('input_file', 'hyp_soln_1184702048_ssins_30l_src8k_300it.fits'))
    
    if i_timeblock is None:
        i_timeblock = config.get('i_timeblock', 0)
    
    if figsize is None:
        figsize_config = config.get('figsize', [18, 10])
        if isinstance(figsize_config, list):
            figsize = tuple(figsize_config)
        else:
            figsize = (18, 10)
    
    if output_amp is None:
        output_amp = config.get('output_amp', './Amplitude Solutions vs Frequency.png')
    
    if output_phase is None:
        output_phase = config.get('output_phase', './Phase Solutions vs Frequency.png')
    
    if dpi is None:
        dpi = config.get('dpi', 300)
    
    if colors is None:
        colors = config.get('colors', {'xx': 'blue', 'yy': 'green', 'xy': 'blue', 'yx': 'green'})
    
    if alpha_main is None:
        alpha_main = config.get('alpha_main', 0.6)
    
    if alpha_cross is None:
        alpha_cross = config.get('alpha_cross', 0.2)
    
    if show is None:
        show = config.get('show', False)
    
    print(f"[INFO] Loading calibration solutions from: {input_file}")
    print(f"[INFO] Using time block: {i_timeblock}")
    
    # Load FITS file
    fits_data = load_calibration_fits(input_file)
    data = fits_data['data']
    name_tiles = fits_data['name_tiles']
    freqs = fits_data['freqs']
    num_tiles = fits_data['num_tiles']
    f = fits_data['f']
    
    try:
        num_tiles_per_row = num_tiles // 16
        if num_tiles_per_row == 0:
            num_tiles_per_row = 1
        
        # Extract time block
        data_complex = extract_timeblock(data, i_timeblock)
        
        # Plot amplitude solutions
        print("[INFO] Plotting amplitude solutions...")
        fig_amp = plot_amplitude_solutions(
            data_complex, freqs, name_tiles, num_tiles, num_tiles_per_row,
            figsize, colors, alpha_main, alpha_cross, output_amp, dpi, show=show
        )
        
        # Plot phase solutions
        print("[INFO] Plotting phase solutions...")
        fig_phase = plot_phase_solutions(
            data_complex, freqs, name_tiles, num_tiles, num_tiles_per_row,
            figsize, colors, alpha_main, alpha_cross, output_phase, dpi, show=show, unwrap=True
        )
        
        if show:
            plt.ion()  # Enable interactive mode for non-blocking display
            # Display all figures (non-blocking)
            plt.show(block=False)
            # Block at the end until all windows are manually closed
            print("[INFO] Displayed 2 plots. Close all windows to continue.")
            plt.show(block=True)
        else:
            print(f"[INFO] Saved amplitude plot: {output_amp}")
            print(f"[INFO] Saved phase plot: {output_phase}")
        
        return {
            'data': data_complex,
            'freqs': freqs,
            'name_tiles': name_tiles,
            'num_tiles': num_tiles,
            'fig_amp': fig_amp,
            'fig_phase': fig_phase
        }
        
    finally:
        f.close()

# ----------------------
# Main function
# ----------------------
def main():
    parser = argparse.ArgumentParser(
        description="Plot calibration solutions (amplitude/phase) vs frequency.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use config.yaml (all parameters from config file)
  python plot_calibration_solutions.py --config config.yaml

  # Override specific parameters from config.yaml
  python plot_calibration_solutions.py --config config.yaml --input_file soln.fits --i_timeblock 1

  # Use command line arguments only (no config file)
  python plot_calibration_solutions.py input_file.fits --i_timeblock 0 --output_amp ./amp.png

  # Show plots interactively instead of saving
  python plot_calibration_solutions.py input_file.fits --show
        """
    )
    parser.add_argument("input_file", nargs='?', default=None,
                        help="Input FITS file. Optional if using --config")
    parser.add_argument("--config", default='config.yaml',
                        help="Path to configuration YAML file (default: 'config.yaml')")
    parser.add_argument("--i_timeblock", type=int, help="Time block index (default: 0). Overrides config.yaml")
    parser.add_argument("--figsize", nargs=2, type=float, metavar=('WIDTH', 'HEIGHT'),
                        help="Figure size (width height). Overrides config.yaml")
    parser.add_argument("--output_amp", help="Output file path for amplitude plot. Overrides config.yaml")
    parser.add_argument("--output_phase", help="Output file path for phase plot. Overrides config.yaml")
    parser.add_argument("--dpi", type=int, help="DPI for saved figures. Overrides config.yaml")
    parser.add_argument("--alpha_main", type=float, help="Alpha value for main polarizations. Overrides config.yaml")
    parser.add_argument("--alpha_cross", type=float, help="Alpha value for cross polarizations. Overrides config.yaml")
    
    # Helper function to parse boolean strings for --show argument
    def str_to_bool(v):
        if v is None:
            return True  # --show without value means True
        if isinstance(v, bool):
            return v
        v_lower = str(v).lower()
        if v_lower in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v_lower in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError(f'Boolean value expected, got {v}')
    
    parser.add_argument("--show", nargs='?', const=True, default=None, type=str_to_bool,
                        help="Show plots interactively instead of saving PNGs. "
                             "Use --show (True) or --show False. If not specified, uses config.yaml value.")
    
    args = parser.parse_args()
    
    # Convert figsize tuple if provided
    figsize = None
    if args.figsize:
        figsize = tuple(args.figsize)
    
    # Call the main processing function
    run_calibration_plotting(
        config_file=args.config,
        input_file=args.input_file,
        i_timeblock=args.i_timeblock,
        figsize=figsize,
        output_amp=args.output_amp,
        output_phase=args.output_phase,
        dpi=args.dpi,
        alpha_main=args.alpha_main,
        alpha_cross=args.alpha_cross,
        show=args.show
    )

if __name__ == "__main__":
    main()
