#!/usr/bin/env python3

from pathlib import Path
import click
import matplotlib as mpl
import matplotlib.pyplot as plt

from pspipe import database, settings

mpl.rcParams['image.cmap'] = 'twilight'
mpl.rcParams['image.origin'] = 'lower'
mpl.rcParams['image.interpolation'] = 'nearest'
mpl.rcParams['axes.grid'] = True


@click.command()
@click.argument("rev_file", type=click.Path(exists=True))
@click.argument("obs_id")
@click.option("--fmin", type=float, default=50, show_default=True,
              help="Minimum frequency in MHz")
@click.option("--fmax", type=float, default=250, show_default=True,
              help="Maximum frequency in MHz")
@click.option("--pol", type=str, default="I", show_default=True,
              help="Comma-separated list of Stokes/pols (e.g. 'I,Q' or 'XX,YY')")
def main(rev_file, obs_id, fmin, fmax, pol):
    """Produce power spectrum products (2D, 1D, variance) for given OBS_ID and polarisations."""

    # Load revision and data
    rev = database.VisRevision(settings.Settings.load_with_defaults(str(rev_file)))
    data = rev.get_data(obs_id)
    data.do_flag()

    # Power spectrum generator
    ps_gen = data.get_ps_gen(fmhz_range=[fmin, fmax], ft_method='lssa')

    # Output dir in the parent of rev_file
    outdir = Path(rev_file).resolve().parent / "results" / obs_id
    outdir.mkdir(parents=True, exist_ok=True)

    pols = [p.strip() for p in pol.split(",") if p.strip()]
    # If only one polarisation, wsclean output just one set of images
    if len(pols) == 1:
        pols = ['I']

    # Loop over polarisations
    for pol_name in pols:
        click.echo(f"Processing polarisation {pol_name}")

        stokes_data = data.get_stokes(pol_name)
        stokes_even = data.get_stokes(f"even_{pol_name}")
        stokes_odd  = data.get_stokes(f"odd_{pol_name}")

        # Compute PS products
        ps2d = ps_gen.get_cross_ps2d(stokes_even, stokes_odd)
        ps1d = ps_gen.get_ps(stokes_data)
        var  = ps_gen.get_variance(stokes_data)

        # Plot and save 2D cross PS
        fig, ax = plt.subplots(figsize=(6, 5))
        ps2d.plot(ax=ax, log_axis=True)
        ax.set_title(f"Cross 2D Power Spectrum – {pol_name}\n{obs_id}")
        outfile = outdir / f"{obs_id}_{rev.settings.image.name}_{pol_name}_ps2d.pdf"
        fig.savefig(outfile)
        plt.tight_layout()
        click.echo(f"Saved Stokes {pol_name} 2D PS figure to {outfile}")

        fig, ax = plt.subplots(figsize=(6, 5))
        ps2d.plot_kpar(ax=ax)
        ax.set_title(f"Cross 2D Power Spectrum – {pol_name}\n{obs_id}")
        ax.set_xscale('log')
        outfile = outdir / f"{obs_id}_{rev.settings.image.name}_{pol_name}_ps2d_kpar.pdf"
        fig.savefig(outfile)
        plt.tight_layout()
        click.echo(f"Saved Stokes {pol_name} 2D PS figure to {outfile}")

        # Plot and save 1D PS
        fig, ax = plt.subplots(figsize=(6, 5))
        ps1d.plot(ax=ax, label=pol_name)
        ax.set_title(f"1D Power Spectrum – {pol_name}\n{obs_id}")
        ax.legend()
        outfile = outdir / f"{obs_id}_{rev.settings.image.name}_{pol_name}_ps1d.pdf"
        fig.savefig(outfile)
        plt.tight_layout()
        click.echo(f"Saved Stokes {pol_name} 1D PS figure to {outfile}")

        # Plot and save Variance
        fig, ax = plt.subplots(figsize=(6, 5))
        var.plot(ax=ax, label=pol_name)
        ax.set_title(f"Variance – {pol_name}\n{obs_id}")
        ax.legend()
        outfile = outdir / f"{obs_id}_{rev.settings.image.name}_{pol_name}_variance.pdf"
        fig.savefig(outfile)
        plt.tight_layout()
        click.echo(f"Saved Stokes {pol_name} Variance figure to {outfile}")


if __name__ == "__main__":
    main()
