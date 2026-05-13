import matplotlib.pyplot as plt

def set_winplot_dark():
    """
    Apply a dark-theme plotting style to Matplotlib figures.

    This function updates Matplotlib's global ``rcParams`` to produce
    high-contrast, dark-background plots suitable for presentations,
    reports, and screen viewing. It modifies figure, axes, text,
    grid, and legend styling.

    Styling Changes
    ----------------
    - Black background for figures and axes
    - White text, axis labels, ticks, and spines
    - Subtle gray grid lines for readability
    - Dark legend background with white border

    Notes
    -----
    - This function modifies Matplotlib's global configuration via
      ``plt.rcParams.update(...)``. All subsequent plots in the current
      Python session will use this style unless rcParams are reset.
    - To revert to Matplotlib defaults, use:

        >>> plt.rcdefaults()

    Examples
    --------
    >>> set_winplot_dark()
    >>> plt.plot([0, 1], [0, 1])
    >>> plt.title("Example Plot")
    >>> plt.show()
    """
    plt.rcParams.update({
        "figure.facecolor": "black",
        "axes.facecolor": "black",
        "savefig.facecolor": "black",
        "text.color": "white",
        "axes.labelcolor": "white",
        "axes.edgecolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "grid.color": "0.25",
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "legend.frameon": True,
        "legend.facecolor": "black",
        "legend.edgecolor": "white",
    })