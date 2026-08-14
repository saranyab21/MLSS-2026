"""
Shared plotting constants and tiny helpers.

The Okabe-Ito colourblind-safe palette was redefined in eight files
(control_figures, cross_dataset, figX5_multiseed, make_lda_panels,
make_umap_panels, make_umap_panels_isolated, tradeoff_figure, visualize).
One definition now lives here.

CB   : named dict  {'blue','orange','green','pink','grey', ...}
OI   : ordered list, for per-class cycling (LDA/UMAP panels)
load_json : read a results JSON, or return None if absent (figures skip
            gracefully rather than crash when a result file is missing)
"""
import json
import os

__all__ = ['CB', 'OI', 'load_json']

# Okabe-Ito. The `steel`/`amber` aliases match the poster's design tokens
# (steel = what was done, amber = what we found).
CB = {
    'blue':   '#0072B2',
    'orange': '#E69F00',
    'green':  '#009E73',
    'pink':   '#CC79A7',
    'grey':   '#999999',
    'yellow': '#F0E442',
    'skyblue': '#56B4E9',
    'vermillion': '#D55E00',
    'black':  '#000000',
    # poster aliases
    'steel':  '#0072B2',
    'amber':  '#E69F00',
    # aliases the existing figure code references by these names
    'purple': '#CC79A7',   # == pink
    'red':    '#D55E00',   # == vermillion
    'sky':    '#56B4E9',   # == skyblue
}

# Ordered palette for cycling across classes (matches the old OI lists).
OI = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#999999',
      '#F0E442', '#56B4E9', '#D55E00', '#000000']


def load_json(path):
    """Load a results JSON, or return None if it does not exist.

    Figure scripts use this so a missing result file skips one panel instead
    of aborting the whole figure build.
    """
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)
