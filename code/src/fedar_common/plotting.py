"""
Shared plotting constants and smol helpers    

The Okabe-Ito colourblind-safe palette was redefined in eight files
(control_figures, cross_dataset, figX5_multiseed, make_lda_panels,
make_umap_panels, make_umap_panels_isolated, tradeoff_figure, visualize).
One definition now lives here.

CB   : named dict  {'blue','orange','green','pink','grey', ...}
OI   : ordered list, for per-class cycling (LDA/UMAP panels)
POSTER : task-vs-demographic sequence tuned to the poster's steel/amber tokens
apply_poster_style : set rcParams so every figure matches the poster
load_json : read a results JSON, or return None if absent (figures skip
            gracefully rather than crash when a result file is missing)
"""
import json
import os

__all__ = ['CB', 'OI', 'POSTER', 'apply_poster_style', 'load_json']

# Okabe-Ito colourblind-safe base. The `steel`/`amber` aliases match the
# poster's design tokens (steel = what was done, amber = what we found).
# steel/amber are nudged from raw Okabe-Ito to sit exactly on the poster
# fills (#1F5C8B steel, #C77F0A amber) so plots and poster read as one piece.
CB = {
    'blue':   '#0072B2',
    'orange': '#E69F00',
    'green':  '#009E73',
    'pink':   '#CC79A7',
    'grey':   '#8A94A0',    # warmer, less muddy than #999999 on paper
    'yellow': '#F0E442',
    'skyblue': '#56B4E9',
    'vermillion': '#D55E00',
    'black':  '#1A1A1A',    # near-black, matches poster ink (not pure #000)
    # poster aliases — tuned to the exact poster fills
    'steel':  '#1F5C8B',
    'steel_light': '#5B8CB3',
    'amber':  '#C77F0A',
    'amber_light': '#E3A94A',
    'amber_deep':  '#8F5C06',
    # aliases the existing figure code references by these names
    'purple': '#CC79A7',   # == pink
    'red':    '#D55E00',   # == vermillion
    'sky':    '#56B4E9',   # == skyblue
}

# Ordered palette for cycling across classes (LDA / UMAP panels).
# Lead with steel + amber so the dominant classes read in the poster's voice.
OI = ['#1F5C8B', '#C77F0A', '#009E73', '#CC79A7', '#8A94A0',
      '#56B4E9', '#D55E00', '#F0E442', '#1A1A1A']

# Task-vs-demographic bar sequence: the task in steel, the three leaked
# demographics in a graded amber family (light -> deep). This makes the hero
# chart say "the task vs. what leaked" in the poster's own two colours, rather
# than an unrelated rainbow.  Order: [task, demo, demo, demo].
POSTER = ['#1F5C8B', '#E3A94A', '#C77F0A', '#8F5C06']


def apply_poster_style():
    """Set matplotlib rcParams so every figure matches the poster.

    Call once at the top of a figure script (after importing pyplot):
        import matplotlib.pyplot as plt
        from fedar_common.plotting import apply_poster_style
        apply_poster_style()

    Safe to call repeatedly. Only touches global style, never data.
    """
    import matplotlib as mpl
    mpl.rcParams.update({
        'axes.edgecolor':   '#4A5560',
        'axes.labelcolor':  '#1A1A1A',
        'axes.titlecolor':  '#1A1A1A',
        'axes.linewidth':   1.0,
        'axes.grid':        True,
        'grid.color':       '#E3E8EE',
        'grid.linewidth':   0.8,
        'axes.axisbelow':   True,           # gridlines behind bars
        'xtick.color':      '#4A5560',
        'ytick.color':      '#4A5560',
        'text.color':       '#1A1A1A',
        'font.size':        11,
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.dpi':      200,
        'savefig.bbox':     'tight',
    })


def load_json(path):
    """Load a results JSON, or return None if it does not exist.

    Figure scripts use this so a missing result file skips one panel instead
    of aborting the whole figure build.
    """
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)