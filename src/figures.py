#### ==========================================================================
#### Dissertation figures
#### Author: Simon Schramm
#### 12.03.2022
#### --------------------------------------------------------------------------
""" 
This script provides basic settings for all figures of the dissertation.
""" 
### ---------------------------------------------------------------------------
#%% Preamble.
### ---------------------------------------------------------------------------
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap
### ---------------------------------------------------------------------------
#%% Graph options.
### ---------------------------------------------------------------------------
# Specify some graph options for the dissertation.
DPI = 300
fontsize = 24
figsize = (16,9)
plt.rc('savefig', dpi=DPI)
plt.rcParams['figure.dpi'] = DPI
plt.rcParams.update({'font.size': fontsize})
plt.rcParams.update({'axes.labelsize': fontsize, 'xtick.labelsize': fontsize, 'ytick.labelsize': fontsize})
plt.rcParams['figure.figsize'] = 6.4, 4.8  # Default.
plt.rcParams['xtick.major.pad'] = 5 # Default: 3.
plt.rcParams['ytick.major.pad'] = 5 # Default: 3.
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
# Use the predefined BMW color scheme.
bmw_color = '#035970'
#petrol = '#004750'
petrol = '#34544A'
berry = '#69103B'
pink = '#8D1E77'
light_pink = '#A33C8D' 
purple = '#4B277B'
light_purple = '#794694'
light_blue = '#04829B'
blue = '#035970'
dark_blue = '#173B68'
# green = '#508130'
green = '#637F44' # COLOR 2.
red = '#B11926'
orange = '#E96D0C'
# yellow = '#FAD022'
yellow = '#FDF2D0' # COLOR 3.
dark_grey = '#382E2C'
grey = '#494949'
lighter_grey = '#6F6F6F'
light_grey = '#A8A8A7'
darker_blue = '#0083C6'
lighter_blue = '#468AC9'

#
def set_graph_options():
    """Set graph options for the dissertation."""
    # Set font size.
    plt.rcParams.update({'font.size': 16})
    # Set figure size.
    plt.rcParams['figure.figsize'] = 6.4, 4.8  # Default.
    # Set padding of ticks.
    plt.rcParams['xtick.major.pad'] = 5 # Default: 3.
    plt.rcParams['ytick.major.pad'] = 5 # Default: 3.
    # Set font family.
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    # Set color scheme.
    bmw_color = '#035970'
    # Create bmw colors as matplolib colormap.
    colors = (bmw_color, green, petrol, lighter_blue, darker_blue, berry,
              yellow, orange, red, pink, light_pink, light_blue, blue , dark_blue,
              purple, light_purple, light_grey, lighter_grey, dark_grey, grey)
    bmw_colors = LinearSegmentedColormap.from_list("bmw_colormap", colors)
    # Set formatter for scientific notation.
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    return bmw_color, bmw_colors
### ---------------------------------------------------------------------------
### End.
#### ==========================================================================
