#computation
import numpy as np
import pandas as pd

#plotting
#import matplotlib.pyplot as pl
#import seaborn as sns
#sns.set_style("whitegrid", {'axes.grid' : True})
#pl.rc("figure", facecolor="gray",figsize = (8,8))
#pl.rc('text', usetex=True)
#pl.rc('lines',markeredgewidth = 2)
#pl.rc('font',size = 24)
#pl.rc('text', usetex=True)

#utils
import time
from copy import deepcopy
from functools import partial
import itertools
from importlib import reload

#notebook config
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
from IPython.display import Math
