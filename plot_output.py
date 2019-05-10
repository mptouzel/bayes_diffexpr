# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %matplotlib inline
import matplotlib as mpl
from pylab import rcParams
rcParams['font.size']=24
rcParams['lines.markersize']=7
rcParams['figure.figsize'] = 16, 8
rcParams['lines.markeredgewidth'] = 2
import seaborn as sns
sns.set()
mpl.rc("figure", facecolor="gray")
import numpy as np
import time
import pylab as pl
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

# # import data

# +
#path information
output_path='../output/'
null_pair='S2_0_F1_S2_0_F2'
diff_pair='S2_0_F1_S2_0_F2'
run_name='diffexpr_pair_'+null_pair+'_v1_2yr_code_check_min0_maxinf'
path=output_path+diff_pair+'/'+run_name+'/'

#load data
sbarvec=np.load(path+'sbarvec.npy')
alpvec=np.load(path+'alpvec.npy')
Lsurface=np.load(path+'LSurface.npy')
Pn1n2_s=np.load(path+'Pn1n2_s_d.npy')
# -

fig,ax=pl.subplots(1,1,figsize=(20,10))
p=ax.imshow(np.flipud(np.log10(-Lsurface)),extent=[sbarvec[0], sbarvec[-1],np.log10(alpvec[0]), np.log10(alpvec[-1])], interpolation='none')
pl.colorbar(p)
X, Y = np.meshgrid(sbarvec,np.log10(alpvec))
# ax.set_xscale('log')
ax.contour(X, Y, np.log10(-Lsurface),levels = 20,colors=('w',),linestyles=('--',),linewidths=(5,))
ax.set_title(r'$-L=-\sum_i log P(n_i,n''_i) $')
ax.set_ylabel(r'$\alpha$')
ax.set_xlabel(r'$\bar{s}$');
# fig.savefig("z_1dot5_surface.pdf",format='pdf',dpi=500,bbox_inches='tight')


