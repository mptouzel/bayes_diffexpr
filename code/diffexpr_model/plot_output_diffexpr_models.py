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
import matplotlib.pyplot as ppl
from matplotlib import rcParams
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

output_path='../output/'
donorstr='S1'
null_pair=donorstr+'_0_F1_'+donorstr+'_0_F2'
diff_pair=donorstr+'_0_F1_'+donorstr+'_15_F1'
run_name='diffexpr_pair_'+null_pair+'_v1_2yr_code_check_min0_maxinf'
path=output_path+diff_pair+'/'+run_name+'/'
Lsurface=np.load(path+'LSurface.npy')

donorstrvec=['P1','P2','Q1','Q2','S1','S2']
from matplotlib import rc
rc('text',usetex=True)
rc('font',family='serif')
for donorstr in donorstrvec:
    #path information
    output_path='../output/'
    null_pair=donorstr+'_0_F1_'+donorstr+'_0_F2'
    diff_pair=donorstr+'_0_F1_'+donorstr+'_15_F1'
    run_name='diffexpr_pair_'+null_pair+'_v1_2yr_code_check_min0_maxinf'
    path=output_path+diff_pair+'/'+run_name+'/'
    outstruct_null=np.load(output_path+null_pair+'/null_pair_v1_2yr_code_check_case0_min0_maxinf/outstruct.npy').item()
    print(str(outstruct_null.x)+str(outstruct_null.fun))
    #load data
    sbarvec=np.load(path+'sbarvec.npy')
    alpvec=np.load(path+'alpvec.npy')
    Lsurface=np.load(path+'LSurface.npy')
    Pn1n2_s=np.load(path+'Pn1n2_s_d.npy')
    fig,ax=pl.subplots(1,1,figsize=(20,10))
    p=ax.imshow(np.log10(-Lsurface),extent=[sbarvec[0], sbarvec[-1],np.log10(alpvec[0]), np.log10(alpvec[-1])], origin='lower',interpolation='none')
    ppl.colorbar(p)
    X, Y = np.meshgrid(sbarvec,np.log10(alpvec))
    # ax.set_xscale('log')
    ax.contour(X, Y, np.log10(-Lsurface),levels = 20,colors=('w',),linestyles=('--',),linewidths=(5,))
    ax.set_title(donorstr+r' $-L=-\sum_i \log P(n_i,n\prime_i)$')#' \bar{s}_{opt}='+str(np.load(path+'optsbar.npy').flatten()[0])+r' \log\alpha_{opt}='+str(np.log10(np.load(path+'optalp.npy').flatten()[0]))+r'$')
    print(str(np.max(Lsurface)))

    ax.scatter(np.load(path+'optsbar.npy').flatten()[0],np.log10(np.load(path+'optalp.npy').flatten()[0]),marker='o',c='w',s=500)
    optinds=np.unravel_index(np.argmin(np.log10(-Lsurface)),(len(sbarvec),len(alpvec)))
    print(optinds)
    ax.scatter(sbarvec[optinds[0]],np.log10(alpvec[optinds[1]]),marker='o',c='k',s=300)

    ax.set_ylabel(r'$\alpha$')
    ax.set_xlabel(r'$\bar{s}$');
    ax.set_xlim(0,sbarvec[-1])
    ax.set_ylim(np.log10(alpvec[0]),0)
    # fig.savefig("z_1dot5_surface.pdf",format='pdf',dpi=500,bbox_inches='tight')
