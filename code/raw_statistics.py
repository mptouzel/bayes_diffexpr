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
import sys,os
root_path = os.path.abspath(os.path.join('..'))
if root_path not in sys.path:
    sys.path.append(root_path)
# %run -i 'lib/utils/ipynb_setup.py'
import lib.utils.plotting
from lib.utils.plotting import plot_n1_vs_n2,add_ticks
from lib.proc import get_sparserep,import_data
# import lib.learning
# %load_ext autoreload
# %autoreload 2

import matplotlib.pyplot as pl
pl.rc("figure", facecolor="gray",figsize = (8,8))
pl.rc('lines',markeredgewidth = 2)
pl.rc('font',size = 24)
pl.rc('text', usetex=True)
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : True})

# Read in,transform, and save data

# +
rootpath = '../../'
daynull="0"
day="0"
headerline=0
colnames = [u'Clone fraction',u'Clone count',u'N. Seq. CDR3',u'AA. Seq. CDR3'] 
datarootpath=rootpath+'data/Yellow_fever/prepostvaccine/' 
mincount=0
maxcount=np.Inf

parvernull = 'raw_stats'
runstr = 'min' + str(mincount) + '_max' + str(maxcount) + '_' + parvernull 
# -

# Plot all null pairs

# +
donorstrvec = [ 'P1', 'P2',  'Q1',   'Q2',  'S1', 'S2']
daystrvec=['pre0','0','7','15','45']
for donor in donorstrvec:
    for daynull in daystrvec:
        dataset_pair=(donor+'_'+daynull+'_F1', donor+'_'+daynull+'_F2')
        Nclones_samp,subset=import_data(datarootpath,dataset_pair[0]+'_.txt',dataset_pair[1]+'_.txt',mincount,maxcount,colnames,colnames)
        #transform to sparse representation
        sparse_rep=get_sparserep(subset.loc[:,['Clone_count_1','Clone_count_2']])
        savepath = rootpath + 'output/'
        outpath = savepath + dataset_pair[0] + '_' + dataset_pair[1] + '/' + runstr + '/'
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        np.save(outpath+'sparse_rep.npy',sparse_rep)     
        
for dit,donor in enumerate(donorstrvec):
    for ddit,day in enumerate(daystrvec):
        savepath = rootpath + 'output/'
        dataset_pair=(donor+'_'+daynull+'_F1', donor+'_'+daynull+'_F2')
        outpath = savepath + dataset_pair[0] + '_' + dataset_pair[1] + '/' + runstr + '/'
        sparse_rep=np.load(outpath+'sparse_rep.npy')
        plot_n1_vs_n2(sparse_rep,'Pn1n2',outpath,True)
# -

# Plot all max responding day pairs

# +
donorstrvec = [ 'P1', 'P2',  'Q1',   'Q2',  'S1', 'S2']
daystrvec=['pre0','0','7','15','45']
for donor in donorstrvec:
    day='15'
    dataset_pair=(donor+'_'+daynull+'_F1', donor+'_'+day+'_F2')
    Nclones_samp,subset=import_data(datarootpath,dataset_pair[0]+'_.txt',dataset_pair[1]+'_.txt',mincount,maxcount,colnames,colnames)
    #transform to sparse representation
    sparse_rep=get_sparserep(subset.loc[:,['Clone_count_1','Clone_count_2']])
    savepath = rootpath + 'output/'
    outpath = savepath + dataset_pair[0] + '_' + dataset_pair[1] + '/' + runstr + '/'
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    np.save(outpath+'sparse_rep.npy',sparse_rep)       

for dit,donor in enumerate(donorstrvec):
    savepath = rootpath + 'output/'
    day='15'
    dataset_pair=(donor+'_'+daynull+'_F1', donor+'_'+day+'_F2')
    outpath = savepath + dataset_pair[0] + '_' + dataset_pair[1] + '/' + runstr + '/'
    sparse_rep=np.load(outpath+'sparse_rep.npy')
    #         indn1,indn2,sparse_rep_counts,unicountvals_1,unicountvals_2,NreadsI,NreadsII=sparse_rep
    plot_n1_vs_n2(sparse_rep,'Pn1n2',outpath,True)
#         N_1=np.sum((uni1[indn1]>0)*countpairs)
#         N_2=np.sum((uni2[indn2]>0)*countpairs)
#         N_all=np.sum(countpairs)
#         print(str(N_all)+' '+str(N_1)+' '+str(N_2))
# -

# Check negative correlation between log fold change and initial size

for dit,donor in enumerate(donorstrvec):
    savepath = rootpath + 'output/'
    fig,ax=pl.subplots(1,2)
    daynull='0'
    day='0'
    dataset_pair=(donor+'_'+daynull+'_F1', donor+'_'+day+'_F2')
    outpath = savepath + dataset_pair[0] + '_' + dataset_pair[1] + '/' + runstr + '/'
    sparse_rep=np.load(outpath+'sparse_rep.npy')
    indn1,indn2,sparse_rep_counts,unicountvals_1,unicountvals_2,NreadsI,NreadsII=sparse_rep
    n1=unicountvals_1[indn1]
    n2=unicountvals_2[indn2]
    both_nonzero=np.logical_and(n1>0,n2>0)
    logn2n1=np.where(both_nonzero,np.log(n2/n1),0)
    ax[0].scatter(n1[both_nonzero],logn2n1[both_nonzero])
    ax[0].set_xlim(0.5,1e5)
    ax[0].set_ylim(-5,5)
    ax[0].set_xlabel('$n_1$')
    ax[0].set_ylabel(r'$\log \frac{n_2}{n_1}$')
    ax[0].set_xscale('log')
    ax[0].set_title('day 0 - day 0')
    day='15'
    dataset_pair=(donor+'_'+daynull+'_F1', donor+'_'+day+'_F2')
    outpath = savepath + dataset_pair[0] + '_' + dataset_pair[1] + '/' + runstr + '/'
    sparse_rep=np.load(outpath+'sparse_rep.npy')
    indn1,indn2,sparse_rep_counts,unicountvals_1,unicountvals_2,NreadsI,NreadsII=sparse_rep
    n1=unicountvals_1[indn1]
    n2=unicountvals_2[indn2]
    both_nonzero=np.logical_and(n1>0,n2>0)
    logn2n1=np.where(both_nonzero,np.log(n2/n1),0)
    ax[1].scatter(n1[both_nonzero],logn2n1[both_nonzero])
    ax[1].set_xlabel('$n_1$')
    ax[1].set_xlim(0.5,1e5)
    ax[1].set_ylim(-5,5)

#     ax[1].set_ylabel('$\log n_2/n1$')
    ax[1].set_xscale('log')
    ax[1].set_title('day 0 - day 15')
    fig.suptitle(donor)
    fig.tight_layout()
    fig.savefig(donor+'log_n2_n1_vs_n1.pdf',format= 'pdf',dpi=300, bbox_inches='tight')

# Plot distribution of naive fold change

# +
rootpath = '../../'
mincount=0
maxcount=np.Inf
parvernull = 'raw_stats'
runstr = 'min' + str(mincount) + '_max' + str(maxcount) + '_' + parvernull 

donorstrvec = [ 'P1', 'P2',  'Q1',   'Q2',  'S1', 'S2']
daystrvec=['pre0','0','7','15','45']

for dit,donor in enumerate(donorstrvec):
    savepath = rootpath + 'output/'
    fig,ax=pl.subplots(1,2,figsize=(8,4))
    daynull='0'
    for it,day in enumerate(['0','15']):
        dataset_pair=(donor+'_'+daynull+'_F1', donor+'_'+day+'_F2')
        outpath = savepath + dataset_pair[0] + '_' + dataset_pair[1] + '/' + runstr + '/'
        sparse_rep=np.load(outpath+'sparse_rep.npy')
        indn1,indn2,sparse_rep_counts,unicountvals_1,unicountvals_2,NreadsI,NreadsII=sparse_rep
        n1=unicountvals_1[indn1]
        n2=unicountvals_2[indn2]
        both_nonzero=np.logical_and(n1>10,n2>10)
        logn2n1=np.where(both_nonzero,np.log(n2/n1),0)
        nbinvec=np.linspace(-5,5,50)
        counts,bins=np.histogram(logn2n1[both_nonzero],nbinvec)
        ax[it].plot(bins[:-1],counts,'.')
        ax[it].plot(bins[:-1],200*np.exp(-(bins[:-1]/0.5)**2),'-')
        ax[it].plot(bins[:-1],200*np.exp(-np.fabs(bins[:-1]/0.5)),'-')
        #         ax[it].set_xlim(0.5,1e5)
        ax[it].set_ylim(0.5,1e3)
        ax[it].set_ylabel('count')
        ax[it].set_xlabel(r'$\log \frac{n_2}{n_1}$')
        ax[it].set_yscale('log')
        ax[it].set_title('day 0 - day '+day)
    fig.suptitle(donor)
    fig.tight_layout()
    fig.savefig(donor+'log_n2_n1_hist_greaterthan_10.pdf',format= 'pdf',dpi=300, bbox_inches='tight')
# -

logn2n1.shape


