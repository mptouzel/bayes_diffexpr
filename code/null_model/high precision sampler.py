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

# ### The limited pecision of scipy's nbinom sampler has adverse affects when sampling.
# ### These can be overcome. We implement a custom sampler to that effect.

# %matplotlib inline
import sys,os
root_path = os.path.abspath(os.path.join('..'))
if root_path not in sys.path:
    sys.path.append(root_path)
# %run -i '../lib/utils/ipynb_setup.py'
import lib.utils.plotting
from lib.utils.plotting import plot_n1_vs_n2,add_ticks
from lib.utils.prob_utils import get_distsample
from lib.proc import get_sparserep,import_data
# %load_ext autoreload
# %autoreload 2
from lib.model import get_rhof, NegBinParMtr,get_logPn_f,get_model_sample_obs
from lib.utils.prob_utils import get_distsample
from scipy.stats import nbinom


import matplotlib.pyplot as pl
pl.rc("figure", facecolor="gray",figsize = (8,8))
pl.rc('lines',markeredgewidth = 2)
pl.rc('font',size = 24)
pl.rc('text', usetex=True)
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : True})

# We sample over a range of means

# +
batchsize=int(1e5)
fig1=pl.figure()
ax1=fig1.add_subplot(111)
paras=np.asarray([-2.15199494,  1.27984874,  1.02263351,  -9.73615977])
Nreads=1464283#1498170+1430396

alpha_rho = paras[0]
fmin=np.power(10,paras[3])
logrhofvec,logfvec = get_rhof(alpha_rho,fmin) 
dlogfby2=np.diff(logfvec)/2. #two for custom trapezoid

beta_mv= paras[1]
alpha_mv=paras[2]

m=float(Nreads)*np.exp(logfvec)
v=m+beta_mv*np.power(m,alpha_mv)
pvec=1-m/v
nvec=m*m/v/pvec
Pn_f=np.empty((len(logfvec),),dtype=object) #define a new Pn2_f on shifted domain at each iteration
for find,(n,p) in enumerate(zip(nvec,pvec)):
    Pn_f[find]=nbinom(n,1-p)
nmax=10000      
for find in range(0,len(logfvec),20):
    nvec = Pn_f[find].rvs(batchsize) # model to array pairs
    if len(np.unique(nvec))>1:
        [Pn_ftest,nvectest] = np.histogram(nvec,range(max(nvec)))
        ax1.plot(nvectest[:-1],Pn_ftest/batchsize,'r.',label=r'$\rho(f)$ sampled')
        ax1.plot(range(nmax), Pn_f[find].pmf(range(nmax)),'k-',label=r'$\rho(f)$ actual')
        ax1.set_yscale('log')
        ax1.set_xscale('log')

        ax1.set_ylim([1e-300,1e3])
        ax1.set_xlim([1,nmax])#unicountvals_1[-1]])
# -

n1vec.shape

# +
case=2
unin=range(nmax)
Pn1_f=np.exp(get_logPn_f(unin,Nreads,logfvec,case,paras))

fig1=pl.figure()
ax1=fig1.add_subplot(111)
for find in range(0,len(logfvec),20):

    frac=np.sum(Pn1_f[find,:])
#     print(str(frac)+' ',end='')
#     print(int(float(batchsize)*frac))

    n1vec = get_distsample(Pn1_f[find,:], int(float(batchsize)*frac)).flatten()  # model to array pairs
#     n1vec = get_modelsample(Pn1_f[find,:], batchsize,seed2)  # model to array pairs
    if len(np.unique(n1vec))>1:
        [Pn1_ftest,n1vectest] = np.histogram(n1vec,range(int(max(n1vec))))
        ax1.plot(n1vectest[:-1],Pn1_ftest/float(batchsize),'r.',label=r'$\rho(f)$ sampled')
        ax1.plot(unin, Pn1_f[find,:],'k-',label=r'$\rho(f)$ actual')
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_ylim([1e-300,1e3])

ax1.set_xlim([1,nmax])#unicountvals_1[-1]])
# -


