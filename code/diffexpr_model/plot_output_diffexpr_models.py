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

# +
# %matplotlib inline
import sys,os
root_path = os.path.abspath(os.path.join('..'))
if root_path not in sys.path:
    sys.path.append(root_path)
# %run -i '../lib/utils/ipynb_setup.py'
# import lib.utils.plotting
# from lib.utils.plotting import plot_n1_vs_n2,add_ticks
# from lib.utils.prob_utils import get_distsample
# from lib.proc import get_sparserep,import_data
from lib.model import get_logPs_pm
# from lib.learning import constr_fn,callback,learn_null_model
# import lib.learning
# %load_ext autoreload
# %autoreload 2

# from scipy.interpolate import interp1d
# from scipy import stats
# from scipy.stats import poisson
# from scipy.stats import nbinom
# from scipy.stats import rv_discrete
# -

import matplotlib.pyplot as pl
pl.rc("figure", facecolor="gray",figsize = (8,8))
pl.rc('lines',markeredgewidth = 2)
pl.rc('font',size = 24)
pl.rc('text', usetex=True)
import seaborn as sns
# sns.set_style("whitegrid", {'axes.grid' : True})
params= {'text.latex.preamble' : [r'\usepackage{amsmath}']}
pl.rcParams.update(params)

# # import data

params= {'text.latex.preamble' : [r'\usepackage{amsmath}']}
pl.rcParams.update(params)

np.log10(np.exp(25))


# + {"code_folding": []}
def plot_pair(runname,day,startind,pl):
    output_path='../../../output/'
    donorstr='S2'
    null_pair=donorstr+'_0_F1_'+donorstr+'_0_F2'
    diff_pair=donorstr+'_0_F1_'+donorstr+'_'+str(day)+'_F2'

    run_name='diffexpr_pair_'+null_pair+'_'+runname
    outpath=output_path+diff_pair+'/'+run_name+'/'

    #Assemble grid
    alpvec=np.load(outpath+'alpvec.npy')
    sbarvec_p=np.load(outpath+'sbarvec_p.npy')
    # sbarvec_p=np.linspace(0.1,5.,20)
    LSurfacestore=np.zeros((len(alpvec),len(sbarvec_p)))
    nit_list=np.zeros((len(alpvec),len(sbarvec_p)))
    shiftMtr=np.zeros((len(alpvec),len(sbarvec_p)))
    Zstore=np.zeros((len(alpvec),len(sbarvec_p)))
    Zdashstore=np.zeros((len(alpvec),len(sbarvec_p)))
    time_elapsed=np.zeros((len(alpvec),len(sbarvec_p)))

    for bit,bet in enumerate(sbarvec_p):
        dim=(slice(None),bit)
        try:
            LSurfacestore[dim]=np.load(outpath+'Lsurface'+str(bit)+'.npy')
            nit_list[dim]=np.load(outpath+'nit_list'+str(bit)+'.npy')
            shiftMtr[dim]=np.load(outpath+'shift'+str(bit)+'.npy')
            Zstore[dim]=np.load(outpath+'Zstore'+str(bit)+'.npy')
            Zdashstore[dim]=np.load(outpath+'Zdashstore'+str(bit)+'.npy')
            time_elapsed[dim]=np.load(outpath+'time_elapsed'+str(bit)+'.npy')/3600.
        except:
            print(bit)

    #excise data
    Lsurface=LSurfacestore[:,startind:]
    Zdashstore=Zdashstore[:,startind:]
    Zstore=Zstore[:,startind:]
    
    #handle 0 values
    minval=-1.95#np.min(np.min(Lsurface))
    Lsurface[Lsurface==0]=minval
    Lsurface[Lsurface<minval]=minval

    alpvec=np.log10(alpvec)
    sbarvec=np.log10(sbarvec_p[startind:])
    X, Y = np.meshgrid(sbarvec,alpvec)

    fig,ax=pl.subplots(1,2,figsize=(20,10))
    titlestrvec=(r"$\langle \mathcal{L} \rangle$",r"$\log_{10}\left(|\langle \mathcal{L} \rangle -\langle \mathcal{L} \rangle_{\textrm{max}} |\right)$")
    for it,titlestr in enumerate(titlestrvec):
        if it==1:
            fac=np.inf
            fac=1+1e-2
            maxL=np.max(np.max(Lsurface))
            Lsurface[Lsurface<fac*maxL]=fac*maxL
            Lsurface=np.log10(maxL- Lsurface)

        p=ax[it].imshow(Lsurface,extent=[sbarvec[0], sbarvec[-1],alpvec[0], alpvec[-1]], aspect='auto',origin='lower',interpolation='none')#,cmap='viridis')
        ax[it].contour(X, Y, Lsurface,levels = 5,colors=('w',),linestyles=('--',),linewidths=(5,))
        cb=pl.colorbar(p,ax=ax[it])
        ax[it].contour(X, Y, Zstore,levels = [1.],colors=('k',),linestyles=('-',),linewidths=(5,))
        ax[it].contour(X, Y, Zstore,levels = [1.4],colors=('k',),linestyles=('--',),linewidths=(5,))
        ax[it].contour(X, Y, Zdashstore,levels = [1.],colors=('gray',),linestyles=('-',),linewidths=(5,))
        ax[it].contour(X, Y, Zdashstore,levels = [1.4],colors=('gray',),linestyles=('--',),linewidths=(5,))

        ax[it].set_ylabel(r'$\log_{10}\alpha$')
        ax[it].set_xlabel(r'$\log_{10}\bar{s}$')
        ax[it].set_title(titlestr)
    fig.tight_layout()
    fig.suptitle(runname.split('_')+['0-'+str(day)],y=1.02)
    fig.savefig("surface_"+runname+"_"+str(day)+".pdf",format='pdf',dpi=500,bbox_inches='tight')
    Lsurface=LSurfacestore[:,startind:]
    print(str(np.max(Lsurface[Lsurface!=0.])))
    return outpath


# -

np.sum(np.exp(get_logPs_pm(alp,bet,sbar_m,sbar_p,smax,s_step,Ps_type)))

smax=25
stp=0.1
smaxt=round(smax/stp)
Ps=np.zeros(2*int(smaxt)+1)
lambp=-stp/sbar_p
Z_p=2*(np.exp((smaxt+1)*lambp)-1)/(np.exp(lambp)-1)-2 #no s=0 contribution
Ps[:int(smaxt)]=np.exp(lambp*np.fabs(np.arange(0-int(smaxt),           0)))/Z_p
Ps[int(smaxt)+1:]  =np.exp(lambp*np.fabs(np.arange(           1,int(smaxt)+1)))/Z_p
Ps*=alp
Ps[int(smaxt)]=(1-alp) #the sole contribution to s=0
print(np.sum(Ps))

for Ps_type in ('rhs_only','cent_gauss','sym_exp'):
    for day in [15]:
        startind=4 if day==0 else 0
        outpath=plot_pair('v1_ct_1_mt_2_st_'+Ps_type+'_min0_maxinf',day,startind,pl)

    alpvec=np.load(outpath+'alpvec.npy')
    sbarvec_p=np.load(outpath+'sbarvec_p.npy')
    bet=0
    sbar_m=0
    fig,ax=pl.subplots(1,1,figsize=(8,8))
    smax=25
    s_step=0.1
    for alp in alpvec:
        for sbar_p in sbarvec_p:
            ax.plot(svec,np.exp(get_logPs_pm(alp,bet,sbar_m,sbar_p,smax,s_step,Ps_type)))
    ax.set_yscale('log')
    ax.set_ylim(1e-10,1e1)
    fig.savefig("Ps_type_"+Ps_type+".pdf",format='pdf',dpi=500,bbox_inches='tight')


svec=np.load('/home/max/Dropbox/scripts/Projects/immuno/diffexpr/output/S2_0_F1_S2_15_F2/diffexpr_pair_S2_0_F1_S2_0_F2_v1_ct_1_mt_2_st_offcent_gauss_min0_maxinf/svec.npy')

np.unravel_index(np.argmax(LSurfacestoretmp),dim)

# +

runname='v1_ct_1_mt_2_st_offcent_gauss_min0_maxinf'
day=15
startind=0
output_path='../../../output/'
donorstr='S2'
null_pair=donorstr+'_0_F1_'+donorstr+'_0_F2'
diff_pair=donorstr+'_0_F1_'+donorstr+'_'+str(day)+'_F2'

run_name='diffexpr_pair_'+null_pair+'_'+runname
outpath=output_path+diff_pair+'/'+run_name+'/'

#Assemble grid
sbarvec_m=np.load(outpath+'offset.npy')
alpvec=np.load(outpath+'alpvec.npy')
sbarvec_p=np.load(outpath+'sbarvec_p.npy')
# sbarvec_p=np.linspace(0.1,5.,20)
dim=(len(alpvec),len(sbarvec_m),len(sbarvec_p))
LSurfacestore=np.zeros(dim)
nit_list=np.zeros(dim)
shiftMtr=np.zeros(dim)
Zstore=np.zeros(dim)
Zdashstore=np.zeros(dim)
time_elapsed=np.zeros(dim)

for bit,bet in enumerate(sbarvec_p):
    dim=(slice(None),slice(None),bit)
    try:
        LSurfacestore[dim]=np.load(outpath+'Lsurface'+str(bit)+'.npy')
        nit_list[dim]=np.load(outpath+'nit_list'+str(bit)+'.npy')
        shiftMtr[dim]=np.load(outpath+'shift'+str(bit)+'.npy')
        Zstore[dim]=np.load(outpath+'Zstore'+str(bit)+'.npy')
        Zdashstore[dim]=np.load(outpath+'Zdashstore'+str(bit)+'.npy')
        time_elapsed[dim]=np.load(outpath+'time_elapsed'+str(bit)+'.npy')/3600.
    except:
        print(bit)
LSurfacestore=LSurfacestore
nit_list=nit_list
shiftMtr=shiftMtr
Zstore=Zstore
Zdashstore=Zdashstore
time_elapsed=time_elapsed


#load data
Lsurface=LSurfacestore[:,0,:]#[:,startind:]
minval=-1.95
Lsurface[Lsurface==0]=minval#np.min(np.min(Lsurface))
Lsurface[np.isnan(Lsurface)]=minval#np.min(np.min(Lsurface))
Lsurface[Lsurface<minval]=minval#np.min(np.min(Lsurface))

alpvec=alpvec
sbarvec=np.log10(sbarvec_p[startind:])
X, Y = np.meshgrid(sbarvec,np.log10(alpvec))

fig,ax=pl.subplots(1,2,figsize=(20,10))

maxL=np.max(np.max(Lsurface))
p=ax[0].imshow(Lsurface,extent=[sbarvec[0], sbarvec[-1],np.log10(alpvec[0]), np.log10(alpvec[-1])], aspect='auto',origin='lower',interpolation='none')#,cmap='viridis')
ax[0].contour(X, Y, Lsurface,levels = 5,colors=('w',),linestyles=('--',),linewidths=(5,))
pl.colorbar(p,ax=ax[0])
ax[0].set_ylabel(r'$\alpha$')
ax[0].set_xlabel(r'$\bar{s}$');
ax[0].set_ylim(np.log10(alpvec[0]),0)
ax[0].set_title(r"$\langle \mathcal{L} \rangle$")
# ax[0].set_aspect('equal')

fac=np.inf
fac=1+1e-2
Lsurface[Lsurface<fac*maxL]=fac*maxL
p2=ax[1].imshow(np.log10(-(Lsurface-maxL)),extent=[sbarvec[0], sbarvec[-1],np.log10(alpvec[0]), np.log10(alpvec[-1])], aspect='auto',origin='lower',interpolation='none')#,cmap='viridis')
pl.colorbar(p2,ax=ax[1])
ax[1].set_ylabel(r'$\alpha$')
ax[1].set_xlabel(r'$\bar{s}$');
ax[1].set_ylim(np.log10(alpvec[0]),0)
ax[1].set_title(r"$\log_{10}\left(\langle \mathcal{L} \rangle -\langle \mathcal{L} \rangle_{\textrm{max}} \right)$") 

# ax[1].set_aspect('equal')
fig.tight_layout()
# fig.savefig("surface.pdf",format='pdf',dpi=500,bbox_inches='tight')
print(str(np.max(Lsurface[Lsurface!=0.])))

# -

Lsurface

svec=np.load(outpath+'svec.npy')


Ps_type='offcent_gauss'
# plot_pair('v1_ct_1_mt_2_st_cent_gauss_min0_maxinf',0,4,pl)
plot_pair('v1_ct_1_mt_2_st_offcent_gauss_min0_maxinf',15,0,pl)
npoints=20
alpvec=np.logspace(-6.,0., npoints)
sbarvec_p=np.logspace(-2,1,npoints)
sbarvec_m=np.linspace(0.1,15,10)
bet=0
fig,ax=pl.subplots(1,1,figsize=(8,8))
smax=25
s_step=0.1
# for alp in alpvec:
alp=0.01
# for sbar_p in sbarvec_p:
sbar_p=1.0
svec=np.load(outpath+'svec.npy')
for sbar_m in sbarvec_m:
    ax.plot(svec,np.exp(get_logPs_pm(alp,bet,sbar_m,sbar_p,smax,s_step,Ps_type)))
ax.set_yscale('log')
ax.set_ylim(1e-10,1e1)

# + {"code_folding": []}
output_path='../../../output/'
donorstr='S2'
null_pair=donorstr+'_0_F1_'+donorstr+'_0_F2'
diff_pair=donorstr+'_0_F1_'+donorstr+'_0_F2'

run_name='diffexpr_pair_'+null_pair+'_v1_ct_1_mt_2_st_rhs_only_min0_maxinf'
outpath=output_path+diff_pair+'/'+run_name+'/'

#Assemble grid
alpvec=np.load(outpath+'alpvec.npy')
sbarvec_p=np.load(outpath+'sbarvec_p.npy')
# sbarvec_p=np.linspace(0.1,5.,20)
LSurfacestore=np.zeros((len(alpvec),len(sbarvec_p)))
nit_list=np.zeros((len(alpvec),len(sbarvec_p)))
shiftMtr=np.zeros((len(alpvec),len(sbarvec_p)))
Zstore=np.zeros((len(alpvec),len(sbarvec_p)))
Zdashstore=np.zeros((len(alpvec),len(sbarvec_p)))
time_elapsed=np.zeros((len(alpvec),len(sbarvec_p)))

for bit,bet in enumerate(sbarvec_p):
    dim=(slice(None),bit)
    try:
        LSurfacestore[dim]=np.load(outpath+'Lsurface'+str(bit)+'.npy')
        nit_list[dim]=np.load(outpath+'nit_list'+str(bit)+'.npy')
        shiftMtr[dim]=np.load(outpath+'shift'+str(bit)+'.npy')
        Zstore[dim]=np.load(outpath+'Zstore'+str(bit)+'.npy')
        Zdashstore[dim]=np.load(outpath+'Zdashstore'+str(bit)+'.npy')
        time_elapsed[dim]=np.load(outpath+'time_elapsed'+str(bit)+'.npy')/3600.
    except:
        print(bit)
LSurfacestore=LSurfacestore
nit_list=nit_list
shiftMtr=shiftMtr
Zstore=Zstore
Zdashstore=Zdashstore
time_elapsed=time_elapsed


#load data
Lsurface=LSurfacestore[:,4:]
minval=-1.95
Lsurface[Lsurface==0]=minval#np.min(np.min(Lsurface))
Lsurface[Lsurface<minval]=minval#np.min(np.min(Lsurface))

alpvec=alpvec
sbarvec=np.log10(sbarvec_p[4:])
X, Y = np.meshgrid(sbarvec,np.log10(alpvec))

fig,ax=pl.subplots(1,2,figsize=(20,10))

maxL=np.max(np.max(Lsurface))
p=ax[0].imshow(Lsurface,extent=[sbarvec[0], sbarvec[-1],np.log10(alpvec[0]), np.log10(alpvec[-1])], aspect='auto',origin='lower',interpolation='none')#,cmap='viridis')
ax[0].contour(X, Y, Lsurface,levels = 5,colors=('w',),linestyles=('--',),linewidths=(5,))
pl.colorbar(p,ax=ax[0])
ax[0].set_ylabel(r'$\alpha$')
ax[0].set_xlabel(r'$\bar{s}$');
ax[0].set_ylim(np.log10(alpvec[0]),0)
ax[0].set_title(r"$\langle \mathcal{L} \rangle$")
# ax[0].set_aspect('equal')

fac=np.inf
fac=1+1e-2
Lsurface[Lsurface<fac*maxL]=fac*maxL
p2=ax[1].imshow(np.log10(-(Lsurface-maxL)),extent=[sbarvec[0], sbarvec[-1],np.log10(alpvec[0]), np.log10(alpvec[-1])], aspect='auto',origin='lower',interpolation='none')#,cmap='viridis')
pl.colorbar(p2,ax=ax[1])
ax[1].set_ylabel(r'$\alpha$')
ax[1].set_xlabel(r'$\bar{s}$');
ax[1].set_ylim(np.log10(alpvec[0]),0)
ax[1].set_title(r"$\log_{10}\left(\langle \mathcal{L} \rangle -\langle \mathcal{L} \rangle_{\textrm{max}} \right)$") 

# ax[1].set_aspect('equal')
fig.tight_layout()
# fig.savefig("surface.pdf",format='pdf',dpi=500,bbox_inches='tight')
print(str(np.max(Lsurface[Lsurface!=0.])))


# +
output_path='../../../output/'
donorstr='S2'
null_pair=donorstr+'_0_F1_'+donorstr+'_0_F2'
diff_pair=donorstr+'_0_F1_'+donorstr+'_15_F2'

run_name='diffexpr_pair_'+null_pair+'_v1_ct_1_mt_2_st_rhs_only_min0_maxinf'
outpath=output_path+diff_pair+'/'+run_name+'/'

#Assemble grid
alpvec=np.load(outpath+'alpvec.npy')
sbarvec_p=np.load(outpath+'sbarvec_p.npy')
# sbarvec_p=np.linspace(0.1,5.,20)
LSurfacestore=np.zeros((len(alpvec),len(sbarvec_p)))
nit_list=np.zeros((len(alpvec),len(sbarvec_p)))
shiftMtr=np.zeros((len(alpvec),len(sbarvec_p)))
Zstore=np.zeros((len(alpvec),len(sbarvec_p)))
Zdashstore=np.zeros((len(alpvec),len(sbarvec_p)))
time_elapsed=np.zeros((len(alpvec),len(sbarvec_p)))

for bit,bet in enumerate(sbarvec_p):
    dim=(slice(None),bit)
    try:
        LSurfacestore[dim]=np.load(outpath+'Lsurface'+str(bit)+'.npy')
        nit_list[dim]=np.load(outpath+'nit_list'+str(bit)+'.npy')
        shiftMtr[dim]=np.load(outpath+'shift'+str(bit)+'.npy')
        Zstore[dim]=np.load(outpath+'Zstore'+str(bit)+'.npy')
        Zdashstore[dim]=np.load(outpath+'Zdashstore'+str(bit)+'.npy')
        time_elapsed[dim]=np.load(outpath+'time_elapsed'+str(bit)+'.npy')/3600.
    except:
        print(bit)
LSurfacestore=LSurfacestore
nit_list=nit_list
shiftMtr=shiftMtr
Zstore=Zstore
Zdashstore=Zdashstore
time_elapsed=time_elapsed


#load data
Lsurface=LSurfacestore[:,4:]
minval=-1.95
Lsurface[Lsurface==0]=minval#np.min(np.min(Lsurface))
Lsurface[Lsurface<minval]=minval#np.min(np.min(Lsurface))

alpvec=alpvec
sbarvec=np.log10(sbarvec_p[4:])
X, Y = np.meshgrid(sbarvec,np.log10(alpvec))

fig,ax=pl.subplots(1,2,figsize=(20,10))

maxL=np.max(np.max(Lsurface))
p=ax[0].imshow(Lsurface,extent=[sbarvec[0], sbarvec[-1],np.log10(alpvec[0]), np.log10(alpvec[-1])], aspect='auto',origin='lower',interpolation='none')#,cmap='viridis')
ax[0].contour(X, Y, Lsurface,levels = 5,colors=('w',),linestyles=('--',),linewidths=(5,))
pl.colorbar(p,ax=ax[0])
ax[0].set_ylabel(r'$\alpha$')
ax[0].set_xlabel(r'$\bar{s}$');
ax[0].set_ylim(np.log10(alpvec[0]),0)
ax[0].set_title(r"$\langle \mathcal{L} \rangle$")
# ax[0].set_aspect('equal')

fac=np.inf
fac=1+1e-2
Lsurface[Lsurface<fac*maxL]=fac*maxL
p2=ax[1].imshow(np.log10(-(Lsurface-maxL)),extent=[sbarvec[0], sbarvec[-1],np.log10(alpvec[0]), np.log10(alpvec[-1])], aspect='auto',origin='lower',interpolation='none')#,cmap='viridis')
pl.colorbar(p2,ax=ax[1])
ax[1].set_ylabel(r'$\alpha$')
ax[1].set_xlabel(r'$\bar{s}$');
ax[1].set_ylim(np.log10(alpvec[0]),0)
ax[1].set_title(r"$\log_{10}\left(\langle \mathcal{L} \rangle -\langle \mathcal{L} \rangle_{\textrm{max}} \right)$") 

# ax[1].set_aspect('equal')
fig.tight_layout()
# fig.savefig("surface.pdf",format='pdf',dpi=500,bbox_inches='tight')
print(str(np.max(Lsurface[Lsurface!=0.])))


# +
output_path='../../../output/'
donorstr='S2'
null_pair=donorstr+'_0_F1_'+donorstr+'_0_F2'
diff_pair=donorstr+'_0_F1_'+donorstr+'_15_F2'

run_name='diffexpr_pair_'+null_pair+'_v1_ct_1_mt_2_st_rhs_only_min0_maxinf'
outpath=output_path+diff_pair+'/'+run_name+'/'

#Assemble grid
alpvec=np.load(outpath+'alpvec.npy')
sbarvec_p=np.load(outpath+'sbarvec_p.npy')
# sbarvec_p=np.linspace(0.1,5.,20)
LSurfacestore=np.zeros((len(alpvec),len(sbarvec_p)))
nit_list=np.zeros((len(alpvec),len(sbarvec_p)))
shiftMtr=np.zeros((len(alpvec),len(sbarvec_p)))
Zstore=np.zeros((len(alpvec),len(sbarvec_p)))
Zdashstore=np.zeros((len(alpvec),len(sbarvec_p)))
time_elapsed=np.zeros((len(alpvec),len(sbarvec_p)))

for bit,bet in enumerate(sbarvec_p):
    dim=(slice(None),bit)
    try:
        LSurfacestore[dim]=np.load(outpath+'Lsurface'+str(bit)+'.npy')
        nit_list[dim]=np.load(outpath+'nit_list'+str(bit)+'.npy')
        shiftMtr[dim]=np.load(outpath+'shift'+str(bit)+'.npy')
        Zstore[dim]=np.load(outpath+'Zstore'+str(bit)+'.npy')
        Zdashstore[dim]=np.load(outpath+'Zdashstore'+str(bit)+'.npy')
        time_elapsed[dim]=np.load(outpath+'time_elapsed'+str(bit)+'.npy')/3600.
    except:
        print(bit)
LSurfacestore=LSurfacestore
nit_list=nit_list
shiftMtr=shiftMtr
Zstore=Zstore
Zdashstore=Zdashstore
time_elapsed=time_elapsed


#load data
Lsurface=LSurfacestore[:,4:]
minval=-1.95
Lsurface[Lsurface==0]=minval#np.min(np.min(Lsurface))
Lsurface[Lsurface<minval]=minval#np.min(np.min(Lsurface))

alpvec=alpvec
sbarvec=np.log10(sbarvec_p[4:])
X, Y = np.meshgrid(sbarvec,np.log10(alpvec))

fig,ax=pl.subplots(1,2,figsize=(20,10))

maxL=np.max(np.max(Lsurface))
p=ax[0].imshow(Lsurface,extent=[sbarvec[0], sbarvec[-1],np.log10(alpvec[0]), np.log10(alpvec[-1])], aspect='auto',origin='lower',interpolation='none')#,cmap='viridis')
ax[0].contour(X, Y, Lsurface,levels = 5,colors=('w',),linestyles=('--',),linewidths=(5,))
pl.colorbar(p,ax=ax[0])
ax[0].set_ylabel(r'$\alpha$')
ax[0].set_xlabel(r'$\bar{s}$');
ax[0].set_ylim(np.log10(alpvec[0]),0)
ax[0].set_title(r"$\langle \mathcal{L} \rangle$")
# ax[0].set_aspect('equal')

fac=np.inf
fac=1+1e-2
Lsurface[Lsurface<fac*maxL]=fac*maxL
p2=ax[1].imshow(np.log10(-(Lsurface-maxL)),extent=[sbarvec[0], sbarvec[-1],np.log10(alpvec[0]), np.log10(alpvec[-1])], aspect='auto',origin='lower',interpolation='none')#,cmap='viridis')
pl.colorbar(p2,ax=ax[1])
ax[1].set_ylabel(r'$\alpha$')
ax[1].set_xlabel(r'$\bar{s}$');
ax[1].set_ylim(np.log10(alpvec[0]),0)
ax[1].set_title(r"$\log_{10}\left(\langle \mathcal{L} \rangle -\langle \mathcal{L} \rangle_{\textrm{max}} \right)$") 

# ax[1].set_aspect('equal')
fig.tight_layout()
# fig.savefig("surface.pdf",format='pdf',dpi=500,bbox_inches='tight')
print(str(np.max(Lsurface[Lsurface!=0.])))


# + {"code_folding": []}
#load data
Lsurface=LSurfacestore
sbarvec=np.log10(sbarvec_p)
# sbarvec=sbarvec_p
fig,ax=pl.subplots(1,1,figsize=(10,10))
# Lsurface[:,0]=np.min(np.min(Lsurface))
maxL=np.max(np.max(Lsurface))
Lmin=-2.2
Lsurface[Lsurface<Lmin]=Lmin
p=ax.imshow(np.log10(-(Lsurface-maxL)),extent=[sbarvec[0], sbarvec[-1],np.log10(alpvec[0]), np.log10(alpvec[-1])], aspect='equal',origin='lower',interpolation='none')#,cmap='viridis')
X, Y = np.meshgrid(sbarvec,np.log10(alpvec))
# ax.set_xscale('log')
# ax.contour(X, Y, np.log10(-(Lsurface-maxL)),levels = 20,colors=('w',),linestyles=('--',),linewidths=(5,))
ax.set_title(donorstr+r' $-L=-\sum_i \log P(n_i,n\prime_i)$')#' \bar{s}_{opt}='+str(np.load(path+'optsbar.npy').flatten()[0])+r' \log\alpha_{opt}='+str(np.log10(np.load(path+'optalp.npy').flatten()[0]))+r'$')
print(str(np.max(Lsurface[Lsurface!=0.])))

# ax.scatter(np.load(path+'optsbar.npy').flatten()[0],np.log10(np.load(path+'optalp.npy').flatten()[0]),marker='o',c='w',s=500)
# optinds=np.unravel_index(np.argmin(np.log10(-Lsurface)),(len(sbarvec),len(alpvec)))
# print(optinds)
# ax.scatter(sbarvec[optinds[0]],np.log10(alpvec[optinds[1]]),marker='o',c='k',s=300)
pl.colorbar(p)

ax.set_ylabel(r'$\alpha$')
ax.set_xlabel(r'$\bar{s}$');
# ax.set_xlim(0,sbarvec[-1])
ax.set_ylim(np.log10(alpvec[0]),0)
ax.set_aspect('equal')
# fig.savefig("surface.pdf",format='pdf',dpi=500,bbox_inches='tight')
# -

fig,ax=pl.subplots(1,2,figsize=(15,15))
ax[0].plot(range(10))
ax[1].plot(range(10))
ax[0].set_aspect('equal')
ax[1].set_aspect('equal')



c

# +
output_path='../../../output/'
donorstr='S2'
null_pair=donorstr+'_0_F1_'+donorstr+'_0_F2'
diff_pair=donorstr+'_0_F1_'+donorstr+'_15_F2'

run_name='diffexpr_pair_'+null_pair+'_v1_ct_1_mt_2_st_cent_gauss_min0_maxinf'
outpath=output_path+diff_pair+'/'+run_name+'/'

#Assemble grid
alpvec=np.load(outpath+'alpvec.npy')
sbarvec_p=np.load(outpath+'sbarvec_p.npy')
# sbarvec_p=np.linspace(0.1,5.,20)
LSurfacestore=np.zeros((len(alpvec),len(sbarvec_p)))
nit_list=np.zeros((len(alpvec),len(sbarvec_p)))
shiftMtr=np.zeros((len(alpvec),len(sbarvec_p)))
Zstore=np.zeros((len(alpvec),len(sbarvec_p)))
Zdashstore=np.zeros((len(alpvec),len(sbarvec_p)))
time_elapsed=np.zeros((len(alpvec),len(sbarvec_p)))

for bit,bet in enumerate(sbarvec_p):
    dim=(slice(None),bit)
    try:
        LSurfacestore[dim]=np.load(outpath+'Lsurface'+str(bit)+'.npy')
        nit_list[dim]=np.load(outpath+'nit_list'+str(bit)+'.npy')
        shiftMtr[dim]=np.load(outpath+'shift'+str(bit)+'.npy')
        Zstore[dim]=np.load(outpath+'Zstore'+str(bit)+'.npy')
        Zdashstore[dim]=np.load(outpath+'Zdashstore'+str(bit)+'.npy')
        time_elapsed[dim]=np.load(outpath+'time_elapsed'+str(bit)+'.npy')/3600.
    except:
        print(bit)
LSurfacestore=LSurfacestore
nit_list=nit_list
shiftMtr=shiftMtr
Zstore=Zstore
Zdashstore=Zdashstore
time_elapsed=time_elapsed


#load data
Lsurface=LSurfacestore#[:,6:]
minval=-1.95
Lsurface[Lsurface==0]=minval#np.min(np.min(Lsurface))
Lsurface[Lsurface<minval]=minval#np.min(np.min(Lsurface))

alpvec=alpvec
sbarvec=np.log10(sbarvec_p)#[6:])#[4:])
fig,ax=pl.subplots(1,1,figsize=(10,10))
maxL=np.max(np.max(Lsurface))
fac=np.inf
# fac=1+1e-2
# Lsurface[Lsurface<fac*maxL]=fac*maxL
# p=ax.imshow(np.log10(-(Lsurface-maxL)),extent=[sbarvec[0], sbarvec[-1],np.log10(alpvec[0]), np.log10(alpvec[-1])], aspect='equal',origin='lower',interpolation='none')#,cmap='viridis')
p=ax.imshow(Lsurface,extent=[sbarvec[0], sbarvec[-1],np.log10(alpvec[0]), np.log10(alpvec[-1])], aspect='equal',origin='lower',interpolation='none')#,cmap='viridis')
X, Y = np.meshgrid(sbarvec,np.log10(alpvec))
# ax.set_xscale('log')
ax.contour(X, Y, Lsurface,levels = 5,colors=('w',),linestyles=('--',),linewidths=(5,))
ax.set_title(donorstr+r' $-L=-\sum_i \log P(n_i,n\prime_i)$')#' \bar{s}_{opt}='+str(np.load(path+'optsbar.npy').flatten()[0])+r' \log\alpha_{opt}='+str(np.log10(np.load(path+'optalp.npy').flatten()[0]))+r'$')
print(str(np.max(Lsurface[Lsurface!=0.])))

# ax.scatter(np.load(path+'optsbar.npy').flatten()[0],np.log10(np.load(path+'optalp.npy').flatten()[0]),marker='o',c='w',s=500)
# optinds=np.unravel_index(np.argmin(np.log10(-Lsurface)),(len(sbarvec),len(alpvec)))
# print(optinds)
# ax.scatter(sbarvec[optinds[0]],np.log10(alpvec[optinds[1]]),marker='o',c='k',s=300)
pl.colorbar(p)

ax.set_ylabel(r'$\alpha$')
ax.set_xlabel(r'$\bar{s}$');
# ax.set_xlim(0,sbarvec[-1])
ax.set_ylim(np.log10(alpvec[0]),0)
ax.set_aspect('equal')
# fig.savefig("surface.pdf",format='pdf',dpi=500,bbox_inches='tight')

# +
output_path='../../../output/'
donorstr='S2'
null_pair=donorstr+'_0_F1_'+donorstr+'_0_F2'
diff_pair=donorstr+'_0_F1_'+donorstr+'_15_F2'

run_name='diffexpr_pair_'+null_pair+'_v1_ct_1_mt_2_st_cent_gauss_min0_maxinf'
outpath=output_path+diff_pair+'/'+run_name+'/'

#Assemble grid
alpvec=np.load(outpath+'alpvec.npy')
sbarvec_p=np.load(outpath+'sbarvec_p.npy')
# sbarvec_p=np.linspace(0.1,5.,20)
LSurfacestore=np.zeros((len(alpvec),len(sbarvec_p)))
nit_list=np.zeros((len(alpvec),len(sbarvec_p)))
shiftMtr=np.zeros((len(alpvec),len(sbarvec_p)))
Zstore=np.zeros((len(alpvec),len(sbarvec_p)))
Zdashstore=np.zeros((len(alpvec),len(sbarvec_p)))
time_elapsed=np.zeros((len(alpvec),len(sbarvec_p)))

for bit,bet in enumerate(sbarvec_p):
    dim=(slice(None),bit)
    try:
        LSurfacestore[dim]=np.load(outpath+'Lsurface'+str(bit)+'.npy')
        nit_list[dim]=np.load(outpath+'nit_list'+str(bit)+'.npy')
        shiftMtr[dim]=np.load(outpath+'shift'+str(bit)+'.npy')
        Zstore[dim]=np.load(outpath+'Zstore'+str(bit)+'.npy')
        Zdashstore[dim]=np.load(outpath+'Zdashstore'+str(bit)+'.npy')
        time_elapsed[dim]=np.load(outpath+'time_elapsed'+str(bit)+'.npy')/3600.
    except:
        print(bit)
LSurfacestore=LSurfacestore
nit_list=nit_list
shiftMtr=shiftMtr
Zstore=Zstore
Zdashstore=Zdashstore
time_elapsed=time_elapsed


#load data
Lsurface=LSurfacestore[:,6:]
alpvec=alpvec
sbarvec=np.log10(sbarvec_p[6:])#[4:])
fig,ax=pl.subplots(1,1,figsize=(10,10))
maxL=np.max(np.max(Lsurface[Lsurface!=0]))
fac=np.inf
# fac=1+1e-2
Lsurface[Lsurface<fac*maxL]=fac*maxL
p=ax.imshow(np.log10(-(Lsurface-maxL)),extent=[sbarvec[0], sbarvec[-1],np.log10(alpvec[0]), np.log10(alpvec[-1])], aspect='equal',origin='lower',interpolation='none')#,cmap='viridis')
X, Y = np.meshgrid(sbarvec,np.log10(alpvec))
# ax.set_xscale('log')
# ax.contour(X, Y, np.log10(-(Lsurface-maxL)),levels = 20,colors=('w',),linestyles=('--',),linewidths=(5,))
ax.set_title(donorstr+r' $-L=-\sum_i \log P(n_i,n\prime_i)$')#' \bar{s}_{opt}='+str(np.load(path+'optsbar.npy').flatten()[0])+r' \log\alpha_{opt}='+str(np.log10(np.load(path+'optalp.npy').flatten()[0]))+r'$')
print(str(np.max(Lsurface[Lsurface!=0.])))

# ax.scatter(np.load(path+'optsbar.npy').flatten()[0],np.log10(np.load(path+'optalp.npy').flatten()[0]),marker='o',c='w',s=500)
# optinds=np.unravel_index(np.argmin(np.log10(-Lsurface)),(len(sbarvec),len(alpvec)))
# print(optinds)
# ax.scatter(sbarvec[optinds[0]],np.log10(alpvec[optinds[1]]),marker='o',c='k',s=300)
pl.colorbar(p)

ax.set_ylabel(r'$\alpha$')
ax.set_xlabel(r'$\bar{s}$');
# ax.set_xlim(0,sbarvec[-1])
ax.set_ylim(np.log10(alpvec[0]),0)
ax.set_aspect('equal')
# fig.savefig("surface.pdf",format='pdf',dpi=500,bbox_inches='tight')

# +
#load data
Lsurface=LSurfacestore[:20,:]
sbarvec=np.log10(sbarvec_p[:20])
alpvec=np.log10(alpvec[:20])
# sbarvec=sbarvec_p
fig,ax=pl.subplots(1,1,figsize=(10,10))
# Lsurface[:,0]=np.min(np.min(Lsurface))
maxL=np.max(np.max(Lsurface))
Lmin=-2.2
Lsurface[Lsurface<Lmin]=Lmin
p=ax.imshow(np.log10(-(Lsurface-maxL)),extent=[sbarvec[0], sbarvec[-1],np.log10(alpvec[0]), np.log10(alpvec[-1])], aspect='equal',origin='lower',interpolation='none')#,cmap='viridis')
X, Y = np.meshgrid(sbarvec,np.log10(alpvec))
# ax.set_xscale('log')
# ax.contour(X, Y, np.log10(-(Lsurface-maxL)),levels = 20,colors=('w',),linestyles=('--',),linewidths=(5,))
ax.set_title(donorstr+r' $-L=-\sum_i \log P(n_i,n\prime_i)$')#' \bar{s}_{opt}='+str(np.load(path+'optsbar.npy').flatten()[0])+r' \log\alpha_{opt}='+str(np.log10(np.load(path+'optalp.npy').flatten()[0]))+r'$')
print(str(np.max(Lsurface[Lsurface!=0.])))

# ax.scatter(np.load(path+'optsbar.npy').flatten()[0],np.log10(np.load(path+'optalp.npy').flatten()[0]),marker='o',c='w',s=500)
# optinds=np.unravel_index(np.argmin(np.log10(-Lsurface)),(len(sbarvec),len(alpvec)))
# print(optinds)
# ax.scatter(sbarvec[optinds[0]],np.log10(alpvec[optinds[1]]),marker='o',c='k',s=300)
pl.colorbar(p)

ax.set_ylabel(r'$\alpha$')
ax.set_xlabel(r'$\bar{s}$');
# ax.set_xlim(0,sbarvec[-1])
ax.set_ylim(np.log10(alpvec[0]),0)
ax.set_aspect('equal')
# fig.savefig("surface.pdf",format='pdf',dpi=500,bbox_inches='tight')

# +
#load data
Lsurface=LSurfacestore
sbarvec=np.log10(sbarvec_p)
# sbarvec=sbarvec_p
fig,ax=pl.subplots(1,1,figsize=(10,10))
# Lsurface[:,0]=np.min(np.min(Lsurface))
maxL=np.max(np.max(Lsurface))
Lmin=-2.2
Lsurface[Lsurface<Lmin]=Lmin
p=ax.imshow(np.log10(-(Lsurface-maxL)),extent=[sbarvec[0], sbarvec[-1],np.log10(alpvec[0]), np.log10(alpvec[-1])], aspect='equal',origin='lower',interpolation='none')#,cmap='viridis')
X, Y = np.meshgrid(sbarvec,np.log10(alpvec))
# ax.set_xscale('log')
# ax.contour(X, Y, np.log10(-(Lsurface-maxL)),levels = 20,colors=('w',),linestyles=('--',),linewidths=(5,))
ax.set_title(donorstr+r' $-L=-\sum_i \log P(n_i,n\prime_i)$')#' \bar{s}_{opt}='+str(np.load(path+'optsbar.npy').flatten()[0])+r' \log\alpha_{opt}='+str(np.log10(np.load(path+'optalp.npy').flatten()[0]))+r'$')
print(str(np.max(Lsurface[Lsurface!=0.])))

# ax.scatter(np.load(path+'optsbar.npy').flatten()[0],np.log10(np.load(path+'optalp.npy').flatten()[0]),marker='o',c='w',s=500)
# optinds=np.unravel_index(np.argmin(np.log10(-Lsurface)),(len(sbarvec),len(alpvec)))
# print(optinds)
# ax.scatter(sbarvec[optinds[0]],np.log10(alpvec[optinds[1]]),marker='o',c='k',s=300)
pl.colorbar(p)

ax.set_ylabel(r'$\alpha$')
ax.set_xlabel(r'$\bar{s}$');
# ax.set_xlim(0,sbarvec[-1])
ax.set_ylim(np.log10(alpvec[0]),0)
ax.set_aspect('equal')
# fig.savefig("surface.pdf",format='pdf',dpi=500,bbox_inches='tight')
# -

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
