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

# +
import matplotlib.pyplot as pl
paper=True
if not paper:
    pl.rc("figure", facecolor="gray",figsize = (8,8))
    pl.rc('lines',markeredgewidth = 2)
    pl.rc('font',size = 24)
else:
    pl.rc("figure", facecolor="none",figsize = (3.5,3.5))
    pl.rc('lines',markeredgewidth = 1)
    pl.rc('font',size = 10)
    
pl.rc('text', usetex=True)

import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : True})
params= {'text.latex.preamble' : [r'\usepackage{amsmath}']}
pl.rcParams.update(params)
# -

# check data

Ps_type='sym_exp'
for donorstr in ['S1','S2','P1','P2','Q1','Q2']:
    for day in ['pre0','0','7','15','45']:
        for rep1 in ['1','2']:
            for rep2 in ['1','2']:
                runname='v4_ct_1_mt_2_st_'+Ps_type+'_min0_maxinf'
                output_path='../../../output/'
                null_pair=donorstr+'_0_F1_'+donorstr+'_0_F2'
                diff_pair=donorstr+'_0_F'+rep1+'_'+donorstr+'_'+str(day)+'_F'+rep2
                run_name='diffexpr_pair_'+null_pair+'_'+runname
                outpath=output_path+diff_pair+'/'+run_name+'/'
                check_vars=['Lsurface']
                for var in check_vars:
                    for it in range(20):
                        try:
                            np.load(outpath+var+str(it)+'.npy')
                        except:
                            print(donorstr+' '+day+' F'+rep1+' F'+rep2+' '+var+str(it))
                            print(outpath+var+str(it)+'.npy')

Ps_type='sym_exp'
for donorstr in ['S1','S2','P1','P2','Q1','Q2']:
    for day in ['pre0','0','7','15','45']:
         for rep1 in ['1','2']:
            for rep2 in ['1','2']:
                runname='v4_ct_1_mt_2_st_'+Ps_type+'_min0_maxinf'
                output_path='../../../output/'
                null_pair=donorstr+'_0_F1_'+donorstr+'_0_F2'
                diff_pair=donorstr+'_0_F'+rep1+'_'+donorstr+'_'+str(day)+'_F'+rep2
                run_name='diffexpr_pair_'+null_pair+'_'+runname
                outpath=output_path+diff_pair+'/'+run_name+'/'
#                 check_vars=['Lsurface']#['alpvec','Lsurface2','diffexpr_success','ellaxis1']
                check_vars=['diffexpr_success']
                for var in check_vars:
                    try:
                        np.load(outpath+var+'.npy')
                    except:
                        print(donorstr+' '+day+' '+var)

Ps_type='sym_exp'
for donorstr in ['S1','S2','P1','P2','Q1','Q2']:
    for day in ['pre0','0','7','15','45']:
        runname='v4_ct_1_mt_2_st_'+Ps_type+'_min0_maxinf'
        output_path='../../../output/'
        null_pair=donorstr+'_0_F1_'+donorstr+'_0_F2'
        diff_pair=donorstr+'_0_F1_'+donorstr+'_'+str(day)+'_F2'
        run_name='diffexpr_pair_'+null_pair+'_'+runname
        outpath=output_path+diff_pair+'/'+run_name+'/'
        check_vars=['Lsurface0']#=['diffexpr_success']
        for var in check_vars:
            try:
                np.load(outpath+var+'.npy')
            except:
                print(donorstr+' '+day+' '+var)

# # import data

np.log10(np.exp(25))

output_path='../../../output/'
    null_pair=donorstr+'_0_F1_'+donorstr+'_0_F2'
    diff_pair=donorstr+'_0_F1_'+donorstr+'_'+str(day)+'_F2'

    run_name='diffexpr_pair_'+null_pair+'_'+runname
    outpath=output_path+diff_pair+'/'+run_name+'/'



# + {"code_folding": []}
def plot_pair(donorstr,runname,day,rep1,rep2,startind,pl):
    output_path='../../../output/'
    null_pair=donorstr+'_0_F1_'+donorstr+'_0_F2'
    diff_pair=donorstr+'_0_F'+rep1+'_'+donorstr+'_'+str(day)+'_F'+rep2
    run_name='diffexpr_pair_'+null_pair+'_'+runname
    outpath=output_path+diff_pair+'/'+run_name+'/'

    #Assemble grid
    try:
        alpvec=np.load(outpath+'alpvec.npy')
        sbarvec_p=np.load(outpath+'sbarvec_p.npy')
        # sbarvec_p=np.linspace(0.1,5.,20)
        LSurfacestore=np.zeros((len(alpvec),len(sbarvec_p)))
        nit_list=np.zeros((len(alpvec),len(sbarvec_p)))
        shiftMtr=np.zeros((len(alpvec),len(sbarvec_p)))
        Zstore=np.zeros((len(alpvec),len(sbarvec_p)))
        Zdashstore=np.zeros((len(alpvec),len(sbarvec_p)))
        time_elapsed=np.zeros((len(alpvec),len(sbarvec_p)))
        
#         try:
#             ell_axis1=np.load(outpath+'ellaxis1.npy')
#             ell_axis2=np.load(outpath+'ellaxis2.npy')
#         except:
#             print(donorstr+' '+day)
            
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

        opt_inds=np.unravel_index(np.argmax(Lsurface),Lsurface.shape)
        diff_paras=[alpvec[opt_inds[0]],sbarvec_p[opt_inds[1]]]

        #handle 0 values
        minval=np.max(Lsurface)*1.1
        Lsurface[Lsurface==0]=minval
        Lsurface[Lsurface<minval]=minval

        alpvec=np.log10(alpvec)
        sbarvec=np.log10(sbarvec_p[startind:])
#         sbarvec=sbarvec_p[startind:]

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
            ax[it].contour(X, Y, Zstore,levels = [0.99,1.,1.01],colors=('k',),linestyles=('-',),linewidths=(5,))
            ax[it].contour(X, Y, Zstore,levels = [1.4],colors=('k',),linestyles=('--',),linewidths=(5,))
            ax[it].contour(X, Y, Zdashstore,levels = [0.99,1.,1.01],colors=('gray',),linestyles=('-',),linewidths=(5,))
            ax[it].contour(X, Y, Zdashstore,levels = [1.4],colors=('gray',),linestyles=('--',),linewidths=(5,))

            ax[it].set_ylabel(r'$\log_{10}\alpha$')
            ax[it].set_xlabel(r'$\log_{10}\bar{s}$')
            ax[it].set_title(titlestr)
        try:
            diff_paras=np.load(outpath+'diffexpr_outstruct.npy').item().x
        except:
            print(day)
        ax[1].scatter([np.log10(diff_paras[1])],[np.log10(diff_paras[0])]) 

        
#         x_axis=ell_axis2[:2]
#         y_axis=ell_axis1[:2]
#         print(x_axis)
#         theta = -np.degrees(np.arctan2(x_axis[0],x_axis[1]))#/np.linalg.norm(ell_axis1[:2]),ell_axis2[:2]/np.linalg.norm(ell_axis2[:2])))
#         ell = mpl.patches.Ellipse(xy=(0, 0),
#                       width=np.linalg.norm(x_axis), height=np.linalg.norm(y_axis),
#                       angle=theta, color='black',linewidth=2,linestyle='-',zorder=10)
#         ell.set_facecolor('None')
#         ax.add_artist(ell)

    #         print(diff_paras)
        fig.tight_layout()
        fig.suptitle([donorstr]+runname.split('_')+['0-'+str(day)],y=1.02)
        fig.savefig("surface_"+runname+"_"+str(day)+".pdf",format='pdf',dpi=500,bbox_inches='tight')
        Lsurface=LSurfacestore[:,startind:]
        print(str(np.max(Lsurface[Lsurface!=0.])))
        opt_inds=np.argmax(Lsurface)
        print('Z='+str(Zstore.flatten()[opt_inds])+' Z='+str(Zdashstore.flatten()[opt_inds]))
    except:
        print('no '+day)
        diff_paras=[]
    ell_axis1=[]
    ell_axis2=[]
    return outpath,diff_paras,ell_axis1,ell_axis2
# -

# By donor for day 15

import copy

outpath

import matplotlib as mpl

# +
Ps_type='sym_exp'
fig,ax=pl.subplots(1,1,figsize=(3,3))
donorstrvec=['S1','S2','P1','P2','Q1','Q2']
daystrvec=['pre0','0','7','15','45']
markervec=('o','s','v','^','d','P')
sns.set_context("paper",rc={"font.size":10})
pl.rc('font',size = 10)
handles = []
null_col =pl.rcParams['axes.prop_cycle'].by_key()['color']

    
for dit,donorstr in enumerate(donorstrvec):
    flag=True
    for yit,day in enumerate(daystrvec):
        for rep1 in ['1','2']:
            for rep2 in ['1','2']:
            #     day='15'
                startind=0
            #     donorstr='S2'
                runname='v4_ct_1_mt_2_st_'+Ps_type+'_min0_maxinf'
                output_path='../../../output/'
                null_pair=donorstr+'_0_F1_'+donorstr+'_0_F2'
                diff_pair=donorstr+'_0_F'+rep1+'_'+donorstr+'_'+str(day)+'_F'+rep2
                run_name='diffexpr_pair_'+null_pair+'_'+runname
                outpath=output_path+diff_pair+'/'+run_name+'/'
                try:
                    success=np.load(outpath+'diffexpr_success.npy').item()
                    if success:
                        diff_paras=np.load(outpath+'opt_diffexpr_paras.npy')#.item().x
                        if rep1=='1' and rep2=='2' and day=='15':
                            print(donorstr+' '+str(diff_paras[0]))
                        if flag:
                            flag=False
                            sc=ax.scatter([np.log10(diff_paras[1])],[np.log10(diff_paras[0])],facecolors='none',linewidth=1,marker=markervec[dit],s=50,color=null_col[yit],label=donorstrvec[dit])
                            handles.append(copy.copy(sc))
                        else:
                            ax.scatter([np.log10(diff_paras[1])],[np.log10(diff_paras[0])],facecolors='none',linewidth=1,marker=markervec[dit],s=50,color=null_col[yit])
                    else:
                        print(outpath+'fail')
                except:
                    print(outpath)
#                 outpath,diff_paras,ell1,ell2=plot_pair(donorstr,'v4_ct_1_mt_2_st_'+Ps_type+'_min0_maxinf',day,rep1,rep2,startind,pl)

#                 if dit==yit and rep1=='1' and rep2=='1':
#                     sc=ax.scatter([np.log10(diff_paras[1])],[np.log10(diff_paras[0])],facecolors='none',linewidth=1,marker=markervec[yit],s=50,color=null_col[dit],label=daystrvec[yit])
#                     handles.append(copy.copy(sc))
#                 else:
#                     ax.scatter([np.log10(diff_paras[1])],[np.log10(diff_paras[0])],facecolors='none',linewidth=1,marker=markervec[yit],s=50,color=null_col[dit])
        #         try:
        #             ell_axis1=np.load(outpath+'ellaxis1.npy')
        #             ell_axis2=np.load(outpath+'ellaxis2.npy')
        #             y_axis=ell_axis2[:2]
        #             x_axis=ell_axis1[:2]
        #             theta = np.degrees(np.arctan2(x_axis[0],x_axis[1]))#/np.linalg.norm(ell_axis1[:2]),ell_axis2[:2]/np.linalg.norm(ell_axis2[:2])))
        #             major_axis=np.linalg.norm(x_axis)
        #             minor_axis=np.linalg.norm(y_axis)
        #             print(minor_axis)
        #             print(major_axis)
        # #             ell = mpl.patches.Ellipse(xy=(np.log10(diff_paras[1]),np.log10(diff_paras[0])),
        # #                           width=2*major_axis, height=2*minor_axis,
        # #                           angle=theta, color='black',linewidth=2,linestyle='-',zorder=10)
        # #             ell.set_facecolor('None')
        # #             ax.add_artist(ell)
        #             data=[]
        #             for t in np.linspace(0,2*np.pi,500):
        #                 data.append((np.log10(diff_paras[1]+np.cos(t)*np.cos(theta*np.pi/180)*major_axis-np.sin(t)*np.sin(theta*np.pi/180)*minor_axis),\
        #                              np.log10(diff_paras[0]+np.cos(t)*np.sin(theta*np.pi/180)*major_axis+np.sin(t)*np.cos(theta*np.pi/180)*minor_axis)))
        #             x,y=zip(*data)
        #             ax.plot(x,y,'-',lw=1)
        #         except:
        #             print(donorstr+' '+day)

        #         ax.text(np.log10(diff_paras[1]),np.log10(diff_paras[0]),day)
        #         if ell1:
        #             ax.scatter([np.log10(diff_paras[1])],[np.log10(diff_paras[0])]) 
for h in handles:
    h.set_color("k")
ax.legend(handles=handles,frameon=False,fontsize=8,handletextpad=0.1)
ypos=np.linspace(-1.95,-1.05,6)[::-1]
for dit,daystr in enumerate(daystrvec):
    ax.text(-1, ypos[dit]-0.1,daystr,color= null_col[dit],weight='bold',fontsize=10)
ax.text(-1, -1.0,r'\underline{day}',color= 'k',fontsize=10)

ax.set_xticks((-1,0))
ax.set_yticks((-2,-1,0))
ax.set_xticklabels((r'$10^{-1}$',r'$10^0$'))
ax.set_yticklabels((r'$10^{-2}$',r'$10^{-1}$',r'$10^0$'))
ax.set_ylabel(r'$\alpha$')
ax.set_xlabel(r'$\bar{s}$')
ax.set_ylim(-2,0.1)
# -

# r stuff

S2=1750/1686798
S1=1454/1390584
P2=1469/2530847
P1=1377/1870000
Q1=668/862620
Q2=2356/1235018
for d in [S1,S2,P1,P2,Q1,Q2]:
    print(d)

np.load(outpath+'opt_diffexpr_paras.npy')

fig.savefig("sbar_alp_vals.pdf",format='pdf',dpi=500,bbox_inches='tight')

# by day (donor S2)

# +
Ps_type='sym_exp'
fig,ax=pl.subplots(1,1)
# for day in ['pre0','0','7','15','45']:
#     # day='45'
#     startind=0
#     donorstr='S2'
#     outpath,diff_paras=plot_pair(donorstr,'v4_ct_1_mt_2_st_'+Ps_type+'_min0_maxinf',day,startind,pl)
#     print(diff_paras)
#     ax.scatter([np.log10(diff_paras[1])],[np.log10(diff_paras[0])]) 
#     ax.text(np.log10(diff_paras[1]),np.log10(diff_paras[0]),day)
# #     ax.set_xlim(-2,1)
# #     ax.set_ylim(-2,0.1)
# for donorstr in ['S1','S2','P1','P2','Q1','Q2']:
#     day='15'
#     startind=0
# #     donorstr='S2'
#     outpath,diff_paras=plot_pair(donorstr,'v4_ct_1_mt_2_st_'+Ps_type+'_min0_maxinf',day,startind,pl)
#     ax.scatter([np.log10(diff_paras[1])],[np.log10(diff_paras[0])]) 
#     ax.text(np.log10(diff_paras[1]),np.log10(diff_paras[0]),donorstr)
    
for donorstr in ['S1','S2','P1','P2','Q1','Q2']:
    day='7'
    startind=0
#     donorstr='S2'
    outpath,diff_paras=plot_pair(donorstr,'v4_ct_1_mt_2_st_'+Ps_type+'_min0_maxinf',day,startind,pl)
    if len(diff_paras)>0:
        ax.scatter([np.log10(diff_paras[1])],[np.log10(diff_paras[0])]) 
        ax.text(np.log10(diff_paras[1]),np.log10(diff_paras[0]),donorstr)
ax.set_ylabel(r'$\log_{10}\alpha$')
ax.set_xlabel(r'$\log_{10}\bar{s}$')
# -

fig.savefig("sbar_alp_vals.pdf",format='pdf',dpi=500,bbox_inches='tight')

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

for Ps_type in ('rhs_only','cent_gauss'):
    for day in [0,15]:
        startind=4 if Ps_type=='rhs_only' else 6
        outpath=plot_pair('v1_ct_1_mt_2_st_'+Ps_type+'_min0_maxinf',day,startind,pl)

# # RHS only

# +
output_path='../../../output/'
donorstr='S2'
null_pair=donorstr+'_0_F1_'+donorstr+'_0_F2'

for testday in (0,15):
    diff_pair=donorstr+'_0_F1_'+donorstr+'_'+str(testday)+'_F2'
    v=1# if testday==0 else 2
    run_name='diffexpr_pair_'+null_pair+'_v'+str(v)+'_ct_1_mt_2_st_rhs_only_min0_maxinf'
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
    fig.suptitle(run_name.split('_')+['0-'+str(testday)],y=1.02)

    # ax[1].set_aspect('equal')
    fig.tight_layout()
    # fig.savefig("surface.pdf",format='pdf',dpi=500,bbox_inches='tight')
    print(str(np.max(Lsurface[Lsurface!=0.])))
    if testday==15:
        opt_paras=np.load(outpath+'svec.npy')
        opt_paras=np.load(outpath+'opt_diffexpr_paras.npy')
        print(opt_paras)
        ax[1].plot([np.log10(opt_paras[1])],[np.log10(opt_paras[0])],'ko',ms=10)

# -


# # Centered Gaussian

output_path='../../../output/'
donorstr='S2'
null_pair=donorstr+'_0_F1_'+donorstr+'_0_F2'
for testday in (0,15):
    diff_pair=donorstr+'_0_F1_'+donorstr+'_'+str(testday)+'_F2'

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
    # Lsurface=LSurfacestore
    Lsurface=LSurfacestore[:,6:]
    minval=-1.95
    Lsurface[Lsurface==0]=minval#np.min(np.min(Lsurface))
    Lsurface[Lsurface<minval]=minval#np.min(np.min(Lsurface))

    alpvec=alpvec
    sbarvec=np.log10(sbarvec_p[6:])
    # sbarvec=np.log10(sbarvec_p)
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
    fig.suptitle(run_name.split('_')+['0-'+str(testday)],y=1.02)

    if testday==15:
        opt_paras=np.load(outpath+'opt_diffexpr_paras.npy')
        struct=np.load(outpath+'diffexpr_outstruct.npy')
        print(struct)
#         print(opt_paras)
        ax[1].plot([np.log10(opt_paras[1])],[np.log10(opt_paras[0])],'ko',ms=10)
        opt_paras=np.load(outpath+'opt_diffexpr_paras_Nelder.npy')
#         print(opt_paras)
        struct=np.load(outpath+'diffexpr_outstruct_Nelder.npy')
        print(struct)
        
        ax[1].plot([np.log10(opt_paras[1])],[np.log10(opt_paras[0])],'kx',ms=10)

    # ax[1].set_aspect('equal')
    fig.tight_layout()
    # fig.savefig("surface.pdf",format='pdf',dpi=500,bbox_inches='tight')
    print(str(np.max(Lsurface[Lsurface!=0.])))

Ps_type='rhs_only'
day=15
startind=0
donorstr='P1'
outpath=plot_pair(donorstr,'v4_ct_1_mt_2_st_'+Ps_type+'_min0_maxinf',day,startind,pl)
pl.scatter([np.log10(0.64)],[np.log10(0.29)])

?
?
?
?
?
?
?
?
?
?
?
?
?
?
?
?
?
?
?
?
?



# Potentially useful code:

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
