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
from lib.model import get_logPs_pm,get_rhof,get_fvec_and_svec,get_logPn_f
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

donorstr='S2'
output_path='../../../output/'

runname='v4_ct_1_mt_2_min0_maxinf'
null_pair=donorstr+'_0_F1_'+donorstr+'_0_F2'
run_name=runname
outpath=output_path+null_pair+'/null_pair_'+run_name+'/'
null_paras= np.load(outpath+'optparas.npy')

# +
smax=25.
s_step=0.1
acq_model_type=2
logrhofvec,logfvec = get_rhof(null_paras[0],np.power(10,null_paras[-1]))
svec,logfvec,logfvecwide,f2s_step,smax,s_step=get_fvec_and_svec(null_paras,s_step,smax)
np.save(outpath+'svec.npy',svec)

sparse_rep=np.load(outpath+'sparserep.npy').item()
indn1,indn2,sparse_rep_counts,unicountvals_1,unicountvals_2,NreadsI,NreadsII=sparse_rep.values()
logPn1_f=get_logPn_f(unicountvals_1,NreadsI,logfvec,acq_model_type,null_paras)
# -

Ps_type='rhs_only'
Ps_type='sym_exp'
day=15
startind=0
runname='v4_ct_1_mt_2_st_'+Ps_type+'_min0_maxinf'
# Ps_n1n2ps=get_posteriors(donorstr,'v4_ct_1_mt_2_st_'+Ps_type+'_min0_maxinf',day,startind,pl)
diff_pair=donorstr+'_0_F1_'+donorstr+'_'+str(day)+'_F2'
run_name='diffexpr_pair_'+null_pair+'_'+runname
outpath=output_path+diff_pair+'/'+run_name+'/'
opt_diffexpr_paras=np.load(outpath+'opt_diffexpr_paras.npy')

# +
logPsvec = get_logPs_pm(opt_diffexpr_paras[:-1],smax,s_step,Ps_type)
shift=opt_diffexpr_paras[-1]

fvecwide_shift=np.exp(logfvecwide-shift) #implements shift in Pn2_fs
svec_shift=svec-shift
logPn2_f=get_logPn_f(unicountvals_2,NreadsII,np.log(fvecwide_shift),acq_model_type,null_paras)
dlogfby2=np.diff(logfvec)/2

# +
logPn1n2_s=np.zeros((len(svec),len(sparse_rep_counts)))
for s_it in range(len(svec)):
    integ=np.exp(logPn1_f[:,indn1] + logPn2_f[f2s_step*s_it:f2s_step*s_it+len(logfvec),indn2] + logrhofvec[:,np.newaxis] + logfvec[:,np.newaxis])
    logPn1n2_s[s_it,:]=np.log(np.sum(dlogfby2[:,np.newaxis]*(integ[1:,:]+integ[:-1,:]),axis=0)) #can't use dot since multidimensional

Psn1n2_ps=np.exp(logPn1n2_s+logPsvec[:,np.newaxis])  
Pn1n2_ps=np.sum(Psn1n2_ps,0)
Ps_n1n2ps=np.exp(logPn1n2_s+logPsvec[:,np.newaxis])/Pn1n2_ps[np.newaxis,:]
# -

# smed versus snaive

n1=unicountvals_1[indn1]
n2=unicountvals_2[indn2]
s_naive=np.where(np.logical_and(n1>0,n2>0),np.log(n2/n1),0)

n1_n2_to_n1n2=np.zeros((len(unicountvals_1),len(unicountvals_2)),dtype=int)
for n1it,n1 in enumerate(unicountvals_1):
    for n2it,n2 in enumerate(unicountvals_2):
        ind=np.where(np.logical_and(n1==unicountvals_1[indn1],n2==unicountvals_2[indn2]))[0]
        if not ind:
            continue
        else:
            n1_n2_to_n1n2[n1it,n2it]=int(ind)

#compute posterior metrics
mean_est=np.zeros((len(n1),))
max_est= np.zeros((len(n1),))
slowvec= np.zeros((len(n1),))
smedvec= np.zeros((len(n1),))
shighvec=np.zeros((len(n1),))
pval=0.025 #double-sided comparison statistical test
pvalvec=[pval,0.5,1-pval] #bound criteria defining slow, smed, and shigh, respectively
for it,column in enumerate(np.transpose(Ps_n1n2ps)):
    mean_est[it]=np.sum(svec*column)
    max_est[it]=svec[np.argmax(column)]
    forwardcmf=np.cumsum(column)
    backwardcmf=np.cumsum(column[::-1])[::-1]
    inds=np.where((forwardcmf[:-1]<pvalvec[0]) & (forwardcmf[1:]>=pvalvec[0]))[0]
    slowvec[it]=np.mean(svec[inds+np.ones((len(inds),),dtype=int)])  #use mean in case there are two values
    inds=np.where((forwardcmf>=pvalvec[1]) & (backwardcmf>=pvalvec[1]))[0]
    smedvec[it]=np.mean(svec[inds])
    inds=np.where((forwardcmf[:-1]<pvalvec[2]) & (forwardcmf[1:]>=pvalvec[2]))[0]
    shighvec[it]=np.mean(svec[inds+np.ones((len(inds),),dtype=int)])

fig,ax=pl.subplots(1,1)
ax.scatter(smedvec[s_naive!=0],s_naive[s_naive!=0],s=(n1[s_naive!=0]+n2[s_naive!=0])/5,alpha=0.2, c=sparse_rep_counts[s_naive!=0], cmap='viridis')
limits=np.asarray(ax.get_xlim())
ax.plot(limits,limits,'k--')
ax.set_ylim(1.05*limits)
ax.plot(limits,[0,0],'k')
ax.plot([0,0],limits,'k')
ax.set_xlim(1.05*limits)
ax.set_xlabel(r'$s_{\textrm{median}}$')
ax.set_ylabel(r'$s_{\textrm{naive}}=\ln \frac{n_2}{n_1}$')
fig.savefig("s_med_vs_snaive.pdf",format='pdf',dpi=500,bbox_inches='tight')

# Look at spread of posteriors

# +
#data
posterior_df=pd.DataFrame(columns={'posterior':pd.Series([],dtype=object)})
posterior_df['indn1']=indn1
posterior_df['indn2']=indn2
for ind_ind in range(len(indn1)):
    posterior_df.at[ind_ind,'posterior']=Ps_n1n2ps[:,ind_ind]
posterior_df['multiplicity']=sparse_rep_counts

#diag
# diag_posterior_df=pd.DataFrame(columns={'posterior':pd.Series([],dtype=object)})
# for n in range(100):
#     diag_posterior_df.at[n,'posterior']=Ps_n1n2ps[:,n1_n2_to_n1n2[n==unicountvals_1,n==unicountvals_2]]

#antidiag
antidiag_posterior_df=pd.DataFrame(columns={'posterior':pd.Series([],dtype=object)})
for n in range(100):
    antidiag_posterior_df.at[n,'posterior']=Ps_n1n2ps[:,n1_n2_to_n1n2[100-n==unicountvals_1,n==unicountvals_2]]

# +
n_min=0
n_max=np.Inf
n1=unicountvals_1[indn1]
n2=unicountvals_2[indn2]
both_nonzero=np.logical_and(np.logical_and(n1>n_min,n2>n_min),np.logical_and(n1<n_max,n2<n_max))
logn2n1=np.where(both_nonzero,np.log(n2/n1),0)
sort_inds=np.argsort(logn2n1)

naive_posterior_df=pd.DataFrame(columns={'posterior':pd.Series([],dtype=object)})
for n,ind in enumerate(sort_inds[::200]):
    naive_posterior_df.at[n,'posterior']=Ps_n1n2ps[:,sort_inds[::200][n]]


# +

# tmp_posts=posterior_df['posterior']
# tmp_posts=antidiag_posterior_df['posterior']
tmp_posts=naive_posterior_df['posterior']

colorsre = pl.cm.inferno(np.linspace(0.1, 1, len(tmp_posts.values)))

fig,ax=pl.subplots(1,1)#,figsize=(1.5,1.5))
for pit,postyrior in enumerate(tmp_posts.values):
    ax.plot(svec_shift,postyrior,color=colorsre[pit])
# avg_posterior=np.sum(Ps_n1n2ps*(sparse_rep_counts[np.newaxis,:]/np.sum(sparse_rep_counts)),axis=1)
ax.plot(svec_shift,avg_posterior,'k--')
ax.plot(svec,np.exp(logPsvec),'k-')
ax.set_yscale('log')
ax.set_ylim(1e-5,1e0)
# ax.plot(svec_shift,Ps,'k--',label='prior')
# ax.set_xlim(1e-1,30)
ax.set_xlim(-17,17)
# ax.set_xscale('log')
ax.set_xlabel(r'$s$')
ax.set_ylabel(r'$P(s)$')
# ax.legend(frameon=False)
ax.locator_params(axis='y',numticks=6)
# fig.tight_layout()
# fig.savefig("posterior_plot_a.pdf",format='pdf',dpi=500)#,bbox_inches='tight',pad_inches=0.1)
# -

# analysis includes: looking at range of posteriors, looking at distribution of summary statistics, making volcano plots and other such visual demosntrations of candidate idenficiation methods

strout='expanded'
table=pd.read_csv(outpath+diff_pair+'table_top_'+strout+'.csv',sep='\t')

table.head()

table.loc[:,[r'$s_{1,low}$', r'$s_{2,med}$', r'$s_{3,high}$', r'$s_{max}$', r'$\bar{s}$', r'$f_1$', r'$f_2$', r'$n_1$', r'$n_2$', r'$1-P(s>0)$']].hist(figsize=(16,16))
pl.gcf().tight_layout()

# Volcano plots

# +
smedianvec=table[r'$s_{2,med}$']
pvalplus=table[r'$1-P(s>0)$']

fig1, ax1 = pl.subplots(1,1)
# cond_pos=clones[u'${CDF}_{s=0}$']<0.025
# cond_neg=1-clones[u'${CDF}_{s=0}$']<0.025
# ax1.scatter(clones.loc[cond_pos,u'$s_{2,med}$'],clones.loc[cond_pos,u'${CDF}_{s=0}$'])
# ax1.scatter(clones.loc[cond_neg,u'$s_{2,med}$'],clones.loc[cond_neg,u'${CDF}_{s=0}$'])

cond_pos=(smedianvec>0) 
cond_neg=(smedianvec<=0)
ax1.scatter(smedianvec[cond_pos],-np.log10(pvalplus[cond_pos]),color='k')
ax1.scatter(smedianvec[cond_neg],-np.log10(1-pvalneg[cond_neg]),color='k')
# cond_pos=(smedianvec>0) & (pvalplus<0.025)
# cond_neg=(smedianvec<=0) & (pvalneg>0.975)
# ax1.scatter(smedianvec[cond_pos],-np.log10(pvalplus[cond_pos]),color='r')
# ax1.scatter(smedianvec[cond_neg],-np.log10(1-pvalneg[cond_neg]),color='r')
# for n in range(2):
#     cond_pos=(smedianvec>0) & (indn1==n) & (indn2!=0)
#     cond_neg=(smedianvec<=0) & (indn2==n) & (indn1!=0)
#     tmpos=unicountvals_2_d[indn2][cond_pos]
#     ax1.plot(smedianvec[cond_pos],-np.log10(pvalplus[cond_pos]),'k',linewidth=1)
#     ax1.scatter(smedianvec[cond_pos],-np.log10(pvalplus[cond_pos]),color='k')#,c=np.log10(tmpos/float(tmpos[-1])),cmap='gray_r' ) #colors=palette_pos)
#     tmpneg=unicountvals_1_d[indn1][cond_neg]
#     ax1.plot(smedianvec[cond_neg],-np.log10(1-pvalneg[cond_neg]),'k',linewidth=1 )
#     ax1.scatter(smedianvec[cond_neg],-np.log10(1-pvalneg[cond_neg]),color='k')#,c=-np.log10(tmpneg/float(tmpneg[-1])),cmap='gray' ) #colors=palette_neg)

# cond_pos=(smedianvec>0)
# cond_neg=(smedianvec<=0)
# tmp=np.argsort(unicountvals_1_d[indn1_d[cond_neg]]/np.asarray(unicountvals_2_d[indn2_d[[cond_neg]]],dtype=float))
# ax1.plot(smedianvec[cond_neg][tmp[:500]],-np.log10(1-pvalneg[cond_neg][tmp[:500]]),'k-')

# cond_pos=(smedianvec>0) & (indn1==indn1) & (indn2!=0)
# cond_neg=(smedianvec<=0) & (indn2==indn2) & (indn1!=0)
# # tmpos=unicountvals_2_d[indn2][cond_pos]
# ax1.plot(smedianvec[cond_pos],-np.log10(pvalplus[cond_pos]),'r')
# ax1.scatter(smedianvec[cond_pos],-np.log10(pvalplus[cond_pos]),c=np.log10(tmpos/float(tmpos[-1])),cmap='gray_r' ) #colors=palette_pos)
# tmpneg=unicountvals_1_d[indn1][cond_neg]
# ax1.plot(smedianvec[cond_neg],-np.log10(1-pvalneg[cond_neg]),'r' )
# ax1.scatter(smedianvec[cond_neg],-np.log10(1-pvalneg[cond_neg]),c=-np.log10(tmpneg/float(tmpneg[-1])),cmap='gray' ) #colors=palette_neg)

    
# ax1.set_xlim([-11.5,11.5])
ax1.set_ylim([0,11])
# ax1.plot(ax1.get_xlim(),[-np.log10(0.025),-np.log10(0.025)],'k--')
# ax1.plot([-s_step/2.,-s_step/2.],ax1.get_ylim(),'k--')
ax1.plot([0,0],ax1.get_ylim(),'k-')
# ax1.set_xlabel('expected log fold-change',fontsize=24)
# ax1.set_ylabel('Probability',fontsize=24)
# ax1.set_xlim([-0.5,0.5])
ax1.set_xlim([-11.5,11.5])
# fig1.savefig(path+donorstr+'_volcano_highlight.pdf',format='pdf',dpi=1000, bbox_inches='tight')

# -


