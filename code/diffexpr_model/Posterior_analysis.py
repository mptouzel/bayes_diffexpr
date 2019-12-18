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
# %run -i '../lib/utils/ipynb_setup.py'
# import lib.utils.plotting
# from lib.utils.plotting import plot_n1_vs_n2,add_ticks
# from lib.utils.prob_utils import get_distsample
from lib.proc import get_sparserep,import_data, suffstats_table, get_sparse_ind
from lib.model import get_logPs_pm,get_rhof,get_fvec_and_svec,get_logPn_f
from lib.learning import get_diffexpr_likelihood,get_shift#constr_fn,callback,learn_null_model
# import lib.learning
# %load_ext autoreload
# %autoreload 2

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

day=15
mincount=0
maxcount=np.Inf
colnames1 = [u'Clone fraction',u'Clone count',u'N. Seq. CDR3',u'AA. Seq. CDR3']
colnames2 = [u'Clone fraction',u'Clone count',u'N. Seq. CDR3',u'AA. Seq. CDR3'] 
dataset_pair=(donorstr+'_0_F1',donorstr+'_'+str(day)+'_F2') 
input_data_path='../../../data/Yellow_fever/prepostvaccine/'
Nclones_samp,subset=import_data(input_data_path,dataset_pair[0]+'_.txt',dataset_pair[1]+'_.txt',mincount,maxcount,colnames1,colnames2)


sparse_rep_d=get_sparserep(subset.loc[:,['Clone_count_1','Clone_count_2']])

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

# select sig expanded

cdfPs_n1n2ps=np.cumsum(Ps_n1n2ps,0)


def get_pvalue(row,svec,cdfPs_n1n2ps):
    return cdfPs_n1n2ps[np.argmin(np.fabs(svec)),row.sparse_ind]


get_pvalue_part=partial(get_pvalue,svec=svec,cdfPs_n1n2ps=cdfPs_n1n2ps)
cdflabel=r'$1-P(s>0)$'
subset[cdflabel]=subset.apply(get_pvalue_part, axis=1)

# n1_n2_to_n1n2=np.zeros((len(unicountvals_1),len(unicountvals_2)),dtype=int)
# for n1it,n1 in enumerate(unicountvals_1):
#     for n2it,n2 in enumerate(unicountvals_2):
#         ind=np.where(np.logical_and(n1==unicountvals_1[indn1],n2==unicountvals_2[indn2]))[0]
#         if not ind:
#             continue
#         else:
#             n1_n2_to_n1n2[n1it,n2it]=int(ind) 
# pval_threshold=0.1  #make data size manageable by outputting only clones with pval below this threshold
# min_pair_sum=0
# get_sparseind_part=partial(get_sparse_ind,unicountvals_1=unicountvals_1,unicountvals_2=unicountvals_2,n1_n2_to_n1n2=n1_n2_to_n1n2)        
# subset['sparse_ind']=subset.apply(get_sparseind_part, axis=1)
# subset.sparse_ind=subset.sparse_ind.astype('int32')
# table=suffstats_table(pval_threshold,svec,logPsvec,subset.loc[subset.Clone_count_1+subset.Clone_count_2>=min_pair_sum],sparse_rep_d,logPn1n2_s)
strout='expanded'
table=pd.read_csv(outpath+diff_pair+'table_top_'+strout+'.csv',sep='\t')

table.head()

# averaeg f versus average s

# +
logPn2_fs=np.zeros((len(logfvec),len(sparse_rep_counts)))
for f_it in range(len(logfvec)):
    tmp=logPn2_f[np.arange(len(svec))*f2s_step+f_it,:] 
    logPn2_fs[f_it,:]=np.sum(np.exp(tmp[:,indn2] + logPsvec[:,np.newaxis]),axis=0) #logrhofvec[:,np.newaxis] + logfvec[:,np.newaxis])

logPn1n2f=logPn1_f[:,indn1] + logPn2_fs + logrhofvec[:,np.newaxis]
integ=np.exp(logPn1n2f + logfvec[:,np.newaxis])

logPn1n2=np.log(np.sum(dlogfby2[:,np.newaxis]*(integ[1:,:]+integ[:-1,:]),axis=0))

integ=np.exp(logPn1n2f + 2*logfvec[:,np.newaxis])
logf_Pn1n2f=np.log(np.sum(dlogfby2[:,np.newaxis]*(integ[1:,:]+integ[:-1,:]),axis=0))

logf_n1n2=logf_Pn1n2f-logPn1n2
# -

# select expanded 

fig,ax=pl.subplots()
counts,bins=np.histogram(logf_n1n2/np.log(10),weights=sparse_rep_counts,bins=100)
ax.plot(bins[:-1],counts)
ax.set_yscale('log')

# smed versus fmed

smed_n1n2=np.sum(svec[:,np.newaxis]*Ps_n1n2ps,axis=0)

# +
fig,ax=pl.subplots()
smedvec= np.zeros((len(indn1),))
smeanvec= np.zeros((len(indn1),))
pval=0.025 #double-sided comparison statistical test
pvalvec=[pval,0.5,1-pval] #bound criteria defining slow, smed, and shigh, respectively
pvals=np.zeros((len(indn1),))
for it,column in enumerate(np.transpose(Ps_n1n2ps)):
    forwardcmf=np.cumsum(column)
    backwardcmf=np.cumsum(column[::-1])[::-1]
    inds=np.where((forwardcmf>=pvalvec[1]) & (backwardcmf>=pvalvec[1]))[0]
    smedvec[it]=np.mean(svec[inds]-shift)
    smeanvec[it]=np.sum((svec-shift)*column)
    pvals[it]=forwardcmf[np.argmin(np.fabs(svec-shift))]
# svals=smedvec
svals=smeanvec

cond=(pvals<pval) & (logf_n1n2/np.log(10)>-5.5)
# cond=np.ones((len(indn1),)).astype(bool)
#positie and detectable bin according to log f. Bin.
ax.scatter(logf_n1n2/np.log(10),svals,s=10,c='gray',linewidth=0)#, c=10, cmap='viridis')
s_cond=svals[cond]
f_cond=logf_n1n2[cond]
ax.scatter(f_cond/np.log(10),s_cond,s=10,c='r',linewidth=0)#, c=10, cmap='viridis')
ax.set_ylabel(r'$\langle s\rangle_{|n_,n^\prime}$')
ax.set_xlabel(r'$\log_{10}\langle f\rangle_{|n_,n^\prime}$')
ax.set_xlim(-6,-2)
ax.set_ylim(0,2)
# ax.scatter(f_cond/np.log(10),s_cond,s=10,c=sparse_rep_counts[cond],linewidth=0,color='r')#, c=10, cmap='viridis')

# for nbins in [5,10,20,50]:
#     logfvec_bins=np.linspace(np.min(f_cond),np.max(f_cond),nbins+1)
#     mean_s=[]
#     std_s=[]
#     counts=[]
#     for it,logf in enumerate(logfvec_bins[:-1]):
#         mean_s.append(np.mean(s_cond[np.logical_and(f_cond>logfvec_bins[it],f_cond<logfvec_bins[it+1])]))
#         std_s.append(np.std(s_cond[np.logical_and(f_cond>logfvec_bins[it],f_cond<logfvec_bins[it+1])]))
#         counts.append(np.sum(np.logical_and(f_cond>logfvec_bins[it],f_cond<logfvec_bins[it+1])))
#     # ax.errorbar((logfvec_bins[:-1]+np.diff(logfvec_bins)/2)/np.log(10),mean_s,yerr=std_s,capsize=20,elinewidth=3)
#     ax.errorbar((logfvec_bins[:-1]+np.diff(logfvec_bins)/2)/np.log(10),mean_s,yerr=mean_s/np.sqrt(counts),capsize=20,elinewidth=3)

fig.savefig("s_mean_vs_logf_mean_correlation.pdf",format='pdf',dpi=500,bbox_inches='tight')

# +
integ=Pf_n1n2=np.exp(logPn1n2f-logPn1n2[np.newaxis,:])

fmed_n1n2=np.sum(svec[:,np.newaxis]*Ps_n1n2ps,axis=0)
# -

pval=np.cumsum(Ps_n1n2ps,axis=1)[:,int((len(svec)-1)/2)]

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

# Range over ridge

shiftMtr= np.load(outpath+'shift.npy')
sbarvec_s= np.load(outpath+'sbarvec_p.npy')
alpvec_s= np.load(outpath+'alpvec.npy')

log10sbar_s=np.linspace(-0.2,0.6,18)
log10sbar=np.concatenate((np.linspace(-1,-0.2,9)[:-1],log10sbar_s))
log10alp=np.concatenate((np.zeros(len(log10sbar_s)),-8*(log10sbar_s+0.2)))
sbar_seq=np.power(10,log10sbar)
alp_seq=np.power(10,log10alp)

log10sbar_s=np.linspace(-0.2,0.6,18)
log10sbar=np.concatenate((np.linspace(-1,-0.2,9)[:-1],log10sbar_s))
log10alp=np.concatenate((np.zeros(len(log10sbar_s)),-8*(log10sbar_s+0.2)))
sbar_seq=np.power(10,log10sbar)
alp_seq=np.power(10,log10alp)
shift_seq=np.zeros(len(sbar_seq))
#ridge parametrization
maxc=10
Ps_9_store=np.zeros((len(sbar_seq),len(svec),maxc))
for it in range(len(sbar_seq)):
    logPsvec = get_logPs_pm((alp_seq[it],sbar_seq[it]),smax,s_step,Ps_type)
    shift,Z,Zdash,shift_it=get_shift(logPsvec,null_paras,sparse_rep,acq_model_type,shift,logfvec,logfvecwide,svec,f2s_step,logPn1_f,logrhofvec)
    shift_seq[it]=shift                    
    fvecwide_shift=np.exp(logfvecwide-shift) #implements shift in Pn2_fs
    svec_shift=svec-shift
    logPn2_f=get_logPn_f(unicountvals_2,NreadsII,np.log(fvecwide_shift),acq_model_type,null_paras)
    dlogfby2=np.diff(logfvec)/2
    logPn1n2_s=np.zeros((len(svec),10))
    for s_it in range(len(svec)):
        integ=np.exp(logPn1_f[:,0,np.newaxis] + logPn2_f[f2s_step*s_it:f2s_step*s_it+len(logfvec),1:maxc+1] + logrhofvec[:,np.newaxis] + logfvec[:,np.newaxis])
        logPn1n2_s[s_it,:]=np.log(np.sum(dlogfby2[:,np.newaxis]*(integ[1:,:]+integ[:-1,:]),axis=0)) #can't use dot since multidimensional
    Psn1n2_ps=np.exp(logPn1n2_s+logPsvec[:,np.newaxis])  
    Pn1n2_ps=np.sum(Psn1n2_ps,0)
    Ps_n1n2ps=np.exp(logPn1n2_s+logPsvec[:,np.newaxis])/Pn1n2_ps[np.newaxis,:]
    Ps_9_store[it,:,:]=Ps_n1n2ps                 

# +
fig,ax=pl.subplots(1,2,figsize=(6,3))
maxvals=np.zeros((len(sbar_seq),))
maxabsc=np.zeros((len(sbar_seq),))
# colorsre = pl.cm.inferno(np.linspace(0.1, 1, len(sbar_seq)))
colorsre = pl.cm.inferno(np.linspace(0.1, 1, 14-8))
for ind in range(1,10):
    for it in range(len(sbar_seq)):    
#         maxind=int(np.argmax(Ps_9_store[it,int((len(svec)-1)/2)+1:,ind])+(len(svec)-1)/2)+1
#         maxvals[it]=Ps_9_store[it,maxind,ind]
#         maxabsc[it]=svec[maxind]-shift_seq[it]
#         maxvals[it]=np.sum((svec-shift_seq[it])*Ps_9_store[it,:,ind])
        maxabsc[it]=np.sum((svec-shift_seq[it])*Ps_9_store[it,:,ind])
        
    ax[1].plot(sbar_seq,maxabsc,'k-')
    if ind==9:
        for it in range(8,14):
            ax[1].plot(sbar_seq[it],maxabsc[it],'o',color=colorsre[it-8])
    

for it in range(8,14):
    ax[0].plot(svec-shift_seq[it],Ps_9_store[it,:,ind],color=colorsre[it-8])
#     ax[0].plot(maxabsc[it],maxvals[it],'o',color=colorsre[it-8])
#     break

ax[1].text(4.2,4.7,r'$1$')
ax[1].text(4.2,8.2,r'$9$')
ax[1].text(4.2,9.0,r'$n^\prime$')
ax[1].set_xlabel(r'$\bar{s}$')
ax[1].set_ylabel(r'$\langle s\rangle_{\rho(s|n,n^\prime)}$')
ax[1].set_xscale('log')
# ax[1].set_ylim(0,10)
# ax[1].set_xlim(0,1.3)
ax[0].set_yscale('log')
ax[0].set_ylim(1e-4,1e-1)
# ax.plot(svec_shift,Ps,'k--',label='prior')
ax[0].set_xlim(-4,12)
# ax.set_xscale('log')
ax[0].set_xlabel(r'$s$')
ax[0].set_ylabel(r'$P(s|n=0,n^\prime=9,\theta(r))$')
# ax.legend(frameon=False)
ax[0].locator_params(axis='y',numticks=6)
fig.tight_layout()
fig.savefig("posterior_ridge_sweep.pdf",format='pdf',dpi=500)#,bbox_inches='tight',pad_inches=0.1)
