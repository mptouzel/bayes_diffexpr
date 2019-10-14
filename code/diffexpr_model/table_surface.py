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
from lib.proc import suffstats_table, import_data,get_sparserep,get_sparse_ind
from lib.model import get_logPs_pm,get_rhof,get_fvec_and_svec,get_logPn_f
from functools import partial
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

# Data

donorstr='S2'
output_path='../../../output/'

acq_model_type=2
runname='v4_ct_1_mt_'+str(acq_model_type)+'_min0_maxinf'
null_pair=donorstr+'_0_F1_'+donorstr+'_0_F2'
run_name=runname
outpath_null=output_path+null_pair+'/null_pair_'+run_name+'/'
null_paras= np.load(outpath_null+'optparas.npy')

day=15
mincount=0
maxcount=np.Inf
colnames1 = [u'Clone fraction',u'Clone count',u'N. Seq. CDR3',u'AA. Seq. CDR3']
colnames2 = [u'Clone fraction',u'Clone count',u'N. Seq. CDR3',u'AA. Seq. CDR3'] 
dataset_pair=(donorstr+'_0_F1',donorstr+'_'+str(day)+'_F2') 
input_data_path='../../../data/Yellow_fever/prepostvaccine/'
Nclones_samp,subset=import_data(input_data_path,dataset_pair[0]+'_.txt',dataset_pair[1]+'_.txt',mincount,maxcount,colnames1,colnames2)

# Table for opt paras

strout='expanded'
table=pd.read_csv(outpath+diff_pair+'table_top_'+strout+'.csv',sep='\t')

table.head()

table.head() #old

table.loc[:,[r'$s_{1,low}$', r'$s_{2,med}$', r'$s_{3,high}$', r'$s_{max}$', r'$\bar{s}$', r'$f_1$', r'$f_2$', r'$n_1$', r'$n_2$', r'$1-P(s>0)$']].hist(figsize=(16,16))
pl.gcf().tight_layout()

# Run model

sparse_rep=get_sparserep(subset.loc[:,['Clone_count_1','Clone_count_2']])       

smax=25.
s_step=0.1
logrhofvec,logfvec = get_rhof(null_paras[0],np.power(10,null_paras[-1]))
svec,logfvec,logfvecwide,f2s_step,smax,s_step=get_fvec_and_svec(null_paras,s_step,smax)

# +
Ps_type='sym_exp'
startind=0

runname='v4_ct_1_mt_'+str(acq_model_type)+'_st_'+Ps_type+'_min0_maxinf'
diff_pair=donorstr+'_0_F1_'+donorstr+'_'+str(day)+'_F2'
run_name='diffexpr_pair_'+null_pair+'_'+runname
outpath=output_path+diff_pair+'/'+run_name+'/'

alpvec_orig=np.load(outpath+'alpvec.npy')
sbarvec_orig=np.load(outpath+'sbarvec_p.npy')
shift_array_orig=np.load(outpath+'shift.npy')
opt_diffexpr_paras=np.load(outpath + 'opt_diffexpr_paras.npy')
Lsurface=np.load(outpath + 'Lsurface.npy')

# sparse_rep=np.load(outpath+'sparserep.npy').item()
indn1,indn2,sparse_rep_counts,unicountvals_1,unicountvals_2,NreadsI,NreadsII=sparse_rep#.values()
Nsamp=np.sum(sparse_rep_counts)
logPn1_f=get_logPn_f(unicountvals_1,NreadsI,logfvec,acq_model_type,null_paras)
# -

fig,ax=pl.subplots(1,1)
ax.imshow(Lsurface,origin='lower')

opt_diffexpr_paras

max_inds=np.unravel_index(np.argmax(Lsurface[(alpvec_orig>np.power(10,-1.8))[:,np.newaxis] & (sbarvec_orig>np.power(10,-0.4))[np.newaxis,:]]),Lsurface.shape)
max_inds

max_inds=np.unravel_index(np.argmax(Lsurface),Lsurface.shape)
alpvec_orig[max_inds[0]]
# sbarvec_orig[max_inds[1]]

alpvec=alpvec_orig[alpvec_orig>np.power(10,-1.8)]
np.log10(alpvec)

sbarvec=sbarvec_orig[sbarvec_orig>np.power(10,-0.4)]
sbarvec

n1_n2_to_n1n2=np.zeros((len(unicountvals_1),len(unicountvals_2)),dtype=int)
for n1it,n1 in enumerate(unicountvals_1):
    for n2it,n2 in enumerate(unicountvals_2):
        ind=np.where(np.logical_and(n1==unicountvals_1[indn1],n2==unicountvals_2[indn2]))[0]
        if not ind:
            continue
        else:
            n1_n2_to_n1n2[n1it,n2it]=int(ind)                
get_sparseind_part=partial(get_sparse_ind,unicountvals_1=unicountvals_1,unicountvals_2=unicountvals_2,n1_n2_to_n1n2=n1_n2_to_n1n2)        
subset['sparse_ind']=subset.apply(get_sparseind_part, axis=1)
subset.sparse_ind=subset.sparse_ind.astype('int32')

alpvec=alpvec_orig[alpvec_orig>np.power(10,-1.8)]
sbarvec=sbarvec_orig[sbarvec_orig>np.power(10,-0.4)]
shift_array=np.reshape(shift_array_orig[(alpvec_orig>np.power(10,-1.8))[:,np.newaxis] & (sbarvec_orig>np.power(10,-0.4))[np.newaxis,:]],(len(alpvec),len(sbarvec)))
tables=np.empty((len(alpvec),len(sbarvec)),dtype=object)
for ait,alp in enumerate(alpvec):
    for sit,sbar in enumerate(sbarvec):
        if not os.path.exists(outpath+'grid_table_top_expanded_'+str(ait-6)+'_'+str(sit)+'.csv'):
            print(str(ait)+' '+str(sit))
            logPsvec = get_logPs_pm([alp,sbar],smax,s_step,Ps_type)
            shift=shift_array[ait,sit]

            fvecwide_shift=np.exp(logfvecwide-shift) #implements shift in Pn2_fs
            svec_shift=svec-shift
            logPn2_f=get_logPn_f(unicountvals_2,NreadsII,np.log(fvecwide_shift),acq_model_type,null_paras) #shift makes it dependent on (sbar,alpha)
            logPn1n2_s=np.zeros((len(svec),len(sparse_rep_counts)))
            dlogfby2=np.diff(logfvec)/2
            for s_it in range(len(svec)):
                integ=np.exp(logPn1_f[:,indn1] + logPn2_f[f2s_step*s_it:f2s_step*s_it+len(logfvec),indn2] + logrhofvec[:,np.newaxis] + logfvec[:,np.newaxis])
                logPn1n2_s[s_it,:]=np.log(np.sum(dlogfby2[:,np.newaxis]*(integ[1:,:]+integ[:-1,:]),axis=0)) #can't use dot since multidimensional

            pval_threshold=0.1  #make data size manageable by outputting only clones with pval below this threshold
            min_pair_sum=0
            print('comp table')
            table=suffstats_table(pval_threshold,svec,logPsvec,subset.loc[subset.Clone_count_1+subset.Clone_count_2>=min_pair_sum],sparse_rep,logPn1n2_s)

            #select only clones whose posterior median pass the given threshold (can also be done in post-processing)
            #smed_threshold=3.46 #ln(2^5) threshold on the posterior median, below which clones are discarded
            #table=table[table[r'$s_{2,med}$']>smed_threshold]

            print("writing to: "+outpath)
            pval_expanded=True #which end of the rank list to pull out. else: most contracted
            cdflabel=r'$1-P(s>0)$'
            table=table.sort_values(by=cdflabel,ascending=True)
            strout='expanded'
            tables[ait,sit]=table  
            table.to_csv(outpath+'grid_table_top_expanded_'+str(ait-6)+'_'+str(sit)+'.csv',sep='\t',index=False)

tables=np.empty((len(alpvec),len(sbarvec)),dtype=object)
for ait,alp in enumerate(alpvec):
    for sit,sbar in enumerate(sbarvec):
        tables[ait,sit]=pd.read_csv(outpath+'grid_table_top_expanded_'+str(ait-6)+'_'+str(sit)+'.csv',sep='\t')

tables[0,0].columns

max_inds

tables.shape

base_table=tables[max_inds]
base_table=base_table.set_index('CDR3_nt')
# list(base_table.loc[base_table[r'$s_{2,med}$']>smed_threshold,'CDR3_nt'].values)
base_table.shape

# +
list_len=np.zeros((len(alpvec),len(sbarvec)))
intersection=np.zeros((len(alpvec),len(sbarvec)))
union=np.zeros((len(alpvec),len(sbarvec)))

smed_threshold=3.46 #ln(2^5) threshold on the posterior median, below which clones are discarded

# base_table=tables[max_inds[::-1]]
for ait,alp in enumerate(alpvec):
    print(ait)
    for sit,sbar in enumerate(sbarvec):
        table=tables[ait,sit]
        table=table.set_index('CDR3_nt')
        list_len[ait,sit]=int(len(table.loc[table[r'$s_{2,med}$']>smed_threshold,'CDR3_AA']))
        st=time.time()
#         intersection[ait,sit]=table.loc[table[r'$s_{2,med}$']>smed_threshold,'CDR3_nt'].isin(base_table.loc[base_table[r'$s_{2,med}$']>smed_threshold,'CDR3_nt']).sum()/len(base_table)
#         union[ait,sit]=table.loc[table[r'$s_{2,med}$']>smed_threshold,'CDR3_nt'].isin(base_table.loc[base_table[r'$s_{2,med}$']>smed_threshold,'CDR3_nt']).sum()/len(base_table)
        intersection[ait,sit]=len(table[table[r'$s_{2,med}$']>smed_threshold].index.intersection(base_table[base_table[r'$s_{2,med}$']>smed_threshold].index))
        union[ait,sit]=len(table[table[r'$s_{2,med}$']>smed_threshold].index.union(base_table[base_table[r'$s_{2,med}$']>smed_threshold].index))
        # intersection
# union
# -

fig,ax=pl.subplots(1,1)
titlestrvec=(r"$\langle \mathcal{L} \rangle$",r"$\log_{10}\left(|\langle \mathcal{L} \rangle -\langle \mathcal{L} \rangle_{\textrm{max}} |\right)$")
it=0
alpvec=alpvec_orig[alpvec_orig>np.power(10,-1.8)]
sbarvec=sbarvec_orig[sbarvec_orig>np.power(10,-0.4)]
alpvect=np.log10(alpvec)
sbarvect=np.log10(sbarvec)
p=ax.imshow(intersection/union,extent=[sbarvect[0], sbarvect[-1],alpvect[0], alpvect[-1]], aspect='auto',origin='lower',interpolation='none')#,cmap='viridis')
X, Y = np.meshgrid(sbarvect,alpvect)
# ax[it].contour(X, Y, list_len,levels = 10,colors=('w',),linestyles=('--',),linewidths=(5,))
cb=pl.colorbar(p,ax=ax)
# ax[it].contour(X, Y, Zstore,levels = [0.99,1.,1.01],colors=('k',),linestyles=('-',),linewidths=(5,))
# ax[it].contour(X, Y, Zstore,levels = [1.4],colors=('k',),linestyles=('--',),linewidths=(5,))
# ax[it].contour(X, Y, Zdashstore,levels = [0.99,1.,1.01],colors=('gray',),linestyles=('-',),linewidths=(5,))
# ax[it].contour(X, Y, Zdashstore,levels = [1.4],colors=('gray',),linestyles=('--',),linewidths=(5,))
for rep1 in ['1','2']:
    for rep2 in ['1','2']:
        day='15'
        startind=0
        donorstr='S2'
        runname='v4_ct_1_mt_2_st_'+Ps_type+'_min0_maxinf'
        output_path='../../../output/'
        null_pair=donorstr+'_0_F1_'+donorstr+'_0_F2'
        diff_pair=donorstr+'_0_F'+rep1+'_'+donorstr+'_'+str(day)+'_F'+rep2
        run_name='diffexpr_pair_'+null_pair+'_'+runname
        outpatht=output_path+diff_pair+'/'+run_name+'/'
        try:
            success=np.load(outpatht+'diffexpr_success.npy').item()
            if success:
                diff_paras=np.load(outpatht+'opt_diffexpr_paras.npy')#.item().x
                sc=ax.scatter([np.log10(diff_paras[1])],[np.log10(diff_paras[0])],facecolors='none',linewidth=1,marker='s',s=50,color='k')
            else:
                print(outpath+'fail')
        except:
            print(outpath)
ax.plot([np.log10(0.73678535)],[np.log10(0.18311211)], 'ko')#[sbarvec[max_inds[1]]],[alpvec[max_inds[0]]],'ko')
ax.set_ylabel(r'$\log_{10}\alpha$')
ax.set_xlabel(r'$\log_{10}\bar{s}$')
ax.set_title(r'Overlap with list at $\bullet$')
fig.tight_layout()
fig.savefig("table_list.pdf",format='pdf',dpi=500,bbox_inches='tight')
# fig.suptitle([donorstr]+runname.split('_')+['0-'+str(day)],y=1.02)

unicountvals_1=range(200)
unicountvals_2=range(200)
logPn1_f=get_logPn_f(unicountvals_1,NreadsI,logfvec,acq_model_type,null_paras)

# +
logPsvec = get_logPs_pm(opt_diffexpr_paras[:-1],smax,s_step,Ps_type)
shift=opt_diffexpr_paras[-1]

fvecwide_shift=np.exp(logfvecwide-shift) #implements shift in Pn2_fs
svec_shift=svec-shift
logPn2_f=get_logPn_f(unicountvals_2,NreadsII,np.log(fvecwide_shift),acq_model_type,null_paras)
dlogfby2=np.diff(logfvec)/2
# -

integ.shape

# +
logPn1n2_s=np.zeros((len(svec),len(unicountvals_1),len(unicountvals_2)))
for s_it in range(len(svec)):
    integ=np.exp(logPn1_f[:,:,np.newaxis] + logPn2_f[f2s_step*s_it:f2s_step*s_it+len(logfvec),np.newaxis,:] + logrhofvec[:,np.newaxis,np.newaxis] + logfvec[:,np.newaxis,np.newaxis])
    logPn1n2_s[s_it,:,:]=np.log(np.sum(dlogfby2[:,np.newaxis,np.newaxis]*(integ[1:,:,:]+integ[:-1,:,:]),axis=0)) #can't use dot since multidimensional

Psn1n2_ps=np.exp(logPn1n2_s+logPsvec[:,np.newaxis,np.newaxis])  
Pn1n2_ps=np.sum(Psn1n2_ps,0)
Ps_n1n2ps=np.exp(logPn1n2_s+logPsvec[:,np.newaxis,np.newaxis])/Pn1n2_ps[np.newaxis,:,:]

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

n1=np.range

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

Ps_

fig,ax=pl.subplots(1,1)
ax.scatter(smedvec[s_naive!=0],s_naive[s_naive!=0],s=(n1[s_naive!=0]+n2[s_naive!=0])/5,alpha=0.2, c=sparse_rep_counts[s_naive!=0], cmap='viridis')
limits=np.asarray(ax.get_xlim())
ax.plot(limits,limits,'k--')
ax.set_ylim(1.05*limits)
ax.plot(limits,[0,0],'k')
ax.plot([0,0],limits,'k')
ax.set_xlim(1.05*limits)
ax.set_xlabel(r'$s_{\textrm{median}}$')
ax.set_ylabel(r'$s_{\textrm{naive}}=\ln \frac{n^{\prime}}{n}$')
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

# Volcano plots

Ps_n1n2ps.shape

# +
table=base_table
smedianvec=table[r'$s_{2,med}$']
pvalplus=table[r'$1-P(s>0)$']
pvalneg=1-table[r'$1-P(s>0)$']

fig1, ax1 = pl.subplots(1,1,figsize=(8,8))
cond_pos=table[r'$1-P(s>0)$']<0.025
cond_neg=1-table[r'$1-P(s>0)$']<0.025
# ax1.scatter(clones.loc[cond_pos,u'$s_{2,med}$'],clones.loc[cond_pos,u'${CDF}_{s=0}$'])
# ax1.scatter(clones.loc[cond_neg,u'$s_{2,med}$'],clones.loc[cond_neg,u'${CDF}_{s=0}$'])

# cond_pos=(smedianvec>0) 
# cond_neg=(smedianvec<=0)
# ax1.scatter(smedianvec[cond_pos],-np.log10(pvalplus[cond_pos]),color='k')
# ax1.scatter(smedianvec[cond_neg],-np.log10(1-pvalneg[cond_neg]),color='k')
mean_est=smedvec
cond_pos=(mean_est>0) 
cond_neg=(mean_est<=0)
ax1.scatter(mean_est[cond_pos],-np.log10( np.sum(Ps_n1n2ps[svec<=0,:],axis=0)[cond_pos]),color='k',marker='.')
ax1.scatter(mean_est[cond_neg],-np.log10(np.sum(Ps_n1n2ps[svec>=0,:],axis=0)[cond_neg]),color='k',marker='.')

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
ax1.set_xlabel('expected log fold-change',fontsize=24)
ax1.set_ylabel('Probability',fontsize=24)
# ax1.set_xlim([-0.5,0.5])
ax1.set_xlim([-7.5,7.5])
ax1.set_ylim([0,4])
# fig1.savefig(path+donorstr+'_volcano_highlight.pdf',format='pdf',dpi=1000, bbox_inches='tight')

# -
unicountvals_1=range(1000)
unicountvals_2=range(1000)
logPn1_f=get_logPn_f(unicountvals_1,NreadsI,logfvec,acq_model_type,null_paras)

# +
logPsvec = get_logPs_pm(opt_diffexpr_paras[:-1],smax,s_step,Ps_type)
shift=opt_diffexpr_paras[-1]

fvecwide_shift=np.exp(logfvecwide-shift) #implements shift in Pn2_fs
svec_shift=svec-shift
logPn2_f=get_logPn_f(unicountvals_2,NreadsII,np.log(fvecwide_shift),acq_model_type,null_paras)
dlogfby2=np.diff(logfvec)/2
# -

logPn1n2_s=np.zeros((len(svec),len(unicountvals_1),len(unicountvals_2)))
for s_it in range(len(svec)):
    integ=np.exp(logPn1_f[:,:,np.newaxis] + logPn2_f[f2s_step*s_it:f2s_step*s_it+len(logfvec),np.newaxis,:] + logrhofvec[:,np.newaxis,np.newaxis] + logfvec[:,np.newaxis,np.newaxis])
    logPn1n2_s[s_it,:,:]=np.log(np.sum(dlogfby2[:,np.newaxis,np.newaxis]*(integ[1:,:,:]+integ[:-1,:,:]),axis=0)) #can't use dot since multidimensional
np.save('logPn1n2_s_1000_by_1000.npy',logPn1n2_s)

logPn1n2_s=np.load('logPn1n2_s_1000_by_1000.npy')
Psn1n2_ps=np.exp(logPn1n2_s+logPsvec[:,np.newaxis,np.newaxis])  
Pn1n2_ps=np.sum(Psn1n2_ps,0)
Ps_n1n2ps=Psn1n2_ps/Pn1n2_ps[np.newaxis,:,:]

fig,ax=pl.subplots(1,1)
for n1 in range(1,10,50):
    ax.plot(svec,Ps_n1n2ps[:,n1,n1])
    print(np.sum(svec*Ps_n1n2ps[:,n1,n1]))
ax.set_yscale('log')
ax.set_ylim(1e-10,1e0)

# +
fig1, ax = pl.subplots(1,1,figsize=(3.5,3.5))

color='k'

#contracted
for nit1,n2 in enumerate([0,1,2,4,8,16,32,64,128,256,512,999]):
    mean_s_n1n2=np.sum(svec[:,np.newaxis]*Ps_n1n2ps[:,n2:,n2],axis=0)
    prob_n1n2=-np.log10( np.sum(Ps_n1n2ps[svec>=0,n2:,n2],axis=0)) 
    ax.plot(mean_s_n1n2, prob_n1n2,'-',color=color,lw=0.5)
    if 1.1*prob_n1n2[-1]>1e-2:
        ax.text(1.05*mean_s_n1n2[-1],1.1*prob_n1n2[-1],r'$'+str(n2)+'$',fontsize=8)

for nit1,n1 in enumerate([1,2,4,8,16,32,64,128,256,512,999]):
    mean_s_n1n2=np.sum(svec[:,np.newaxis]*Ps_n1n2ps[:,n1,:n1],axis=0)
    prob_n1n2=-np.log10( np.sum(Ps_n1n2ps[svec>=0,n1,:n1],axis=0)) 
    ax.plot(mean_s_n1n2, prob_n1n2,'-',color=color,lw=0.5)
    if 0.9*prob_n1n2[0]>1e-2:
        ax.text(1.3*mean_s_n1n2[0],0.9*prob_n1n2[0],r'$'+str(n1)+'$',fontsize=8)   
    
#expanded
for nit1,n2 in enumerate([1,2,4,8,16,32,64,128,256,512,999]):
    mean_s_n1n2=np.sum(svec[:,np.newaxis]*Ps_n1n2ps[:,:n2,n2],axis=0)
    prob_n1n2=-np.log10( np.sum(Ps_n1n2ps[svec<=0,:n2,n2],axis=0)) 
    ax.plot(mean_s_n1n2, prob_n1n2,'-',color=color,lw=0.5)
    if 0.9*prob_n1n2[-1]>1e-2:
        ax.text(1.05*mean_s_n1n2[0],0.9*prob_n1n2[0],r'$'+str(n2)+'$',fontsize=8)

for nit1,n1 in enumerate([0,1,2,4,8,16,32,64,128,256,512,999]):
    mean_s_n1n2=np.sum(svec[:,np.newaxis]*Ps_n1n2ps[:,n1,n1:],axis=0)
    prob_n1n2=-np.log10( np.sum(Ps_n1n2ps[svec<=0,n1,n1:],axis=0)) 
    ax.plot(mean_s_n1n2, prob_n1n2,'-',color=color,lw=0.5)
    if 1.2*prob_n1n2[-1]>1e-2:
        ax.text(0.95*mean_s_n1n2[-1],1.2*prob_n1n2[-1],r'$'+str(n1)+'$',fontsize=8) 

ax.set_xlabel(r'expected log fold-change,$\langle s\rangle_{\rho(s|n,n^{\prime})}$')
ax.set_ylabel(r'Confidence, $-\log_{10}P_{\textrm{null}}$')
# ax.grid('off')
ax.set_xlim([-10,10])
ax.set_ylim([1e-2,5e2])
ax.set_xticks(np.arange(-10,11,2))
ax.set_yscale('log')
# ax.set_xscale('log')
pthresh=0.025
ax.plot(ax.get_xlim(),[-np.log10(pthresh),-np.log10(pthresh)],'k--')
ax.text(ax.get_xlim()[0],-np.log10(pthresh)*0.6,'significance \n threshold') 
ax.text(-7,10,r'$n$',fontsize=14) 
ax.text(-4,2.8e2,r'$n^{\prime}$',fontsize=14) 
ax.text(8.5,10,r'$n^{\prime}$',fontsize=14) 
ax.text(4,2.8e2,r'$n$',fontsize=14) 

# ax.plot([0,0],ax.get_ylim(),'k-')
fig1.savefig('volcano_highlight.pdf',format='pdf',dpi=1000, bbox_inches='tight')

# -


mean_s_n1n2=np.sum(svec[:,np.newaxis,np.newaxis]*Ps_n1n2ps,axis=0)
prob_n1n2_con=-np.log10( np.sum(Ps_n1n2ps[svec>=0,:,:],axis=0)) 
prob_n1n2_exp=-np.log10( np.sum(Ps_n1n2ps[svec<=0,:,:],axis=0)) 
region=(prob_n1n2_exp> -np.log10(0.025))*(mean_s_n1n2>0)+ (prob_n1n2_con> -np.log10(0.025))*(mean_s_n1n2<0)

# +
exp_region=(prob_n1n2_exp> -np.log10(0.025))*(mean_s_n1n2>0)
exp_thresh=np.zeros((1000,))
for n in range(1000):
    exp_thresh[n]=np.argmax(exp_region[n,:])+1

con_region=(prob_n1n2_con> -np.log10(0.025))*(mean_s_n1n2<0)
con_thresh=np.zeros((1000,))
for n in range(1000):
    con_thresh[n]=np.argmin(con_region[n,:])+1
# -

fig,ax=pl.subplots(1,1,figsize=(1,1))
ax.imshow(region,origin='lower')
# ax.plot(con_thresh)
# ax.plot(exp_thresh)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1,1000)
ax.set_ylim(1,1000)
ax.set_ylabel('$n^{\prime}$')
ax.set_xlabel('$n$')
fig.savefig('volcano_inset.pdf',format='pdf',dpi=1000, bbox_inches='tight')



