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

# # Replicate model: definition, inference consistency on sampled data and example validation on real data

# %matplotlib inline
import sys,os
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
# %run -i '../lib/utils/ipynb_setup.py'
from lib.utils.plotting import plot_n1_vs_n2,add_ticks
from lib.proc import get_sparserep,import_data
from lib.model import get_Pn1n2_s, get_rhof, NegBinParMtr,get_logPn_f,get_model_sample_obs
from lib.learning import nullmodel_constr_fn,callback,learn_null_model
import lib.learning
# %load_ext autoreload
# %autoreload 2

# +
import matplotlib.pyplot as pl
paper=False
if not paper:
    pl.rc("figure", facecolor="gray",figsize = (8,8))
    pl.rc('lines',markeredgewidth = 2)
    pl.rc('font',size = 24)
else:
    pl.rc("figure", facecolor="none",figsize = (3.5,3.5))
    pl.rc('lines',markeredgewidth = 0)
    pl.rc('font',size = 10)
    
pl.rc('text', usetex=True)

import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : True})
params= {'text.latex.preamble' : [r'\usepackage{amsmath}']}
pl.rcParams.update(params)
# -

# We begin by defining the parameters of the model:

#set as those learned on S2 day0-day0
acq_model_type=2
paras=np.asarray([-2.15199494,  1.27984874,  1.02263351,  -9.73615977])
Nreads=1464283#1498170+1430396
Nsamp=1497221

logf_samples_eff,pair_samples=get_model_sample_obs(paras,Nsamp,Nreads)

# What does this sample look like?

sparse_rep=get_sparserep(pair_samples)
plot_n1_vs_n2(sparse_rep,'','joint_histogram_effective',0)

# Now, scale up to a human repertoire and try to reinfer the parameters from this sample:

# ## Reinference

# +
output=[]
fsamplesS=[]
pair_samplesS=[]
constr_storeS=[]
N_trials=20
paras_store=list()
f_samples_store=list()
pair_samples_store=list()
constr_store=list()

initparas = paras#*(1+0.001*(np.random.random(size=len(paras))-1))
logPn1_f=np.zeros((1,))
logPn2_f=np.zeros((1,))
case_output=[]
for nit in range(N_trials):
    tmpdict={}
    np.random.seed(1+nit)
    #this method only samples in the observed domain
    f_samples, pair_samples=get_model_sample_obs(paras,Nsamp,Nreads)
    tmpdict['f']=f_samples
    tmpdict['n1n2']=pair_samples
    #transform to sparse representation
    sparse_rep=get_sparserep(pair_samples)

    for constr_type in range(3):
        
        #learn
        init_paras=paras
        st=time.time()
        outstruct,constr_value=learn_null_model(sparse_rep,acq_model_type,init_paras,constr_type=constr_type)
        print('constraint values: ')
        print(constr_value)
        print('took '+str(time.time()-st))        
        #store results
        f_samples_store.append(f_samples)
        pair_samples_store.append(pair_samples)
        constr_store.append(constr_value)
        case_output.append(outstruct)
        
    output.append(case_output)
    fsamplesS.append(f_samples_store)
    pair_samplesS.append(pair_samples_store)
    constr_storeS.append(constr_store)
#save dictionary of results    
data={}
data['f']=fsamplesS
data['n1n2']=pair_samplesS
data['constr']=constr_storeS
data['out']=output
np.save('syn_null_models',data)
# -

data=np.load('syn_null_models.npy',encoding='latin1').item()

sumf=np.zeros((20,))
for tit,trial in enumerate(data['f'][it]):    
#     maxf[tit]=np.max(trial)
    sumf[tit]=np.sum(trial[np.argpartition(trial,-nummax)[-nummax:]])
np.argsort(sumf)

maxf=np.zeros((20,))
for tit,trial in enumerate(data['f'][it]):    
#     maxf[tit]=np.max(trial)
    maxf[tit]=np.sum(trial[np.argpartition(trial,-nummax)[-nummax:]])
fig,ax=pl.subplots(1,1)
# df=pd.DataFrame(maxf.T,columns=['C1','C2','C1+C2']).melt(value_vars=['C1','C2','C1+C2'],value_name='$f_{max}$',var_name='constraint')
sns.scatterplot(ax=ax,
               data=pd.DataFrame(maxf), 
              color='k', # Make points black
              alpha=0.7) # and slightly transparent
print(np.where(maxf>-3))

# +
it=0
maxf=np.zeros((20,))
for tit,trial in enumerate(data['f'][it]):    
    maxf[tit]=np.max(trial)
maxn=np.zeros((20,))
for tit,trial in enumerate(data['n1n2'][it]):    
    maxn[tit]=trial.max().max()
    
logfthresh=-3.
nthresh=7e4
fig,ax=pl.subplots(1,3,figsize=(16,8))
for axit in range(3):
    
    #select trials based on fmax
    if axit==2:
#         seltrials=np.where(maxf>logfthresh)[0]
        seltrials=np.where(maxf>logfthresh)[0]
        title='4/20 $f_{max}>10^{-3}$ trials '
    elif axit==1:
#         seltrials=np.where(maxf<logfthresh)[0]
        seltrials=[4,3,6,8,9,13]
        title='6/20 high-error trials'
    else:
        seltrials=range(20)
        title='20 trials'


    constr_names=(r'$N\langle f \rangle=1$',r'$N\langle f \rangle_{|data}=1$','both')
    para_names=(r'$\alpha_{\rho}$',r'$\log_{10}a$',r'$\gamma$',r'$\log_{10}f_\textrm{min}$')
    df=pd.DataFrame(columns=['mag. rel. error','constr','parameter'])

    for it,constr in enumerate(constr_names):
        for tit,trial in enumerate(data['out'][it]):
            if tit in list(seltrials):
                for pit,para in enumerate(paras):
                    row={'mag. rel. error':np.log10(np.abs((trial.x[pit]-para)/para)),'constr':constr,'parameter':para_names[pit]}
                    df=df.append(row,ignore_index=True)
    # Create plot
    sns.violinplot(ax=ax[axit],
                   x='parameter',
                   y='mag. rel. error', 
                   data=df, 
                   hue='constr',
                   inner=None) # Remove the bars inside the violins
                   #palette=pkmn_type_colors)
    handles, labels = ax[axit].get_legend_handles_labels()

    sns.swarmplot(ax=ax[axit],
                  x='parameter',
                   y='mag. rel. error', 
                   data=df, 
                   hue='constr',
                  color='k', # Make points black
                  dodge=True,
                  alpha=0.7) # and slightly transparent
    ax[axit].set_ylim(-8,0)
    # ax.set_yticklabels([r'$10^{'+str(x)+'}$' for x in range(-10,1,2)]);
    # # Set title with matplotlib
    # plt.title('Attack by Type')
    ax[axit].legend(handles, labels,frameon=False)
    ax[axit].set_title(title)
    ax[axit].set_ylabel('$\log_{10}|(p-p^*)/p^*|$')
    ax[axit].set_xlabel('parameter, $p$')

fig.tight_layout()
fig.savefig('high_error_mode_reinference_analysis.pdf',format='pdf',dpi=1000, bbox_inches='tight')
# -

# So there is a low and high error mode. Probably also for alpha_rho, just smaller in size. DIfference not apparent when using only the C2 (realization-based) constraint

# NOw analyze data:

# Loads and plots existing data to evaluate null NB->Pois model reinference over a range of parameters. Needs to be updated

alpha_rhovec=[-2.6,-2.3,-2.0]
logbetavec=[-0.2,0.2, 0.4]
alpha_mvec=[1.1,1.5,1.8]
logm_totalvec=[6.,6.5,7.0]
parasStore=(alpha_rhovec,logbetavec,alpha_mvec,logm_totalvec)
casestrvec=(r'$NB\rightarrow Pois$',r'$Pois \rightarrow NB$','$NB$','$Pois$')
fig,ax=pl.subplots(1,4,figsize=(16,4))
case=0
outstructs=np.empty((3,3,3,3),dtype=dict)
nitvec=list([0])
alpha_mtest=list()
logbetatest=list()
alpha_mtest_act=list()
logbetatest_act=list()
for mtit,logm_total in enumerate(logm_totalvec):
# mtit=2
# logmtotal=7.0
    for rit, alpha_rho in enumerate(alpha_rhovec):
#         rit=0
#         alpha_rho=-2.6
        for lit, logbeta in enumerate(logbetavec):
            for mit,alpha_m  in enumerate(alpha_mvec):
                paras=[alpha_rho,logbeta,alpha_m,logm_total]
                runstr='../../../output/syn_data/nullmodel_reinfer_data/learnnulltestv5_case_'+str(case)+'_'+str(rit)+'_'+str(lit)+'_'+str(mit)+'_'+str(mtit)+'_'
                try:
                    outstructs[rit,lit,mit,mtit]=np.load(runstr+'outstruct.npy',encoding='latin1').flatten()[0]
                    if outstructs[rit,lit,mit,mtit].success==False:
                        print('failed!')
                    
                    nitvec.append(outstructs[rit,lit,mit,mtit].nit)
                    pred_paras=outstructs[rit,lit,mit,mtit].x
#                     if np.sum(np.fabs(pred_paras)-np.fabs(paras))>0.2:
#                         print(str([rit,lit,mit,mtit]))
#                         print('error:'+str(pred_paras-paras))
                    for pit,p in enumerate(pred_paras):
                        ax[pit].plot(paras[pit],p,'*')
                    logbetatest.append(pred_paras[1])
                    alpha_mtest.append(pred_paras[2])
                    logbetatest_act.append(paras[1])
                    alpha_mtest_act.append(paras[2])
                except:
#                     print('')
                    print(str(rit*3*3*3+lit*3*3+mit*3+mtit)+' '+runstr)
parastrvec=(r'$\alpha_{\rho}$',r'$\log_{10}a$',r'$\gamma$',r'$\log_{10}M$')
for pit in range(4):                    
    ax[pit].plot([parasStore[pit][0],parasStore[pit][-1]],[parasStore[pit][0],parasStore[pit][-1]],'k--')
    ax[pit].set_xlabel('actual',fontsize=24)
    ax[pit].set_ylabel('estimated',fontsize=24)
    ax[pit].set_title(parastrvec[pit],fontsize=24)
fig.suptitle(casestrvec[case],y=1.01)
fig.tight_layout()
fig.savefig('NB_Pois_nullpara_fits.pdf',format='pdf',dpi=1000, bbox_inches='tight')

# cool plot for anti corr in alpha and gamma

fig,ax=pl.subplots(1,1)
ax.scatter(alpha_mtest,logbetatest)
ax.scatter(alpha_mtest_act,logbetatest_act,marker='+',color='r',s=160)
ax.set_ylabel(r'$\log_{10}a$',size=24)
ax.set_xlabel(r'$\gamma$',size=24)
fig.savefig('NB_Pois_nullpara_fits_loga_versus_gamma.pdf',format='pdf',dpi=1000, bbox_inches='tight')

# Now for NBonly model

alpha_rhovec=[-2.6,-2.3,-2.0]
logbetavec=[-0.2,0.2, 0.4]
alpha_mvec=[1.1,1.5,1.8]
logm_totalvec=[6.,6.5,7.0]
parasStore=(alpha_rhovec,logbetavec,alpha_mvec,logm_totalvec)
casestrvec=(r'$NB\rightarrow Pois$',r'$Pois \rightarrow NB$','$NB$','$Pois$')
fig,ax=pl.subplots(1,3,figsize=(16,4))
case=2
mtit=0
outstructs=np.empty((3,3,3),dtype=dict)
for rit, alpha_rho in enumerate(alpha_rhovec):
    for lit, logbeta in enumerate(logbetavec):
        for mit,alpha_m  in enumerate(alpha_mvec):
            paras=[alpha_rho,logbeta,alpha_m,logm_total]
            runstr='learnnullv4_case_'+str(case)+'_'+str(rit)+'_'+str(lit)+'_'+str(mit)+'_'+str(mtit)+'_'
            try:
                outstructs[rit,lit,mit]=np.load(runstr+'outstruct.npy',encoding='latin1').flatten()[0]
                pred_paras=outstructs[rit,lit,mit].x
                
                for pit,p in enumerate(pred_paras):
                    ax[pit].plot(paras[pit],p,'*')
            except:
                print(str(rit*3*3*3+lit*3*3+mit)+' '+runstr)
parastrvec=(r'$\alpha_{\rho}$',r'$\log_{10}a$',r'$\gamma$')
for pit in range(3):                    
    ax[pit].plot([parasStore[pit][0],parasStore[pit][-1]],[parasStore[pit][0],parasStore[pit][-1]],'k--')
    ax[pit].set_xlabel('actual',fontsize=24)
    ax[pit].set_ylabel('estimated',fontsize=24)
    ax[pit].set_title(parastrvec[pit],fontsize=24)
fig.suptitle(casestrvec[case],y=1.01)
fig.tight_layout()

# And Pois model

alpha_rhovec=[-2.6,-2.3,-2.0]
logbetavec=[-0.2,0.2, 0.4]
alpha_mvec=[1.1,1.5,1.8]
logm_totalvec=[6.,6.5,7.0]
parasStore=(alpha_rhovec,logbetavec,alpha_mvec,logm_totalvec)
casestrvec=(r'$NB\rightarrow Pois$',r'$Pois \rightarrow NB$','$NB$','$Pois$')
fig,ax=pl.subplots(1,1,figsize=(16,4))
case=3
mtit=0
lit=0
mit=0
mtit=0
outstructs=np.empty((3),dtype=dict)
for rit, alpha_rho in enumerate(alpha_rhovec):
    paras=[alpha_rho]
    runstr='learnnullv3_case_'+str(case)+'_'+str(rit)+'_'+str(lit)+'_'+str(mit)+'_'+str(mtit)+'_'
#     try:
    outstructs[rit]=np.load(runstr+'outstruct.npy').flatten()[0]
    pred_paras=outstructs[rit].x
    for pit,p in enumerate(pred_paras):
        ax.plot(paras[0],p,'*')
#     except:
#         print(str(rit*3)+' '+runstr)
parastrvec=(r'$\alpha_{\rho}$',r'$\log_{10}a$',r'$\gamma$')
ax.plot([parasStore[pit][0],parasStore[pit][-1]],[parasStore[pit][0],parasStore[pit][-1]],'k--')
ax.set_xlabel('actual',fontsize=24)
ax.set_ylabel('estimated',fontsize=24)
ax.set_title(parastrvec[pit],fontsize=24)
fig.suptitle(casestrvec[case],y=1.01)
fig.tight_layout()

# compare AICs

for case in range(4):
    try:
        outstruct=np.load('learnnullv2_Azh_case_'+str(case)+'_outstruct.npy').flatten()[0]
        Nsamp=1497221 #Azh
        print(str(case)+': '+str(outstruct.x)+' AIC='+str(Nsamp*(2*len(outstruct.x)/Nsamp+2*outstruct.fun))+', BIC='+str(Nsamp*(len(outstruct.x)*np.log(Nsamp)/Nsamp+2*outstruct.fun)))
    except:
        print('not v2')
#     try:
#         outstruct=np.load('learnnullv2_Azh_case_'+str(case)+'_outstruct.py').flatten()[0]
#     except:
        print('not v2')

# Other stuff:

likelihoods=list()
for tit, outstruct in enumerate(paras_store):
    parastmp=paras_store[tit].x

    indn1_d,indn2_d,countpaircounts_d,unicountvals_1_d,unicountvals_2_d,NreadsI_d,NreadsII_d=get_sparserep(pair_samples_store[tit])   
    partialobjfunc=partial(get_Pn1n2_s,svec=-1,unicountvals_1=unicountvals_1_d, unicountvals_2=unicountvals_2_d, NreadsI=NreadsI_d, NreadsII=NreadsII_d, nfbins=nfbins,repfac=repfac,indn1=indn1_d ,indn2=indn2_d,countpaircounts_d=countpaircounts_d,acq_model_type=case)
    npoints=7
    logLikelihood_NBPois=np.zeros((npoints,npoints))
    lowfac=0.95
    for pit, para in enumerate(parastmp):
        paratmpp=deepcopy(parastmp)
        for ppit,ppara in enumerate(np.linspace(para*lowfac,para+para*(1-lowfac),npoints)):
            paratmpp[pit]=ppara
            logLikelihood_NBPois[pit,ppit]=partialobjfunc(paratmpp)
    likelihoods.append(logLikelihood_NBPois)

import numpy.polynomial.polynomial as poly

# Data has a low and high error mode with different parameters and likelihoods:

ig1, axarr = pl.subplots(1, 4)
strvec=(r'\alpha_{\rho}',r'\beta_m',r'\alpha_m',r'\log_{10}f_{min}')
linstylist=('-o','-v','-^','-s')
rcParams['figure.figsize']=(20,5)
rcParams['font.size']= 24
pl.rcParams.update({'font.size': 24})
for tit, likelihood in enumerate(likelihoods):
    parastmp=paras_store2[tit].x
    coeffs=np.zeros((len(paras),3))
    for pit,para in enumerate(parastmp):
        ifptmp=np.asarray([outstruct.x[pit] for outstruct in paras_store2])
        mcolor='r' if (np.log10(np.fabs((para-paras[pit])/paras[pit]))>-2.5) else 'b'
        vec=np.linspace(para*lowfac,para+para*(1-lowfac),npoints)
        coeffs[pit,:]=poly.polyfit(vec-vec[3], likelihood[pit,:], 2)
        axarr[pit].plot(vec,likelihood[pit,:],linstylist[pit][1],label='data',Markersize=10)
        axarr[pit].set_ylabel(r'$\log \mathcal{L}$')
        axarr[pit].set_xlabel(r'$'+strvec[pit]+' $')
        axarr[pit].locator_params(nbins=4)
        vecdense=np.linspace(vec[0],vec[-1],100)
        axarr[pit].plot(vecdense,poly.polyval(vecdense-vec[3],coeffs[pit,:]),mcolor+'-')

# +
figf1,axf1=pl.subplots(1,1)
fign1,axn1=pl.subplots(1,1)
figf2,axf2=pl.subplots(1,1)
fign2,axn2=pl.subplots(1,1)
alpha_rho = paras[0]
nfbins=800
fmin=np.power(10,paras[3])
rhofvec,fvec = get_rhof(alpha_rho,nfbins,fmin) 
for tit,trial in enumerate(paras_store2):    
    for pit,para in enumerate(paras):
        rel_error=np.fabs((paras_store2[tit].x[pit]-paras[pit])/paras[pit])
        if rel_error<0.005:
            [Pn_data,binsn] = np.histogram(f_samples_store2[tit],fvec)
            axf1.plot(binsn[:-1],Pn_data,'-',label=r'effective')
            [Pn_data,binsn] = np.histogram(pair_samples_store2[tit][:,0],np.arange(np.max(pair_samples_store2[tit][:,0])))
            p=axn1.plot(binsn[:-1],Pn_data,'-',label=r'effective')
            [Pn_data,binsn] = np.histogram(pair_samples_store2[tit][:,1],np.arange(np.max(pair_samples_store2[tit][:,1])))
            axn1.plot(binsn[:-1],Pn_data,'--',color=p[-1].get_color(),label=r'effective')
        else:
            [Pn_data,binsn] = np.histogram(f_samples_store2[tit],fvec)
            axf2.plot(binsn[:-1],Pn_data,'-',label=r'effective')
            [Pn_data,binsn] = np.histogram(pair_samples_store2[tit][:,0],np.arange(np.max(pair_samples_store2[tit][:,0])))
            p=axn2.plot(binsn[:-1],Pn_data,'-',label=r'effective')
            [Pn_data,binsn] = np.histogram(pair_samples_store2[tit][:,1],np.arange(np.max(pair_samples_store2[tit][:,1])))
            axn2.plot(binsn[:-1],Pn_data,'--',color=p[-1].get_color(),label=r'effective')
axn1.set_xlabel(r'$f_1$',fontsize=24)
axn1.set_ylabel('count',fontsize=24)
axn1.set_yscale('log')
axn1.set_xscale('log')
axf1.set_xlabel(r'$n_1$',fontsize=24)
axf1.set_ylabel('count',fontsize=24)
axf1.set_yscale('log')
axf1.set_xscale('log')
axn2.set_xlabel(r'$f_2$',fontsize=24)
axn2.set_ylabel('count',fontsize=24)
axn2.set_yscale('log')
axn2.set_xscale('log')
axf2.set_xlabel(r'$n_2$',fontsize=24)
axf2.set_ylabel('count',fontsize=24)
axf2.set_yscale('log')
axf2.set_xscale('log')
axn1.set_ylim(0.5,axn1.get_ylim()[1])
axn1.set_xlim(1,1e5)
axf1.set_ylim(0.5,axf1.get_ylim()[1])

axf2.set_ylim(0.5,axf2.get_ylim()[1])
axn2.set_ylim(0.5,axn2.get_ylim()[1])
axn2.set_xlim(1,1e5)
# -

fig1, ax= pl.subplots(1, 1)
ax.plot(np.asarray([outstruct.x[0] for outstruct in paras_store2]),np.asarray([outstruct.x[3] for outstruct in paras_store2]),'o',label='data',Markersize=10)    
fig1.tight_layout()

fig,ax=pl.subplots(1,4)
trial_err=np.zeros((len(paras_store2),))
for tit,trial in enumerate(paras_store2):
    for pit,para in enumerate(paras):
        rel_error=((trial.x[pit]-paras[pit])/paras[pit])
        
        if np.fabs(rel_error)<0.005:
#             ax[pit].plot(trial.nit,rel_error,'x')
#             ax[pit].plot(trial.fun,rel_error,'x')
#             ax[pit].plot(trial.x[pit],rel_error,'x')
#             ax[pit].plot(np.sum(pair_samples_store2[tit][:,1]),np.fabs(rel_error),'x')
            ax[pit].plot(np.log10(constr_store[tit]),np.log10(np.fabs(rel_error)),'x')


        else:
#             ax[pit].plot(trial.nit,rel_error,'o')
#             ax[pit].plot(np.sum(pair_samples_store2[tit][:,1]),np.fabs(rel_error),'o')
            ax[pit].plot(np.log10(constr_store[tit]),np.log10(np.fabs(rel_error)),'o')
#             ax[pit].plot(trial.x[pit],rel_error,'o')


# Rest is duplicates

# Anticorrelation in learned a and gamma

fig1, ax= pl.subplots(1, 1)
ax.plot(np.asarray([outstruct.x[0] for outstruct in paras_store]),np.asarray([outstruct.x[3] for outstruct in paras_store]),'o',label='data',Markersize=10)    
fig1.tight_layout()

# +
figf1,axf1=pl.subplots(1,1)
fign1,axn1=pl.subplots(1,1)
figf2,axf2=pl.subplots(1,1)
fign2,axn2=pl.subplots(1,1)
for tit,trial in enumerate(paras_store):    
    for pit,para in enumerate(paras):
        rel_error=np.fabs((paras_store[tit].x[pit]-paras[pit])/paras[pit])
        if rel_error<0.005:
            [Pn_data,binsn] = np.histogram(f_samples_store[tit],fvec)
            axf1.plot(binsn[:-1],Pn_data,'-',label=r'effective')
            [Pn_data,binsn] = np.histogram(pair_samples_store[tit][:,0],np.arange(np.max(pair_samples_store[tit][:,0])))
            p=axn1.plot(binsn[:-1],Pn_data,'-',label=r'effective')
            [Pn_data,binsn] = np.histogram(pair_samples_store[tit][:,1],np.arange(np.max(pair_samples_store[tit][:,1])))
            axn1.plot(binsn[:-1],Pn_data,'--',color=p[-1].get_color(),label=r'effective')
        else:
            [Pn_data,binsn] = np.histogram(f_samples_store[tit],fvec)
            axf2.plot(binsn[:-1],Pn_data,'-',label=r'effective')
            [Pn_data,binsn] = np.histogram(pair_samples_store[tit][:,0],np.arange(np.max(pair_samples_store[tit][:,0])))
            p=axn2.plot(binsn[:-1],Pn_data,'-',label=r'effective')
            [Pn_data,binsn] = np.histogram(pair_samples_store[tit][:,1],np.arange(np.max(pair_samples_store[tit][:,1])))
            axn2.plot(binsn[:-1],Pn_data,'--',color=p[-1].get_color(),label=r'effective')
axn1.set_xlabel(r'$f_1$',fontsize=24)
axn1.set_ylabel('count',fontsize=24)
axn1.set_yscale('log')
axn1.set_xscale('log')
axf1.set_xlabel(r'$n_1$',fontsize=24)
axf1.set_ylabel('count',fontsize=24)
axf1.set_yscale('log')
axf1.set_xscale('log')
axn2.set_xlabel(r'$f_2$',fontsize=24)
axn2.set_ylabel('count',fontsize=24)
axn2.set_yscale('log')
axn2.set_xscale('log')
axf2.set_xlabel(r'$n_2$',fontsize=24)
axf2.set_ylabel('count',fontsize=24)
axf2.set_yscale('log')
axf2.set_xscale('log')
axn1.set_ylim(0.5,axn1.get_ylim()[1])
axn1.set_xlim(1,1e5)
axf1.set_ylim(0.5,axf1.get_ylim()[1])

axf2.set_ylim(0.5,axf2.get_ylim()[1])
axn2.set_ylim(0.5,axn2.get_ylim()[1])
axn2.set_xlim(1,1e5)
# -

fig,ax=pl.subplots(1,4)
for tit,trial in enumerate(paras_store):    
    for pit,para in enumerate(paras):
        rel_error=((trial.x[pit]-paras[pit])/paras[pit])
        
        if np.fabs(rel_error)<0.005:
#             ax[pit].plot(trial.nit,rel_error,'x')
#             ax[pit].plot(trial.fun,rel_error,'x')
#             ax[pit].plot(trial.x[pit],rel_error,'x')
            ax[pit].plot(np.sum(pair_samples_store[tit][:,1]),np.fabs(rel_error),'x')

        else:
#             ax[pit].plot(trial.nit,rel_error,'o')
            ax[pit].plot(np.sum(pair_samples_store[tit][:,1]),np.fabs(rel_error),'o')
#             ax[pit].plot(trial.x[pit],rel_error,'o')


fig,ax=pl.subplots(1,1)
counts,bins=np.histogram(np.asarray([np.sum(trial) for trial in f_samples_store]),10)
ax.plot(bins[:-1],counts)


# ## Direct sampling method:

# + {"code_folding": [0, 9, 32]}
def get_sample_direct(Pn_f,logfvec,logrhofvec,N_mouserep,Nsamp):#define D region
    #sample frequencies
    integ=np.exp(logrhofvec+logfvec)
    f_samples_inds=get_distsample(dlogf*(integ[1:] + integ[:-1]),N_mouserep).flatten()
    Z=np.sum(fvec[f_samples_inds])
    fvec/=
#     np.random.shuffle(f_samples_inds)
#     cumfreqsum=np.cumsum(fvec[f_samples_inds])
#     Nsamp=np.argmin(np.fabs(cumfreqsum-1))  #sample until they sum to one.
#     f_samples_inds=f_samples_inds[:Nsamp]
    f_samples_inds=np.sort(f_samples_inds)
    direct_f_samples=fvec[f_samples_inds]
    find_vals,f_start_ind,f_counts=np.unique(f_samples_inds,return_counts=True,return_index=True)

    #sample first count
    direct_n1_samples=np.zeros((Nsamp,),dtype=int)
    direct_n2_samples=np.zeros((Nsamp,),dtype=int)

    for it,find in enumerate(find_vals):       
        direct_n1_samples[f_start_ind[it]:f_start_ind[it]+f_counts[it]]=Pn_f[find].rvs(size=f_counts[it])
        direct_n2_samples[f_start_ind[it]:f_start_ind[it]+f_counts[it]]=Pn_f[find].rvs(size=f_counts[it])
# #         tmp=get_distsample(np.exp(logPn_f[find,:]),f_counts[it]).flatten()
# #         tmp=Pn_f[find,:]),f_counts[it]).flatten()
# #         np.random.shuffle(tmp)
# #         direct_n1_samples[f_start_ind[it]:f_start_ind[it]+f_counts[it]]=tmp

#     #sample second count
#     for it,find in enumerate(find_vals):
#         samples=np.random.random(size=f_counts[it]) * (1-Pn_f[find].cdf(0)) + Pn_f[find].cdf(0)
#         direct_n2_samples[f_start_ind[it]:f_start_ind[it]+f_counts[it]]=Pn_f[find].ppf(samples)        
# #         tmp=get_distsample(np.exp(logPn_f[find,:]),f_counts[it]).flatten()
# #         np.random.shuffle(tmp)
# #         direct_n2_samples[f_start_ind[it]:f_start_ind[it]+f_counts[it]]=tmp

    #remove unseen
    direct_seen=np.logical_or(direct_n1_samples>0, direct_n2_samples>0)
    direct_n1_samples=direct_n1_samples[direct_seen]
    direct_n2_samples=direct_n2_samples[direct_seen]

    direct_f_samples=direct_f_samples[direct_seen]
    print(str(len(direct_n1_samples))+' clones left after discard')
    return direct_f_samples,np.hstack((direct_n1_samples[:,np.newaxis],direct_n2_samples[:,np.newaxis]))


# -

direct_f_samples,direct_pair_samples=get_sample_direct(Pn_f,logfvec,logrhofvec,N_mouserep,Nsamp)

plot_n1_vs_n2(pd.DataFrame({'Clone_count_1':direct_pair_samples[:,0],'Clone_count_2':direct_pair_samples[:,1]}),'','joint_histogram_direct',0)


# + {"code_folding": [0]}
def get_sample_indirect(Pn_f,logfvec,logrhofvec,N_mouserep):#define D region
    print('sample f1 and n1 in notD')

    f0=1e-5

    d=(logfvec<np.log(f0))
    notd=np.where(~d,0,-np.Inf)
    d=np.where(d,0,-np.Inf)

    #P(D)
    integ = np.exp(logfvec+logrhofvec+d)
    logPd = np.log(np.dot(dlogf,integ[1:]+integ[:-1]))

    #<f>_D
    logPf_d=logrhofvec+d-logPd
    integ = np.exp(2*logfvec+logPf_d)
    logavgf_Pd = np.log(np.dot(dlogf,integ[1:]+integ[:-1]))

    #get Nexp
    Nexp=int(N_mouserep*(1-np.exp(logPd)))

    #sample f1 and n1
    logPf_notd = logrhofvec+notd-np.log(1-np.exp(logPd))
    integ=np.exp(logPf_notd+logfvec)
    f_samples_inds=get_distsample(dlogf*(integ[1:] + integ[:-1]),Nexp).flatten()
    f_samples_inds=np.sort(f_samples_inds)
    f_samples_exp=fvec[f_samples_inds]
    find_vals,f_start_ind,f_counts=np.unique(f_samples_inds,return_counts=True,return_index=True)

    n1_samples=np.zeros((Nexp,),dtype=int)
    n2_samples=np.zeros((Nexp,),dtype=int)
    for it,find in enumerate(find_vals):
        samples=np.random.random(size=f_counts[it]) * (1-Pn_f[find].cdf(0)) + Pn_f[find].cdf(0)
        n1_samples[f_start_ind[it]:f_start_ind[it]+f_counts[it]]=Pn_f[find].ppf(samples)
        samples=np.random.random(size=f_counts[it]) * (1-Pn_f[find].cdf(0)) + Pn_f[find].cdf(0)
        n2_samples[f_start_ind[it]:f_start_ind[it]+f_counts[it]]=Pn_f[find].ppf(samples)
#         tmp=get_distsample(np.exp(logPn_f[find,:]),f_counts[it]).flatten()
#         np.random.shuffle(tmp)
#         n1_samples[f_start_ind[it]:f_start_ind[it]+f_counts[it]]=tmp
#         np.random.shuffle(tmp)
#         n2_samples[f_start_ind[it]:f_start_ind[it]+f_counts[it]]=tmp

    seen=np.logical_or(n1_samples>0, n2_samples>0)
    n1_samples=n1_samples[seen]
    n2_samples=n2_samples[seen] 
    f_samples_exp=f_samples_exp[seen]

    print('sample f and n in D')
    logPn0_f=np.asarray([np.log(dist.pmf(0)) for dist in Pn_f])

    integ=np.exp(2*logPn0_f+logrhofvec+logfvec-logPd+d)
    logPnng0_d=np.log(1-np.dot(dlogf,integ[1:]+integ[:-1]))
    Nimp=int(np.exp(logPd+logPnng0_d+np.log(N_mouserep)))

    logPqx0_f=np.log(1-np.exp(logPn0_f))+logPn0_f
    logPq0x_f=logPn0_f+np.log(1-np.exp(logPn0_f))
    logPqxx_f=np.log(1-np.exp(logPn0_f))+np.log(1-np.exp(logPn0_f)) #since conditionals also normalize
    logPfqx0=logPqx0_f+logrhofvec
    logPfq0x=logPq0x_f+logrhofvec
    logPfqxx=logPqxx_f+logrhofvec

    logPfqx0=logPfqx0-logPd
    logPfq0x=logPfq0x-logPd 
    logPfqxx=logPfqxx-logPd
    # d=(logfvec[np.newaxis,:]<np.log(f0))*(logfvec[np.newaxis,:]+svec[:,np.newaxis]<np.log(f0))
    logPfqx0+=d
    logPfq0x+=d
    logPfqxx+=d

    Pqx0=np.trapz(np.exp(logPfqx0+logfvec),x=logfvec)
    Pq0x=np.trapz(np.exp(logPfq0x+logfvec),x=logfvec)
    Pqxx=np.trapz(np.exp(logPfqxx+logfvec),x=logfvec)

    newPqZ=Pqx0+Pq0x+Pqxx
    print(str(Pqx0)+' '+str(Pq0x)+' '+str(Pqxx))
    logPf_qx0=logPfqx0-np.log(Pqx0) #don't add renormalization yet since it would cancel here anyway
    logPf_q0x=logPfq0x-np.log(Pq0x)
    logPf_qxx=logPfqxx-np.log(Pqxx)

    Pqx0/=newPqZ
    Pq0x/=newPqZ
    Pqxx/=newPqZ
    #don't renormalize Pfsq... since no longer needed

    #Sample-----------------------------------------------------------------------------

    #quadrant, q={qx0,q0x,qxx}
    seed1=1
    np.random.seed=1.
    q_samples=np.random.choice(range(3), Nimp, p=(Pqx0,Pq0x,Pqxx))
    vals,counts=np.unique(q_samples,return_counts=True)
    num_qx0=counts[0]
    num_q0x=counts[1]
    num_qxx=counts[2]
    print('q samples: '+str(sum(counts))+' '+str(num_qx0)+' '+str(num_q0x)+' '+str(num_qxx))
    print('q sampled probs: '+str(num_qx0/float(sum(counts)))+' '+str(num_q0x/float(sum(counts)))+' '+str(num_qxx/float(sum(counts))))
    print('q act probs: '+str(Pqx0)+' '+str(Pq0x)+' '+str(Pqxx))

    #f,n1 in x0    
    integ=np.exp(logPf_qx0+logfvec)
    f_samples_inds=get_distsample((dlogf*(integ[1:] + integ[:-1])),num_qx0).flatten()
    f_samples_inds=np.asarray(f_samples_inds,dtype=int) # +1   #handles blip ???????????????
    f_sorted_inds=np.argsort(f_samples_inds)
    f_samples_inds=f_samples_inds[f_sorted_inds] 
    qx0_f_samples=fvec[f_samples_inds]
    find_vals,f_start_ind,f_counts=np.unique(f_samples_inds,return_counts=True,return_index=True)
    qx0_samples=np.zeros((num_qx0,))
    qx0_m_samples=np.zeros((num_qx0,))
    for it,find in enumerate(find_vals):
        samples=np.random.random(size=f_counts[it]) * (1-Pn_f[find].cdf(0)) + Pn_f[find].cdf(0)
        qx0_samples[f_start_ind[it]:f_start_ind[it]+f_counts[it]]=Pn_f[find].ppf(samples)
    qx0_pair_samples=np.hstack((qx0_samples[:,np.newaxis],np.zeros((num_qx0,1)))) 

    #f,s,n2 in 0x
    integ=np.exp(logPf_q0x+logfvec)
    f_samples_inds=get_distsample((dlogf*(integ[1:] + integ[:-1])),num_q0x).flatten()      
    f_samples_inds=np.sort(f_samples_inds)
    find_vals,f_start_ind,f_counts=np.unique(f_samples_inds,return_counts=True,return_index=True)
    q0x_samples=np.zeros((num_q0x,))
    q0x_f_samples=fvec[f_samples_inds]
    for it,find in enumerate(find_vals):
        samples=np.random.random(size=f_counts[it]) * (1-Pn_f[find].cdf(0)) + Pn_f[find].cdf(0)
        q0x_samples[f_start_ind[it]:f_start_ind[it]+f_counts[it]]=Pn_f[find].ppf(samples)
    q0x_pair_samples=np.hstack((np.zeros((num_q0x,1)),q0x_samples[:,np.newaxis]))

    #f,s,n1,n2 in xx\
    integ=np.exp(logPf_qxx+logfvec)
    f_samples_inds=get_distsample((dlogf*(integ[1:] + integ[:-1])),num_qxx).flatten() 
    f_samples_inds=np.sort(f_samples_inds)
    find_vals,f_start_ind,f_counts=np.unique(f_samples_inds,return_counts=True,return_index=True)
    qxx_f_samples=fvec[f_samples_inds]
    qxx_n1_samples=np.zeros((num_qxx,))
    qxx_n2_samples=np.zeros((num_qxx,))
    for it,find in enumerate(find_vals):
            samples=np.random.random(size=f_counts[it]) * (1-Pn_f[find].cdf(0)) + Pn_f[find].cdf(0)
            qxx_n1_samples[f_start_ind[it]:f_start_ind[it]+f_counts[it]]=Pn_f[find].ppf(samples)
            samples=np.random.random(size=f_counts[it]) * (1-Pn_f[find].cdf(0)) + Pn_f[find].cdf(0)
            qxx_n2_samples[f_start_ind[it]:f_start_ind[it]+f_counts[it]]=Pn_f[find].ppf(samples)
    qxx_pair_samples=np.hstack((qxx_n1_samples[:,np.newaxis],qxx_n2_samples[:,np.newaxis]))

    f_samples_imp=np.concatenate((qx0_f_samples,q0x_f_samples,qxx_f_samples))

    pair_samples_imp=np.vstack((qx0_pair_samples,q0x_pair_samples,qxx_pair_samples))
    pair_samples=np.vstack((pair_samples_imp,np.hstack((n1_samples[:,np.newaxis],n2_samples[:,np.newaxis]))))

    f_samples_eff=np.concatenate((f_samples_exp,f_samples_imp))
    
    return f_samples_eff,pair_samples
