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

output_path='../output/syn_data/'
fig,ax=pl.subplots(1,1)
for trial in range(10):
    outstruct=np.load(output_path+'v1_N1e9_test3outstruct_v1_N1e9_test3_'+str(trial)+'.npy').item()
    optparas=outstruct.x
    ax.scatter(np.power(10,optparas[1]),np.power(10,optparas[0]))
ax.set_xlim(0,5)
ax.set_ylim(1e-4,1e0)
ax.set_yscale('log')

output_path='../output/syn_data/'
fig,ax=pl.subplots(1,1)
for trial in range(10):
    outstruct=np.load(output_path+'v1_N1e9_test3outstruct_v1_N1e9_test3_'+str(trial)+'.npy').item()
    optparas=outstruct.x
    ax.scatter(optparas[0],optparas[1])
    uni1=np.load('/home/max/Dropbox/scripts/Projects/immuno/diffexpr/output/syn_data/v1_N1e9_test3unicountvals_1_d'+str(trial)+'.npy')
    uni2=np.load('/home/max/Dropbox/scripts/Projects/immuno/diffexpr/output/syn_data/v1_N1e9_test3unicountvals_2_d'+str(trial)+'.npy')
    indn1=np.load('/home/max/Dropbox/scripts/Projects/immuno/diffexpr/output/syn_data/v1_N1e9_test3indn1_d'+str(trial)+'.npy')
    indn2=np.load('/home/max/Dropbox/scripts/Projects/immuno/diffexpr/output/syn_data/v1_N1e9_test3indn2_d'+str(trial)+'.npy')
    shift=np.load('/home/max/Dropbox/scripts/Projects/immuno/diffexpr/output/syn_data/v1_N1e9_test3shift_v1_N1e9_test3_'+str(trial)+'.npy')
    print(shift)
    ax.plot(uni1[indn1],uni2[indn2],'o')
    ax.set_yscale('log')
    ax.set_xscale('log')
ax.plot(ax.get_xlim(),ax.get_xlim(),'k-')

from importlib import reload
import infer_diffexpr_lib
reload(infer_diffexpr_lib)
from infer_diffexpr_lib import get_Ps_pm,get_rhof,get_distsample
from functools import partial
from copy import deepcopy

# +
# script paras
nfbins=800        #nuber of frequency bins
smax = 25.0  #maximum absolute logfold change value
s_step =0.1 #logfold change step size

NreadsI=2e6
NreadsII=NreadsI
Nclones=int(1e8)
alpha_rho=-2.05
freq_dtype='float32'
def fmin_func(logfmin,alpha_rho,nfbins,Nclones):
    fmin=np.power(10.,logfmin)
    logrhofvec,logfvec = get_rhof(alpha_rho,nfbins,fmin,freq_dtype)
    dlogf=np.diff(logfvec)/2.
    integ=np.exp(logrhofvec+2*logfvec,dtype='float64')
    return np.exp(np.log(Nclones)+np.log(np.sum(dlogf*(integ[1:] + integ[:-1]))))-1

fmin_func_part=partial(fmin_func,alpha_rho=alpha_rho,nfbins=nfbins,Nclones=Nclones)

from scipy.optimize import fsolve

logfmin_guess = -9
logfmin_sol= fsolve(fmin_func_part, logfmin_guess)
fmin=np.power(10.,logfmin_sol[0])
logfmin=np.log(fmin)
logfmax=1
paras_null=[alpha_rho,np.log10(fmin)]
paras=deepcopy(paras_null)

logrhofvec,logfvec = get_rhof(alpha_rho,nfbins,fmin,freq_dtype)
dlogf=np.diff(logfvec)/2.
#svec
logf_step=logfvec[1] - logfvec[0] #use natural log here since f2 increments in increments in exp().
f2s_step=int(round(s_step/logf_step)) #rounded number of f steps in 1 s step
s_step_old=s_step
s_step=float(f2s_step)*logf_step
smax=s_step*(smax/s_step_old)
svec=s_step*np.arange(0,int(round(smax/s_step)+1)) #(0,smax,int(round(smax/stp)))   
svec=np.append(-svec[1:][::-1],svec)

#fvecwide
smaxind=(len(svec)-1)/2
logfmin=logfvec[0]-f2s_step*smaxind*logf_step
logfmax=logfvec[-1]+f2s_step*smaxind*logf_step    
fvecwide=np.exp(np.linspace(logfmin,logfmax,len(logfvec)+2*smaxind*f2s_step))
################################sample
#np.random.seed(trial+1)
#integ=np.exp(np.log(rhofvec)+logfvec)
integ=np.exp(logrhofvec+logfvec)
logfsamples=logfvec[get_distsample(dlogf*(integ[:-1]+integ[1:]),Nclones)]

alp=0.01
realsbar=1.0
bet=1.0
logf2samples=logfsamples+np.random.permutation(svec[get_distsample(get_Ps_pm(alp,bet,realsbar,realsbar,smax,s_step),Nclones)])

Zf=np.sum(np.exp(logfsamples))
print('f1norm='+str(Zf))
n1_samples=np.array(np.random.poisson(lam=NreadsI*np.exp(logfsamples)),dtype='uint32')
Zfp=np.sum(np.exp(logf2samples))
print('f2norm='+str(Zfp))
logf2samples-=np.log(Zfp)-np.log(Zf)
n2_samples=np.array(np.random.poisson(lam=NreadsI*np.exp(logf2samples)),dtype='uint32')
seen=np.logical_or(n1_samples>0,n2_samples>0)
n1_samples=n1_samples[seen]
n2_samples=n2_samples[seen]

# +
fig,ax=pl.subplots(1,2,figsize=(25,10))
counts,bins=np.histogram(s_samples,svec)
ax[0].plot(bins[:-1],counts)
ax[0].set_yscale('log')
ax[0].set_ylim(1e0,1e10)

counts,bins=np.histogram(logf2samples,np.log(fvecwide))
ax[1].plot(bins[:-1],counts,'v')
ax[1].set_yscale('log')
ax[1].set_ylim(1e0,1e10)
counts,bins=np.histogram(logfsamples,logfvec)
ax[1].plot(bins[:-1],counts)
ax[1].set_yscale('log')
ax[1].set_ylim(1e0,1e10)
# -

# Run 1 version of a grid in alp and sbar

import importlib
import pandas as pd
import infer_diffexpr_lib
importlib.reload(infer_diffexpr_lib)
from infer_diffexpr_lib import get_distsample,get_Pn1n2_s, get_sparserep, get_distsample,import_data,PoisPar,NegBinPar,NegBinParMtr,get_Ps_pm,get_rhof
from infer_diffexpr_lib import get_Ps
import numpy.polynomial.polynomial as poly
from scipy.optimize import minimize


# +

case=3
st=time.time()
# #fvecwide
# smaxind=(len(svec)-1)/2
# logfmin=np.log(fvec[0 ])-f2s_step*smaxind*logf_step
# logfmax=np.log(fvec[-1])+f2s_step*smaxind*logf_step
# fvecwide=np.exp(np.linspace(logfmin,logfmax,len(fvec)+2*smaxind*f2s_step))


#real sbar loop
ntrials=2
realsbarvec=np.ones((ntrials,))#np.power(2.,np.arange(0,2))
nsbarpoints=21 #does this need to be so high?
LlandStore = np.zeros((len(realsbarvec),nsbarpoints,nsbarpoints))

shiftMtr =np.zeros((len(realsbarvec),nsbarpoints,nsbarpoints))
fexpsMtr =np.zeros((len(realsbarvec),nsbarpoints,nsbarpoints))
nit_list =np.zeros((len(realsbarvec),nsbarpoints,nsbarpoints))
Zstore =np.zeros((len(realsbarvec),nsbarpoints,nsbarpoints))
Zdashstore =np.zeros((len(realsbarvec),nsbarpoints,nsbarpoints))
LSurface=np.zeros((len(realsbarvec),nsbarpoints,nsbarpoints))
sbarest=np.zeros((len(realsbarvec),))

Zf_data_store=np.zeros((len(realsbarvec),))
Zfp_data_store=np.zeros((len(realsbarvec),))

NsampclonesStore=np.zeros((len(realsbarvec),nsbarpoints,nsbarpoints))
logPnn0Store=np.zeros((len(realsbarvec),nsbarpoints,nsbarpoints))
logavgf_n0n0Store=np.zeros((len(realsbarvec),nsbarpoints,nsbarpoints))
logavgfStore=np.zeros((len(realsbarvec),nsbarpoints,nsbarpoints))
logavgfexps_n0n0Store=np.zeros((len(realsbarvec),nsbarpoints,nsbarpoints))
logavgfexpsStore=np.zeros((len(realsbarvec),nsbarpoints,nsbarpoints))

alp=0.01#Z_p/(1+Z_p)
bet=0
for rit,realsbar in enumerate(realsbarvec):
    print(realsbar)
    
    sbar_m=realsbar
    sbar_p=realsbar
    logPsvec=np.log(get_Ps_pm(alp,bet,realsbar,realsbar,smax,s_step))

    #sample from model
    st=time.time()
    
    integ=np.exp(logrhofvec+logfvec)
    logfsamples=logfvec[get_distsample(dlogf*(integ[:-1]+integ[1:]),Nclones)]

    alp=0.01
    realsbar=1.0
    bet=1.0
    logf2samples=logfsamples+np.random.permutation(svec[get_distsample(get_Ps_pm(alp,bet,realsbar,realsbar,smax,s_step),Nclones)])

    Zf=np.sum(np.exp(logfsamples))
    print('f1norm='+str(Zf))
    n1_samples=np.array(np.random.poisson(lam=NreadsI*np.exp(logfsamples)),dtype='uint32')
    Zfp=np.sum(np.exp(logf2samples))
    print('f2norm='+str(Zfp))
    logf2samples-=np.log(Zfp)-np.log(Zf)
    n2_samples=np.array(np.random.poisson(lam=NreadsI*np.exp(logf2samples)),dtype='uint32')
    seen=np.logical_or(n1_samples>0,n2_samples>0)
    n1_samples=n1_samples[seen]
    n2_samples=n2_samples[seen]

    #report--------------------------------------------------------------------------
    print("samples n1: "+str(np.mean(n1_samples))+" | "+str(max(n1_samples))+", n2 "+str(np.mean(n2_samples))+" | "+str(max(n2_samples)))
    et=time.time()
    print("sampling elapsed: "+str(round(et-st)))

    #compute model over this data set-----------------------------------------------------
    counts_d = pd.DataFrame({'Clone_count_1': n1_samples, 'Clone_count_2': n2_samples})
    indn1_d, indn2_d, countpaircounts_d, unicountvals_1_d, unicountvals_2_d, NreadsI_d, NreadsII_d=get_sparserep(counts_d)
    Nsampclones=np.sum(countpaircounts_d)
    print("num n1: "+str(len(unicountvals_1_d))+ ' Nreads:'+str(NreadsI_d))
    print("num n2: "+str(len(unicountvals_2_d))+ ' Nreads:'+str(NreadsII_d))

    Nreadsvec=(float(NreadsI),float(NreadsII))

    #compute Pn1_f
    it=0
    unicounts=deepcopy(unicountvals_1_d)

    mean_n=Nreadsvec[it]*np.exp(logfvec)
    Pn_f=PoisPar(mean_n,unicounts)
    logPn1_f=np.log(Pn_f)#[:,uinds1]) #throws warning


    #get likelihood over range of sbars around the actual to see if likelihood of model acheives it's maximum there
    first_shift=0
    sbarvec=np.logspace(-0.4,0.4,nsbarpoints)#np.power(2.,np.arange(-3,4)) #(0.5, 1.0)#(0.5,1.0,1.5)#
    alpvec=np.logspace(-4,0,nsbarpoints)#np.logspace(-4,-2,8)#np.asarray((0.001,0.1),dtype=float)#np.power(2.,np.arange(-,)) #(0.5, 1.0)#(0.5,1.0,1.5)#

    for sbarit,sbar in enumerate(sbarvec):
        print(sbarit)
        shift=deepcopy(first_shift)
        smit_flag=True
        addshift=0
        for ait,alptest in enumerate(alpvec):
            

            Ps = get_Ps_pm(alptest,bet,sbar,sbar,smax,s_step)
            logPsvec=np.log(Ps)

            shift_flag=True
            if shift_flag:
                tol=1e-5
                diffval=np.Inf
#                     addshift=0
                it=0
                Z=0
                Zdash=0
                while diffval>tol:
                    it+=1
                    shift+=addshift
                    print(str(it)+' '+str(shift)+' '+str(diffval)+' Z: '+str(Z)+' '+str(Zdash))
                    fvecwide_shift=np.exp(np.log(fvecwide)-shift) #implements shift in Pn2_fs
                    svec_shift=svec-shift
                    unicounts=deepcopy(unicountvals_2_d)
                    Pn2_f=np.zeros((len(fvecwide_shift),len(unicounts)))
                    if case==0:
                        mean_m=m_total*fvecwide_shift
                        var_m=mean_m+beta_mv*np.power(mean_m,alpha_mv)
                        Poisvec=PoisPar(mvec*r_cvec[1],unicounts)
                        for f_it in range(len(fvecwide_shift)):
                            NBvec=NegBinPar(mean_m[f_it],var_m[f_it],mvec)
                            for n_it,n in enumerate(unicounts):
                                Pn2_f[f_it,n_it]=np.dot(NBvec[m_low[n_it]:m_high[n_it]+1],Poisvec[m_low[n_it]:m_high[n_it]+1,n_it]) 
                    elif case==1:
                        Poisvec=PoisPar(m_total*fvecwide_shift,mvec)
                        mean_n=r_cvec[1]*mvec
                        NBmtr=NegBinParMtr(mean_n,mean_n+beta_mv*np.power(mean_m,alpha_mv),unicounts)
                        for f_it in range(len(fvecwide_shift)):
                            Poisvectmp=Poisvec[f_it,:]
                            for n_it,n in enumerate(unicounts):
                                Pn2_f[f_it,n_it]=np.dot(Poisvectmp[m_low[n_it]:m_high[n_it]+1],NBmtr[m_low[n_it]:m_high[n_it]+1,n_it]) 
                    elif case==2:
                        mean_n=Nreadsvec[1]*fvecwide_shift
                        var_n=mean_n+beta_mv*np.power(mean_n,alpha_mv)
                        Pn2_f=NegBinParMtr(mean_n,var_n,unicounts)
                    else:# case==3:
                        mean_n=Nreadsvec[1]*fvecwide_shift
                        Pn2_f=PoisPar(mean_n,unicounts)
                    logPn2_f=Pn2_f
                    logPn2_f=np.log(logPn2_f)
                    #logPn2_s=np.zeros((len(svec),nfbins,len(unicounts))) #svec is svec_shift
                    #for s_it in range(len(svec)):
                        #logPn2_s[s_it,:,:]=logPn2_f[f2s_step*s_it:f2s_step*s_it+nfbins,:]   #note here this is fvec long

                    log_Pn2_f=np.zeros((len(logfvec),len(unicountvals_2_d)))
                    for s_it in range(len(svec)):
                        log_Pn2_f+=np.exp(logPn2_f[f2s_step*s_it:f2s_step*s_it+nfbins,:]+logPsvec[s_it,np.newaxis,np.newaxis])
                    log_Pn2_f=np.log(log_Pn2_f) #ok
                    #log_Pn2_f=np.log(np.sum(np.exp(logPn2_s+logPsvec[:,np.newaxis,np.newaxis]),axis=0))
                    integ=np.exp(log_Pn2_f[:,indn2_d]+logPn1_f[:,indn1_d]+logrhofvec[:,np.newaxis]+logfvec[:,np.newaxis])
                    log_Pn1n2=np.log(np.sum(dlogf[:,np.newaxis]*(integ[1:,:] + integ[:-1,:]),axis=0))

                    #log_Pn2_f=np.log(np.sum(np.exp(logPn2_s+logPsvec[:,np.newaxis,np.newaxis]),axis=0))
                    integ=np.exp(np.log(integ)+logfvec[:,np.newaxis]) 
                          #np.exp(log_Pn2_f[:,indn2]+logPn1_f[:,indn1]+logrhofvec[:,np.newaxis]+logfvec[:,np.newaxis]+logfvec[:,np.newaxis])
                    tmp=deepcopy(log_Pn1n2)
                    tmp[tmp==-np.Inf]=np.Inf
                    avgf_n1n2=    np.exp(np.log(np.sum(dlogf[:,np.newaxis]*(integ[1:,:] + integ[:-1,:]),axis=0))-tmp)
                    log_avgf=np.log(np.dot(countpaircounts_d,avgf_n1n2))

                    log_expsavg_Pn2_f=np.zeros((len(logfvec),len(unicountvals_2_d)))
                    for s_it in range(len(svec)):
                        log_expsavg_Pn2_f+=np.exp(svec_shift[s_it,np.newaxis,np.newaxis]+logPn2_f[f2s_step*s_it:f2s_step*s_it+nfbins,:]+logPsvec[s_it,np.newaxis,np.newaxis]) #-------------svec_shift?
                    log_expsavg_Pn2_f=np.log(log_expsavg_Pn2_f)
                    #log_expsavg_Pn2_f=np.log(np.sum(np.exp(svec[:,np.newaxis,np.newaxis]+logPn2_s+logPsvec[:,np.newaxis,np.newaxis]),axis=0))
                    integ=np.exp(log_expsavg_Pn2_f[:,indn2_d]+logPn1_f[:,indn1_d]+logrhofvec[:,np.newaxis]+2*logfvec[:,np.newaxis])
                    avgfexps_n1n2=np.exp(np.log(np.sum(dlogf[:,np.newaxis]*(integ[1:,:] + integ[:-1,:]),axis=0))-tmp)
                    log_avgfexps=np.log(np.dot(countpaircounts_d,avgfexps_n1n2))

                    logPn20_s=np.zeros((len(svec),len(logfvec))) #svec is svec_shift
                    for s_it in range(len(svec)):
                        logPn20_s[s_it,:]=logPn2_f[f2s_step*s_it:f2s_step*s_it+len(logfvec),0]   #note here this is fvec long on shifted s
                    log_Pnn0_fs=logPn1_f[np.newaxis,:,0]+logPn20_s
                    log_Pfsnn0=log_Pnn0_fs+logrhofvec[np.newaxis,:]+logPsvec[:,np.newaxis]
                    log_Pfsnng0=np.log(1-np.exp(log_Pnn0_fs))+logrhofvec[np.newaxis,:]+logPsvec[:,np.newaxis]
                    log_Pfnn0=np.log(np.sum(np.exp(log_Pfsnn0),axis=0))
                    integ=np.exp(log_Pfnn0+logfvec)
                    logPnn0=np.log(np.sum(dlogf*(integ[1:]+integ[:-1])))

                    log_Pnng0=np.log(1-np.exp(logPnn0))
                    log_Pfs_nng0=log_Pfsnng0-log_Pnng0

                    #decomposed f averages
                    integ = np.exp(logPn1_f[:,0]+logrhofvec+2*logfvec+np.log(np.sum(np.exp(logPn20_s+logPsvec[:,np.newaxis]),axis=0)))
                    log_avgf_n0n0 = np.log(np.dot(dlogf,integ[1:]+integ[:-1]))

                    #decomposed fexps averages
                    integ = np.exp(logPn1_f[:,0]+logrhofvec+2*logfvec+np.log(np.sum(np.exp(svec_shift[:,np.newaxis]+logPn20_s+logPsvec[:,np.newaxis]),axis=0)))  #----------svec
                    log_avgfexps_n0n0 = np.log(np.dot(dlogf,integ[1:]+integ[:-1]))

    #                 Z     = np.exp(logPnn0+ log_avgf_n0n0    ) + np.exp( log_avgf    )
    #                 Zdash = np.exp(logPnn0+ log_avgfexps_n0n0) + np.exp( log_avgfexps)
    #                 Nclones=Nsampclones
                    logNclones=np.log(Nsampclones)-log_Pnng0
                    Z     = np.exp(logNclones+logPnn0+ log_avgf_n0n0    ) + np.exp(log_avgf    )
                    Zdash = np.exp(logNclones+logPnn0+ log_avgfexps_n0n0) + np.exp(log_avgfexps)
    #                 Z     = np.exp(logNclones + logPnn0 + log_avgf_n0n0    ) + np.exp(log_avgf     - log_Pnng0) # Ncl P(0,0) <f>_p|00  + sum <f>_p(n,n')/(1-P00)
    #                 Zdash = np.exp(logNclones + logPnn0 + log_avgfexps_n0n0) + np.exp(log_avgfexps - log_Pnng0)

                    NsampclonesStore[rit,sbarit,ait]=Nsampclones
                    logPnn0Store[rit,sbarit,ait]=logPnn0
                    logavgf_n0n0Store[rit,sbarit,ait]=log_avgf_n0n0
                    logavgfStore[rit,sbarit,ait]=log_avgf
                    logavgfexps_n0n0Store[rit,sbarit,ait]=log_avgfexps_n0n0
                    logavgfexpsStore[rit,sbarit,ait]=log_avgfexps

                    addshiftold=deepcopy(addshift)
                    addshift=np.log(Zdash)-np.log(Z)
                    diffval=np.fabs(addshift-addshiftold)

                shiftMtr[rit,sbarit,ait]=shift
                nit_list[rit,sbarit,ait]=it
                Zstore[rit,sbarit,ait]=Z
                Zdashstore[rit,sbarit,ait]=Zdash

            else:
                fvecwide_shift=np.exp(np.log(fvecwide)-np.log(Z))#_data)) #implements shift in Pn2_fs
                mean_n=Nreadsvec[1]*fvecwide_shift
                if case==2:
                    var_n=mean_n+beta_mv*np.power(mean_n,alpha_mv)
                    Pn2_f=NegBinParMtr(mean_n,var_n,unicountvals_2_d)
                else: #case==3
                    Pn2_f=PoisPar(mean_n,unicountvals_2_d)
                logPn2_f=np.log(Pn2_f)

                #compute joint
                log_Pn2_f=np.zeros((len(logfvec),len(unicountvals_2_d)))
                for s_it in range(len(svec)):
                    log_Pn2_f+=np.exp(logPn2_f[f2s_step*s_it:f2s_step*s_it+nfbins,:]+logPsvec[s_it,np.newaxis,np.newaxis])
                log_Pn2_f=np.log(log_Pn2_f) #ok
                #log_Pn2_f=np.log(np.sum(np.exp(logPn2_s+logPsvec[:,np.newaxis,np.newaxis]),axis=0))
                integ=np.exp(log_Pn2_f[:,indn2_d]+logPn1_f[:,indn1_d]+logrhofvec[:,np.newaxis]+logfvec[:,np.newaxis])
                log_Pn1n2=np.log(np.sum(dlogf[:,np.newaxis]*(integ[1:,:] + integ[:-1,:]),axis=0))


                integ=np.zeros((len(logfvec),))
                for s_it in range(len(svec)):
                    integ+=np.exp(logPn2_f[f2s_step*s_it:f2s_step*s_it+nfbins,0]+logPsvec[np.newaxis,s_it])
                integ = np.exp(logPn1_f[:,0]+logrhofvec+logfvec+np.log(integ))
                logPnn0 = np.log(np.dot(dlogf,integ[1:]+integ[:-1]))

            tmp=np.exp(log_Pn1n2-np.log(1-np.exp(logPnn0))) #renormalize
            #print(np.dot(countpaircounts_d/float(clonesum),np.where(tmp>0,np.log(tmp),0)))
            LSurface[rit,sbarit,ait]=np.dot(countpaircounts_d/float(Nsampclones),np.where(tmp>0,np.log(tmp),0))
            ets=time.time()

            if smit_flag:
                first_shift=deepcopy(shift)
                smit_flag=False

print(time.time()-st)
# -

np.save('shiftMtr.npy',shiftMtr)
np.save('LSurface.npy',LSurface)
np.save('Zfp_data_store.npy',Zfp_data_store)
np.save('Zf_data_store.npy',Zf_data_store)


# +
def MSError_fcn(para,data,diag_pair):
    M=np.diag(diag_pair)
    M[0,1]=para
    M[1,0]=para
    res=0
    for val in data.itertuples():
        res+=np.power(val.y-M.dot(val.x).dot(val.x)/2,2)
    return res/len(data)

def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]


# +
#iterate over trials
fig,axarr=pl.subplots(1,2)
outstructs=list()
# for trial in range(50):
trial=1
#load in 5x5 likelihood surface
Nc=np.sum(countpaircounts_d)#np.load('../infer_sim/countpaircounts_d'+str(trial)+'.npy'))
LSurfacetmp=LSurface[trial,:,:]#np.load('../infer_sim/LSurface_'+str(trial)+'.npy')*Nc
parasopt=(alp,realsbar)#np.load('../infer_sim/outstruct_v1_test3_'+str(trial)+'.npy').flatten()[0].x
npoints=len(LSurfacetmp)                
cind=int((npoints-1)/2)
logLikelihood_NBPois_diag=np.zeros((len(parasopt),npoints))
for pit in range(len(parasopt)):
    if pit==0:
        logLikelihood_NBPois_diag[pit,:]=LSurfacetmp[:,cind]
    else:
        logLikelihood_NBPois_diag[pit,:]=LSurfacetmp[cind,:]
        
coeffs=np.zeros((len(parasopt),3))
lowfac=0.95
for pit,paraopt in enumerate(parasopt):
    paravec=np.linspace(paraopt*lowfac,paraopt*(2-lowfac),npoints)
    coeffs[pit,:]=poly.polyfit(paravec-paravec[cind], -(logLikelihood_NBPois_diag[pit,:]-logLikelihood_NBPois_diag[pit,cind]), 2)
    fval=-(coeffs[pit,2]*(paravec-paravec[cind])**2+coeffs[pit,1]*(paravec-paravec[cind])+coeffs[pit,0])
    axarr[pit].plot(paravec-paravec[cind],fval)#,color=axtmp[-1].get_color())
    axarr[pit].plot(paravec-paravec[cind],logLikelihood_NBPois_diag[pit,:]-logLikelihood_NBPois_diag[pit,cind])#,color=axtmp[-1].get_color())
diag_entries=2*coeffs[:,2]

likelihoodtmp=LSurfacetmp
pair_ind=(0,1)
para1vec=np.linspace(parasopt[pair_ind[0]]*lowfac,parasopt[pair_ind[0]]+parasopt[pair_ind[0]]*(1-lowfac),npoints)
para2vec=np.linspace(parasopt[pair_ind[1]]*lowfac,parasopt[pair_ind[1]]+parasopt[pair_ind[1]]*(1-lowfac),npoints)
data_df=pd.DataFrame()
for pit1,para1 in enumerate(para1vec):
    for pit2,para2 in enumerate(para2vec):
        data_df=data_df.append(pd.Series({'x':np.array([para1-para1vec[cind],para2-para2vec[cind]]),'y':-(likelihoodtmp[pit1,pit2]-likelihoodtmp[cind,cind])}),ignore_index=True)

partial_MSError_fcn=partial(MSError_fcn,data=data_df,diag_pair=diag_entries)
initparas=1.
outstruct_quad = minimize(partial_MSError_fcn, initparas, method='Nelder-Mead', tol=1e-10,options={'ftol':1e-10 ,'disp': False,'maxiter':200})
#             outstruct_quad = minimize(partial_MSError_fcn, initparas, method='SLSQP', tol=1e-10,options={'ftol':1e-10 ,'disp': False,'maxiter':200})

M=np.diag(diag_entries)
M[pair_ind[0],pair_ind[1]]=outstruct_quad.x
M[pair_ind[1],pair_ind[0]]=outstruct_quad.x
# M[0,0]=M[0,0]*2
#plot
axtmp=axarr[pair_ind[1]].plot(para2vec-para2vec[cind],likelihoodtmp[cind,:]-likelihoodtmp[cind,cind],'x')
axtmp=axarr[pair_ind[0]].plot(para1vec-para1vec[cind],likelihoodtmp[:,cind]-likelihoodtmp[cind,cind],'x')
ndense=51
cpdense=int((ndense-1)/2)
fval=np.zeros(ndense)
para2vec_dense=np.linspace(parasopt[pair_ind[1]]*lowfac,parasopt[pair_ind[1]]*(2-lowfac),ndense)
for pit2,para2 in enumerate(para2vec_dense):
    x=np.array([0,para2-para2vec_dense[cpdense]])
    fval[pit2]=-M.dot(x).dot(x)/2
axarr[pair_ind[1]].plot(para2vec_dense-para2vec[cind],fval,color=axtmp[-1].get_color())

fval=np.zeros(ndense)
para1vec_dense=np.linspace(parasopt[pair_ind[0]]*lowfac,parasopt[pair_ind[0]]*(2-lowfac),ndense)
for pit1,para1 in enumerate(para1vec_dense):
    x=np.array([para1-para1vec_dense[cpdense],0])
    fval[pit1]=-M.dot(x).dot(x)/2
axarr[pair_ind[0]].plot(para1vec_dense-para1vec[cind],fval,color=axtmp[-1].get_color())

#         print(pd.DataFrame(M))

Cov=np.linalg.inv(M)
e_val,e_vec=np.linalg.eig(Cov.T)

# +
# fig2, ax2 = pl.subplots(1, 1,figsize=(6,6))
sbarstar=realsbar
alpstar=alp
x = sbarstar
y = alpstar
nstd = 1
cov = np.flipud(np.fliplr(Cov))
vals, vecs = eigsorted(cov)
theta = -np.degrees(np.arctan2(*vecs[:,0][::-1]))
w, h = 2 * nstd * np.sqrt(vals)
ell = mpl.patches.Ellipse(xy=(np.mean(x), np.mean(y)),
              width=w, height=h,
              angle=theta, color='black',linewidth=2,linestyle='-',zorder=10)
ell.set_facecolor('None')

fig, ax = pl.subplots(1, 1,figsize=(3,3))

#shift heatmap
shift_pl=np.log10(np.exp(shiftMtr[0,:,:])-1)
X, Y = np.meshgrid(np.log10(sbarvec),np.log10(alpvec))
ax.contour(X, Y, shift_pl,levels = [np.log10(0.5)],colors=('k',),linestyles=('--',),linewidths=(2,),zorder=10)
CS=ax.contourf(X,Y,shift_pl,range(-1,5),cmap=pl.cm.gray_r)
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar
ax_div=make_axes_locatable(ax)
cax=ax_div.append_axes("top",size='7%',pad='2%')
cb=colorbar(CS,cax=cax,orientation='horizontal') # draw colorbar
cax.xaxis.set_ticks_position("top")
cax.xaxis.set_ticks(range(-1,4))
cax.xaxis.set_ticklabels([r'$10^{'+str(int(x))+'}$' for x in range(-1,4)])

Ltmp=deepcopy(LSurface[trial,:,:])
minval=np.min(Ltmp)
Ltmp[Ltmp<minval]=minval
Ltmp[Ltmp==0]=minval
Lmax=np.max(Ltmp)
z=Ltmp
sbarvec=np.logspace(-0.4,0.4,nsbarpoints)#np.power(2.,np.arange(-3,4)) #(0.5, 1.0)#(0.5,1.0,1.5)#
alpvec=np.logspace(-4,0,nsbarpoints)#np.logspace(-4,-2,8)#np.asarray((0.001,0.1),dtype=float)#np.power(2.,np.arange(-,)) #(0.5, 1.0)#(0.5,1.0,1.5)#
x,y=np.meshgrid(np.log10(sbarvec),np.log10(alpvec))
# CS = ax.contour(x,y,z,ncontours,linewidths=0.5,colors='k')
# CS=ax.contourf(x,y,z,ncontours,cmap=pl.cm.viridis)
# fig.colorbar(CS) # draw colorbar
lwtmp=1
ax.contour(x,y,z,Lmax-np.logspace(-4,0,10)[::-1],linewidths=1,colors='gray',linestyles='-')
ax.set_xlim(np.log10(sbarvec[0]), np.log10(sbarvec[-1]))
ax.set_ylim(np.log10(alpvec[0]), np.log10(alpvec[-1]))
#     ax.plot([-0.2,0.4],[0,(-0.8)*4.5],'k--',lw=lwtmp)
#     ax.set_title(r'$Zp_f='+str(Zfp_data_store[trial])+',\; '+str(Zf_data_store[trial])+'$')
ax.set_ylabel(r'$\bar{s}$')
ax.set_xlabel(r'$\alpha$')
ax.set_yticks(range(-4,1))
ax.set_yticklabels([r'$10^{'+str(int(x))+'}$' for x in range(-4,0)])
ax.set_xticks(np.linspace(-0.4,0.4,5))
ax.set_xticklabels([r'$10^{'+str(int(x))+'}$' for x in np.linspace(-0.,0.4,5)])
optparas_Store=[]
for tr in range(10):
    outstruct=np.load(output_path+'v1_N1e9_test3outstruct_v1_N1e9_test3_'+str(tr)+'.npy').item()
    optparas_Store.append(list(outstruct.x[::-1]))
# alplist,sbarlist=*zip(*optparas_Store)
ax.scatter(*zip(*optparas_Store),s=100,marker='x',lw=lwtmp,color='gray',label=r'$\hat{\theta}$')
ax.scatter(np.mean(sbarstar),np.mean(alpstar),s=100,marker='x',color='k', lw=lwtmp,zorder=10,label=r'$\bar{\hat{\theta}}$')#,edgecolors='r',facecolors='none')
ax.scatter(0,-2,s=100,marker='.',color='k',edgecolors='k',facecolors='k',zorder=10,label=r'$\theta^*$')
# ax.set_aspect(0.5)
ax.add_artist(ell)
fig.suptitle(r'$\log \left[\mathcal{L}_{\mathrm{max}}-\mathcal{L}\right]\;\mathcal{I}^{-1}(\hat{\theta})$',y=1.1)
#
ax.legend(frameon=False,loc=3)
def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]

#     ax2.contour(x,y,z,[Lmax-1e-5],linewidths=1,colors='b',linestyles='-')
#     ax2.set_xlim(np.log10(sbarvec[0]), np.log10(sbarvec[-1]))
#     ax2.set_ylim(np.log10(alpvec[0]), np.log10(alpvec[-1]))
#     ax2.scatter(np.log10(sbarvec[sitopt]),np.log10(alpvec[aitopt]),s=100,marker='x',lw=2,c='r')
#     ax2.scatter(np.log10(realsbar),np.log10(alp),s=200,marker='o',lw=1,edgecolors='r',facecolors='none')
#     ax2.set_title(r'$Zp_f='+str(Zfp_data_store[trial])+',\; '+str(Zf_data_store[trial])+'$')
# ax.scatter(x,y,marker='o',c='k',s=5)
# fig.savefig('L_contours.pdf',format='pdf',dpi=500,bbox_inches='tight')


# -

optparas_Store


