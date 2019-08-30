import numpy as np
import time
import sys
from infer_diffexpr_lib import import_data, get_Pn1n2_s, get_sparserep, save_table, get_rhof, get_Ps,\
        callbackFdiffexpr,NegBinParMtr,get_Ps_pm,PoisPar,NegBinPar,get_distsample,setup_outpath_and_logs
from functools import partial
from scipy.optimize import minimize
import os
from functools import partial
import pandas as pd
import ctypes
from copy import deepcopy
from scipy import stats
from scipy.stats import poisson
mkl_rt = ctypes.CDLL('libmkl_rt.so')
num_threads=4
mkl_set_num_threads = mkl_rt.MKL_Set_Num_Threads(num_threads)
print(str(num_threads)+' cores available')
shift=0
def get_likelihood_constr(paras,null_paras,svec,smax,s_step,indn1_d,indn2_d,logfvec,fvecwide,logrhofvec,\
                                 unicountvals_1_d,unicountvals_2_d,countpaircounts_d,\
                                 NreadsI, NreadsII, nfbins,f2s_step,\
                                 m_low,m_high,mvec,Nsamp,r_cvec,logPn1_f,case):
    #print('l:'+str(paras[1]))
    dlogf=np.diff(logfvec)/2.
    alpha_rho = null_paras[0]
    if case<2: #case: 0=NB->Pois, 1=Pois->NB, 2=NB, 3=Pois 
        m_total=float(np.power(10, null_paras[3]))
        r_c1=NreadsI/m_total 
        r_c2=NreadsII/m_total     
        r_cvec=(r_c1,r_c2)
        fmin=np.power(10,null_paras[4])
    else:
        fmin=np.power(10,null_paras[-1])

    if case<3:
        beta_mv= null_paras[1]
        alpha_mv=null_paras[2]
    Nreadsvec=(NreadsI,NreadsII)

    #Ps = get_Ps_pm(np.power(10,paras[0]),np.power(10,paras[1]),paras[2],paras[3],smax,s_step)
    Ps = get_Ps_pm(np.power(10,paras[0]),1,np.power(10,paras[1]),np.power(10,paras[1]),smax,s_step)

    logPsvec=np.log(Ps)
    shift=paras[-1]
    

    fvecwide_shift=np.exp(np.log(fvecwide)-shift) #implements shift in Pn2_fs
    svec_shift=svec-shift
    unicounts=unicountvals_2_d
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

    log_Pn2_f=np.zeros((len(fvec),len(unicountvals_2_d)))
    for s_it in range(len(svec)):
        log_Pn2_f+=np.exp(logPn2_f[f2s_step*s_it:f2s_step*s_it+nfbins,:]+logPsvec[s_it,np.newaxis,np.newaxis])
    log_Pn2_f=np.log(log_Pn2_f)
    #log_Pn2_f=np.log(np.sum(np.exp(logPn2_s+logPsvec[:,np.newaxis,np.newaxis]),axis=0))
    integ=np.exp(log_Pn2_f[:,indn2_d]+logPn1_f[:,indn1_d]+logrhofvec[:,np.newaxis]+logfvec[:,np.newaxis])
    log_Pn1n2=np.log(np.sum(dlogf[:,np.newaxis]*(integ[1:,:] + integ[:-1,:]),axis=0))
    
    integ=np.exp(log_Pn2_f[:,0]+logPn1_f[:,0]+logrhofvec+logfvec)
    logPnn0=np.log(np.sum(dlogf*(integ[1:] + integ[:-1]),axis=0))
    
    
    tmp=np.exp(log_Pn1n2-np.log(1-np.exp(logPnn0))) #renormalize
    return -np.dot(countpaircounts_d/float(Nsamp),np.where(tmp>0,np.log(tmp),0))


def constr_fn_diffexpr(paras,null_paras,svec,smax,s_step,indn1_d,indn2_d,logfvec,fvecwide,logrhofvec,\
                            unicountvals_1_d,unicountvals_2_d,countpaircounts_d,\
                            NreadsI, NreadsII, nfbins,f2s_step,\
                            m_low,m_high,mvec,Nsamp,r_cvec,logPn1_f,case):
    #Ps = get_Ps_pm(np.power(10,paras[0]),np.power(10,paras[1]),paras[2],paras[3],smax,s_step)
    Ps = get_Ps_pm(np.power(10,paras[0]),1,np.power(10,paras[1]),np.power(10,paras[1]),smax,s_step)
    #print('c:'+str(paras[1]))

    Nreadsvec=(NreadsI,NreadsII)
    logPsvec=np.log(Ps) 
    shift=paras[-1]

    alpha_rho = null_paras[0]
    if case<2: #case: 0=NB->Pois, 1=Pois->NB, 2=NB, 3=Pois 
        m_total=float(np.power(10, null_paras[3]))
        r_c1=NreadsI/m_total 
        r_c2=NreadsII/m_total      
        r_cvec=(r_c1,r_c2)
        fmin=np.power(10,null_paras[4])
    else:
        fmin=np.power(10,null_paras[-1])

    if case<3:
        beta_mv= null_paras[1]
        alpha_mv=null_paras[2]

    dlogf=np.diff(logfvec)/2.
    fvecwide_shift=np.exp(np.log(fvecwide)-shift) #implements shift in Pn2_fs
    svec_shift=svec-shift
    unicounts=unicountvals_2_d
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
    log_Pn2_f=np.log(log_Pn2_f)
    #log_Pn2_f=np.log(np.sum(np.exp(logPn2_s+logPsvec[:,np.newaxis,np.newaxis]),axis=0))
    integ=np.exp(log_Pn2_f[:,indn2_d]+logPn1_f[:,indn1_d]+logrhofvec[:,np.newaxis]+logfvec[:,np.newaxis])
    log_Pn1n2=np.log(np.sum(dlogf[:,np.newaxis]*(integ[1:,:] + integ[:-1,:]),axis=0))
    #tmp=np.exp(log_Pn1n2-np.log(1-np.exp(logPnn0))) #renormalize
    #log_Pn2_f=np.log(np.sum(np.exp(logPn2_s+logPsvec[:,np.newaxis,np.newaxis]),axis=0))
    integ=np.exp(np.log(integ)+logfvec[:,np.newaxis]) 
    #np.exp(log_Pn2_f[:,indn2]+logPn1_f[:,indn1]+logrhofvec[:,np.newaxis]+logfvec[:,np.newaxis]+logfvec[:,np.newaxis])
    tmp=deepcopy(log_Pn1n2)
    tmp[tmp==-np.Inf]=np.Inf #since subtracted in next line
    avgf_n1n2=    np.exp(np.log(np.sum(dlogf[:,np.newaxis]*(integ[1:,:] + integ[:-1,:]),axis=0))-tmp)
    log_avgf=np.log(np.dot(countpaircounts_d,avgf_n1n2))

    log_expsavg_Pn2_f=np.zeros((len(logfvec),len(unicountvals_2_d)))
    for s_it in range(len(svec)):
        log_expsavg_Pn2_f+=np.exp(svec_shift[s_it,np.newaxis,np.newaxis]+logPn2_f[f2s_step*s_it:f2s_step*s_it+nfbins,:]+logPsvec[s_it,np.newaxis,np.newaxis]) #cuts down on memory constraints
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

    logNclones=np.log(Nsamp)-log_Pnng0
    Z     = np.exp(logNclones + logPnn0 + log_avgf_n0n0    ) + np.exp(log_avgf)    
    Zdash = np.exp(logNclones + logPnn0 + log_avgfexps_n0n0) + np.exp(log_avgfexps)

    return np.log(Zdash)-np.log(Z)

def get_likelihood(paras,null_paras,svec,smax,s_step,indn1_d,indn2_d,logfvec,fvecwide,logrhofvec,\
                                 unicountvals_1_d,unicountvals_2_d,countpaircounts_d,\
                                 NreadsI, NreadsII, nfbins,f2s_step,\
                                 m_low,m_high,mvec,Nsamp,r_cvec,logPn1_f,case):
    global shift
    dlogf=np.diff(logfvec)/2.
    alpha_rho = null_paras[0]
    if case<2: #case: 0=NB->Pois, 1=Pois->NB, 2=NB, 3=Pois 
        m_total=float(np.power(10, null_paras[3]))
        r_c1=NreadsI/m_total 
        r_c2=NreadsII/m_total     
        r_cvec=(r_c1,r_c2)
        fmin=np.power(10,null_paras[4])
    else:
        fmin=np.power(10,null_paras[-1])

    if case<3:
        beta_mv= null_paras[1]
        alpha_mv=null_paras[2]
        
    Nreadsvec=(NreadsI,NreadsII)
    Ps = get_Ps_pm(np.power(10,paras[0]),1,np.power(10,paras[1]),np.power(10,paras[1]),smax,s_step)
    logPsvec=np.log(Ps)
    #shift=0
    addshift=0
    tol=1e-5
    diffval=np.Inf
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
        logNclones=np.log(Nsamp)-log_Pnng0
        Z     = np.exp(logNclones+logPnn0+ log_avgf_n0n0    ) + np.exp(log_avgf    )
        Zdash = np.exp(logNclones+logPnn0+ log_avgfexps_n0n0) + np.exp(log_avgfexps)
#                 Z     = np.exp(logNclones + logPnn0 + log_avgf_n0n0    ) + np.exp(log_avgf     - log_Pnng0) # Ncl P(0,0) <f>_p|00  + sum <f>_p(n,n')/(1-P00)
#                 Zdash = np.exp(logNclones + logPnn0 + log_avgfexps_n0n0) + np.exp(log_avgfexps - log_Pnng0)

        #NsampclonesStore[rit,sbarit,ait]=Nsampclones
        #logPnn0Store[rit,sbarit,ait]=logPnn0
        #logavgf_n0n0Store[rit,sbarit,ait]=log_avgf_n0n0
        #logavgfStore[rit,sbarit,ait]=log_avgf
        #logavgfexps_n0n0Store[rit,sbarit,ait]=log_avgfexps_n0n0
        #logavgfexpsStore[rit,sbarit,ait]=log_avgfexps

        addshiftold=deepcopy(addshift)
        addshift=np.log(Zdash)-np.log(Z)
        diffval=np.fabs(addshift-addshiftold)

    tmp=np.exp(log_Pn1n2-np.log(1-np.exp(logPnn0))) #renormalize
    return -np.dot(countpaircounts_d/float(Nsamp),np.where(tmp>0,np.log(tmp),0))

def main(ntrials):

######################################Preprocessing###########################################3
    case=3
    outpath='../output/syn_data/'
    runname='v1_N1e9_test'+str(case)
    outpath+=runname
    logfilename=runname
    prtfn= setup_outpath_and_logs(outpath,logfilename)
    callbackFdiffexpr_part=partial(callbackFdiffexpr,prtfn=prtfn)
    # script paras
    nfbins=800        #nuber of frequency bins
    smax = 25.0  #maximum absolute logfold change value
    s_step =0.1 #logfold change step size

    NreadsI=2e6
    NreadsII=NreadsI
    Nclones=int(1e9)
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
    prtfn('logfmin: '+str(logfmin_sol))
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
    fvecwide=np.exp(np.linspace(logfmin,logfmax,int(len(logfvec)+2*smaxind*f2s_step)))

    realsbar=1.0
    alp=1e-2
    bet=1.
    sbar_m=realsbar
    sbar_p=realsbar
    Ps = get_Ps_pm(alp,bet,sbar_m,sbar_p,smax,s_step)
    logPsvec=np.log(Ps)

    #sample from model    
    integ=np.exp(logrhofvec+logfvec)
    #dPf = stats.rv_discrete(name='dPf', values=(range(len(fvec[:-1])), dlogf*(integ[1:]+integ[:-1])))
    #dPs = stats.rv_discrete(name='dPs', values=(range(len(svec)), Ps))

    Zf_data_store=np.zeros((ntrials,))
    Zfp_data_store=np.zeros((ntrials,))
    
    for trial in range(ntrials):
        prtfn('trial '+str(trial))
        logrhofvec,logfvec = get_rhof(alpha_rho,nfbins,fmin,freq_dtype)
        dlogf=np.diff(logfvec)/2.
        fvecwide=np.exp(np.linspace(logfmin,logfmax,len(logfvec)+2*smaxind*f2s_step))
        ################################sample
        #np.random.seed(trial+1)
        #integ=np.exp(np.log(rhofvec)+logfvec)
        integ=np.exp(logrhofvec+logfvec)
        logfsamples=logfvec[get_distsample(dlogf*(integ[:-1]+integ[1:]),Nclones)]
        
        logf2samples=logfsamples+np.random.permutation(svec[get_distsample(get_Ps_pm(alp,bet,realsbar,realsbar,smax,s_step),Nclones)])
        
        Zf=np.sum(np.exp(logfsamples))
        n1_samples=np.array(np.random.poisson(lam=NreadsI*np.exp(logfsamples)),dtype='uint32')
        Zfp=np.sum(np.exp(logf2samples))
        logf2samples-=np.log(Zfp)-np.log(Zf)
        n2_samples=np.array(np.random.poisson(lam=NreadsI*np.exp(logf2samples)),dtype='uint32')
        seen=np.logical_or(n1_samples>0,n2_samples>0)
        n1_samples=n1_samples[seen]
        n2_samples=n2_samples[seen]
        del(logfsamples)
        del(logf2samples)
        #f_samples_inds=dPf.rvs(size=Nclones)
        #f_samples=fvec[f_samples_inds]
        #Zf_data_store[trial]=np.sum(f_samples)
        #s_samples_inds = dPs.rvs(size=Nclones)#get_modelsample(rhosvec, samplesize).flatten()  #LINE 4 # model to array pairs, sampling desnity same as probs since step is the same
        #s_samples=svec[s_samples_inds]
        #fdash_samples=f_samples*np.exp(s_samples)
        ##normalize fp to f
        #Zfp_data_store[trial]=np.sum(fdash_samples)/Zf_data_store[trial]
        #print('Z='+str(Zfp_data_store[trial]))
        #fdash_samples/=Zfp_data_store[trial]  #---------------------------normalize
        
        #n1_samples=np.random.poisson(lam=NreadsI *f_samples)#,np.asarray([np.random.poisson(m, size=1) for m in n_total*f_samples]).flatten()  #LINE 3
        #n2_samples=np.random.poisson(lam=NreadsII*fdash_samples)
        #bruteseen=np.logical_or(n1_samples>0, n2_samples>0)
        #print('left:'+str(np.sum(bruteseen)))
        #n1_samples=n1_samples[bruteseen]
        #n2_samples=n2_samples[bruteseen]
        #f_samples=f_samples[bruteseen]
        #s_samples=s_samples[bruteseen]
        #fdash_samples=fdash_samples[bruteseen]
        #maxn1_samples=np.max(n1_samples)
        #maxn2_samples=np.max(n2_samples)
        et=time.time()

        prtfn("samples n1: "+str(np.mean(n1_samples))+" | "+str(max(n1_samples))+", n2 "+str(np.mean(n2_samples))+" | "+str(max(n2_samples)))

        ################################import data
        counts_d = pd.DataFrame({'Clone_count_1': n1_samples, 'Clone_count_2': n2_samples})            
        #transform to sparse representation
        indn1_d,indn2_d,countpaircounts_d,unicountvals_1_d,unicountvals_2_d,NreadsI_d,NreadsII_d=get_sparserep(counts_d) 
        prtfn("num n1: "+str(len(unicountvals_1_d))+ ' Nreads:'+str(NreadsI_d))
        prtfn("num n2: "+str(len(unicountvals_2_d))+ ' Nreads:'+str(NreadsII_d))
        Nsamp=np.sum(countpaircounts_d)
        np.save(outpath+"NreadsI_d_"+str(trial)+".npy",NreadsI_d)
        np.save(outpath+"NreadsII_d"+str(trial)+".npy",NreadsII_d)
        np.save(outpath+"indn1_d"+str(trial)+".npy",indn1_d)
        np.save(outpath+"indn2_d"+str(trial)+".npy",indn2_d)
        np.save(outpath+"countpaircounts_d"+str(trial)+".npy",countpaircounts_d)
        np.save(outpath + "unicountvals_1_d"+str(trial)+".npy", unicountvals_1_d)
        np.save(outpath + "unicountvals_2_d"+str(trial)+".npy", unicountvals_2_d)
            
        ################################diffexpr learning

        stt=time.time()
        
        #paras
        repfac=NreadsII_d/NreadsI_d
        Nreadsvec=(NreadsI_d,NreadsII_d)
        if case<2: #case: 0=NB->Pois, 1=Pois->NB, 2=NB, 3=Pois 
            m_total=float(np.power(10, paras[3]))
            r_c1=NreadsI_d/m_total 
            r_c2=repfac*r_c1     
            r_cvec=(r_c1,r_c2)
            fmin=np.power(10,paras[4])
        #else:
            #fmin=np.power(10,paras[3])
        if case<3:
            beta_mv= paras[1]
            alpha_mv=paras[2]

        #define coutn distribtions
        fvectmp=deepcopy(logfvec)
        for it in range(2):
            if it==0:
                unicounts=unicountvals_1_d
            else:
                unicounts=unicountvals_2_d
                if isinstance(svec,np.ndarray): #for diff expr with shift use shifted range for wide f2
                    logfvec=deepcopy(np.log(fvecwide)) #contains s-shift for sampled data method
            if case<2:
                nsigma=5.
                nmin=300.
                #for each n, get actual range of m to compute around n-dependent mean m
                m_low =np.zeros((len(unicounts),),dtype=int)
                m_high=np.zeros((len(unicounts),),dtype=int)
                for nit,n in enumerate(unicounts):
                    mean_m=n/r_cvec[it]
                    dev=nsigma*np.sqrt(mean_m)
                    m_low[nit] =int(mean_m-  dev) if (mean_m>dev**2) else 0                         
                    m_high[nit]=int(mean_m+5*dev) if (      n>nmin) else int(10*nmin/r_cvec[it])
                m_cellmax=np.max(m_high)
                #across n, collect all in-range m
                mvec_bool=np.zeros((m_cellmax+1,),dtype=bool) #cheap bool
                nvec=range(len(unicounts))
                for nit in nvec:
                    mvec_bool[m_low[nit]:m_high[nit]+1]=True  #mask vector
                mvec=np.arange(m_cellmax+1)[mvec_bool]                
                #transform to in-range index
                for nit in nvec:
                    m_low[nit]=np.where(m_low[nit]==mvec)[0][0]
                    m_high[nit]=np.where(m_high[nit]==mvec)[0][0]

            Pn_f=np.zeros((len(logfvec),len(unicounts)))
            if case==0:
                mean_m=m_total*np.exp(logfvec)
                var_m=mean_m+beta_mv*np.power(mean_m,alpha_mv)
                Poisvec=PoisPar(mvec*r_cvec[it],unicounts)
                for f_it in range(len(fvec)):
                    NBvec=NegBinPar(mean_m[f_it],var_m[f_it],mvec)
                    for n_it,n in enumerate(unicounts):
                        Pn_f[f_it,n_it]=np.dot(NBvec[m_low[n_it]:m_high[n_it]+1],Poisvec[m_low[n_it]:m_high[n_it]+1,n_it]) 
            elif case==1:
                Poisvec=PoisPar(m_total*logfvec,mvec)
                mean_n=r_cvec[it]*mvec
                NBmtr=NegBinParMtr(mean_n,mean_n+beta_mv*np.power(mean_m,alpha_mv),unicounts)
                for f_it in range(len(fvec)):
                    Poisvectmp=Poisvec[f_it,:]
                    for n_it,n in enumerate(unicounts):
                        Pn_f[f_it,n_it]=np.dot(Poisvectmp[m_low[n_it]:m_high[n_it]+1],NBmtr[m_low[n_it]:m_high[n_it]+1,n_it]) 
            elif case==2:
                mean_n=Nreadsvec[it]*logfvec
                var_n=mean_n+beta_mv*np.power(mean_n,alpha_mv)
                Pn_f=NegBinParMtr(mean_n,var_n,unicounts)
            else:# case==3:
                mean_n=Nreadsvec[it]*np.exp(logfvec)
                Pn_f=PoisPar(mean_n,unicounts)

            if it==0:
                logPn1_f=np.log(Pn_f)#[:,uinds1]) #throws warning
                #if isinstance(svec,np.ndarray):                 #when only computin Pn_f once
                #smaxind=(len(svec)-1)/2
                ##logPn1_f=logPn1_f[f2s_step*smaxind+shift_s_step:-f2s_step*smaxind+shift_s_step,:]
            else:
                logPn2_f=np.log(Pn_f)
                #logPn2_f=np.log(logPn2_f)#[:,uinds2])
        prtfn('zeros:'+str(np.sum(Pn_f==0)/float(np.prod(Pn_f.shape)))+' '+str(Pn_f.shape)+' '+str(str(np.sum(logPn2_f==-np.inf)/float(np.prod(Pn_f.shape)))))
        logfvec=fvectmp
  
        
        #learn
        m_low=0
        m_high=0
        mvec=0
        r_cvec=0

        partialobjfunc=partial(get_likelihood,null_paras=paras,svec=svec,smax=smax,s_step=s_step,indn1_d=indn1_d ,indn2_d=indn2_d,logfvec=logfvec,fvecwide=fvecwide,logrhofvec=logrhofvec,\
                                           unicountvals_1_d=unicountvals_1_d, unicountvals_2_d=unicountvals_2_d, countpaircounts_d=countpaircounts_d,\
                                           NreadsI=NreadsI_d, NreadsII=NreadsII_d, nfbins=nfbins,f2s_step=f2s_step,\
                                           m_low=m_low,m_high=m_high,mvec=mvec,Nsamp=Nsamp,r_cvec=r_cvec,logPn1_f=logPn1_f,case=case)
        
        
        
        #condict={'type':'eq','fun':constr_fn_diffexpr,'args': (paras,svec,smax,s_step,indn1_d,indn2_d,logfvec,fvecwide,logrhofvec,\
                                                              #unicountvals_1_d,unicountvals_2_d,countpaircounts_d,\
                                                              #NreadsI_d, NreadsII_d,nfbins, f2s_step,\
                                                              #m_low,m_high,mvec,Nsamp,r_cvec,logPn1_f,case)\
            #}
        
        
        initparas=(-2.,0)#,0.26)   
        #bnds = ((-4,0), (-1,1),(0,1))
        outstruct = minimize(partialobjfunc, initparas, method='SLSQP', callback=callbackFdiffexpr_part, tol=1e-6,options={'ftol':1e-8 ,'disp': True,'maxiter':30})
        #outstruct = minimize(partialobjfunc, initparas, method='SLSQP', bounds= bnds,callback=callbackF, constraints=condict,tol=1e-6,options={'ftol':1e-8 ,'disp': True,'maxiter':30})

        np.save(outpath+'outstruct_'+runname+'_'+str(trial)+'.npy',outstruct)
        np.save(outpath+'shift_'+runname+'_'+str(trial)+'.npy',shift)
        ett=time.time()
        prtfn(ett-stt)

        #eval likelihood on grid around maximum
        #nsbarpoints=5
        #shiftMtr =np.zeros((nsbarpoints,nsbarpoints))
        #fexpsMtr =np.zeros((nsbarpoints,nsbarpoints))
        #nit_list =np.zeros((nsbarpoints,nsbarpoints))
        #Zstore =np.zeros((nsbarpoints,nsbarpoints))
        #Zdashstore =np.zeros((nsbarpoints,nsbarpoints))
        #LSurface=np.zeros((nsbarpoints,nsbarpoints))

        #logalpopt=outstruct.x[0]
        #logsbaropt=outstruct.x[1]
        #fac=0.95
        #alpvec=np.linspace(fac*logalpopt,(2-fac)*logalpopt,nsbarpoints)
        #sbarvec=np.linspace(fac*logsbaropt,(2-fac)*logsbaropt,nsbarpoints)
        #first_shift=0
        #for sit,sbar in enumerate(sbarvec):
            #shift=deepcopy(first_shift)
            #smit_flag=True
            #addshift=0
            #for ait,alptest in enumerate(alpvec):
                #LSurface[ait,sit]=-partialobjfunc([alptest,sbar])

        #np.save(outpath+'LSurface_'+str(trial)+'.npy',LSurface)
        #np.save(outpath+'ShiftMtr_'+str(trial)+'.npy',shiftMtr)

        

        ##for 
        ###learn 2-form approximation

        ###invert

        ###diagnalize

        ###write evs and evecs to disk


        ##posteriors
        ##pick counts to evaluate at:
        #unicountvals_1_d
        #unicountvals_2_d

        ##get model
        #logPn_f1
        #logPn_f2
        #integ= logPn_f1 logPn_f1 logrhofvec


        #logPn1n2_s=sum integ

        

if __name__ == "__main__": 
    ntrials=int(sys.argv[1])
    
    main(ntrials)
