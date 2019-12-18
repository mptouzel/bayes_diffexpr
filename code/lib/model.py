import numpy as np
import math
import pandas as pd
from functools import partial
from copy import deepcopy
import os
from scipy.stats import nbinom
from scipy.stats import poisson
from scipy.stats import rv_discrete
from lib.proc import get_distsample
#import ctypes
#mkl_rt = ctypes.CDLL('libmkl_rt.so')
#num_threads=4
#mkl_set_num_threads = mkl_rt.MKL_Set_Num_Threads(num_threads)

def NegBinParMtr(m,v,nvec): #speed up only insofar as the log and exp are called once on array instead of multiple times on rows
    ''' 
    computes NegBin probabilities over the ordered (but possibly discontiguous) vector (nvec) 
    for mean/variance combinations given by the mean (m) and variance (v) vectors. 
    Note that m<v for negative binomial.
    Output is (len(m),len(nvec)) array
    '''
    nmax=nvec[-1]
    p = 1-m/v
    r = m*m/v/p
    NBvec=np.arange(nmax+1,dtype=float)[np.newaxis,:]
    NBvec=np.log((NBvec+r[:,np.newaxis]-1)*(p[:,np.newaxis]/NBvec))
    NBvec[:,0]=r*np.log(m/v) #handle NBvec[0]=0, treated specially when m[0]=0, see below
    NBvec=np.exp(np.cumsum(NBvec,axis=1)) #save a bit here
    if m[0]==0:
        NBvec[0,:]=0.
        NBvec[0,0]=1.
    NBvec=NBvec[:,nvec]
    return NBvec


def NegBinPar(m,v,mvec): 
    '''
    Same as NegBinParMtr, but for m and v being scalars.
    Assumes m>0.
    Output is (len(mvec),) array
    '''
    mmax=mvec[-1]
    p = 1-m/v
    r = m*m/v/p
    NBvec=np.arange(mmax+1,dtype=float)   
    NBvec[1:]=np.log((NBvec[1:]+r-1)/NBvec[1:]*p) #vectorization won't help unfortuneately here since log needs to be over array
    NBvec[0]=r*math.log(m/v)
    NBvec=np.exp(np.cumsum(NBvec)[mvec]) #save a bit here
    return NBvec
  
def PoisPar(Mvec,unicountvals):
    #assert Mvec[0]==0, "first element needs to be zero"
    nmax=unicountvals[-1]
    nlen=len(unicountvals)
    mlen=len(Mvec)
    Nvec=unicountvals
    logNvec=-np.insert(np.cumsum(np.log(np.arange(1,nmax+1))),0,0.)[unicountvals] #avoid n=0 nans  
    Nmtr=np.exp(Nvec[np.newaxis,:]*np.log(Mvec)[:,np.newaxis]+logNvec[np.newaxis,:]-Mvec[:,np.newaxis]) # np.log(Mvec) throws warning: since log(0)=-inf
    if Mvec[0]==0:
        Nmtr[0,:]=np.zeros((nlen,)) #when m=0, n=0, and so get rid of nans from log(0)
        Nmtr[0,0]=1. #handled belowacq_model_type
    if unicountvals[0]==0: #if n=0 included get rid of nans from log(0)
        Nmtr[:,0]=np.exp(-Mvec)
    return Nmtr
  

def get_rhof(alpha_rho,fmin,freq_nbins=800,freq_dtype='float64'):
    '''
    generates power law (power is alpha_rho) clone frequency distribution over 
    freq_nbins discrete logarithmically spaced frequences between fmin and 1 of dtype freq_dtype
    Outputs log probabilities obtained at log frequencies'''
    fmax=1e0
    logfvec=np.linspace(np.log10(fmin),np.log10(fmax),freq_nbins)
    logfvec=np.array(np.log(np.power(10,logfvec)) ,dtype=freq_dtype).flatten()  
    logrhovec=logfvec*alpha_rho
    integ=np.exp(logrhovec+logfvec,dtype=freq_dtype)
    normconst=np.log(np.dot(np.diff(logfvec)/2.,integ[1:]+integ[:-1]))
    logrhovec-=normconst 
    return logrhovec,logfvec

def get_fvec_and_svec(paras,s_step,smax,freq_dtype='float64'):
    '''
    biuld discrete domain of s, centered on s=0, also extend fvec range from [fmin,1] to [fmin-smax*s2f,1+s2f*smax] 
    '''
    logrhofvec,logfvec = get_rhof(paras[0],np.power(10,paras[-1]),freq_dtype=freq_dtype)
    s_step_old=deepcopy(s_step)
    logf_step=logfvec[1] - logfvec[0] #use natural log here since f2 increments in increments in exp().  
    f2s_step=int(round(s_step/logf_step)) #rounded number of f-steps in one s-step
    s_step=float(f2s_step)*logf_step
    smax=s_step*(smax/s_step_old)
    svec=s_step*np.arange(0,int(round(smax/s_step)+1))   
    svec=np.append(-svec[1:][::-1],svec)
    smaxind=(len(svec)-1)/2
    logfmin=logfvec[0 ]-f2s_step*smaxind*logf_step
    logfmax=logfvec[-1]+f2s_step*smaxind*logf_step
    logfvecwide=np.linspace(logfmin,logfmax,len(logfvec)+2*smaxind*f2s_step)
    return svec,logfvec,logfvecwide,f2s_step,smax,s_step

def get_Ps(alp,sbar,smax,stp):
    '''
    generates symmetric exponential distribution over log fold change
    with effect size sbar and nonresponding fraction 1-alp at s=0.
    computed over discrete range of s from -smax to smax in steps of size stp
    '''
    lamb=-stp/sbar
    smaxt=round(smax/stp)
    s_zeroind=int(smaxt)
    Z=2*(np.exp((smaxt+1)*lamb)-1)/(np.exp(lamb)-1)-1
    Ps=alp*np.exp(lamb*np.fabs(np.arange(-smaxt,smaxt+1)))/Z
    Ps[s_zeroind]+=(1-alp)
    return Ps
  
def get_logPs_pm(paras,smax,stp,func_type):
    '''
    generates asymmetric exponential distribution over log fold change
    with contraction effect size sbar_m expansion effect size sbar_p and responding fraction alp.
    computed over discrete range of s from -smax to smax in steps of size stp.
    note that the responding fraction has no s=0 contribution.
    '''
    smaxt=round(smax/stp)
    Ps=np.zeros(2*int(smaxt)+1)
    
    alp=paras[0]
    sbar_p=paras[1]
    if func_type=='rhs_only':        #(alp,sbar_p)        
        lambp=-stp/sbar_p
        Z_p=(np.exp((smaxt+1)*lambp)-1)/(np.exp(lambp)-1)-1 #no s=0 contribution
        Ps[int(smaxt)+1:] = np.exp(lambp*np.fabs(np.arange(           1,int(smaxt)+1)))/Z_p
        Ps*=alp
        Ps[int(smaxt)]=(1-alp) #the sole contribution to s=0
    elif func_type=='sym_exp':         #(alp,sbar_p)
        lambp=-stp/sbar_p
        Z_p=2*((np.exp((smaxt+1)*lambp)-1)/(np.exp(lambp)-1)-1) #no s=0 contribution
        Ps[:int(smaxt)]=np.exp(lambp*np.fabs(np.arange(0-int(smaxt),           0)))/Z_p
        Ps[int(smaxt)+1:]  =np.exp(lambp*np.fabs(np.arange(           1,int(smaxt)+1)))/Z_p
        Ps*=alp
        Ps[int(smaxt)]=(1-alp) #the sole contribution to s=0
    elif func_type=='asym_exp':         #(alp,bet,sbar_m,sbar_p)
        sbar_m=paras[2]
        bet=paras[3]
        lambm=-stp/sbar_m
        lambp=-stp/sbar_p
        Z_m=(np.exp((smaxt+1)*lambm)-1)/(np.exp(lambm)-1)-1 #no s=0 contribution
        Z_p=(np.exp((smaxt+1)*lambp)-1)/(np.exp(lambp)-1)-1 #no s=0 contribution
        Ps[:int(smaxt)]=(1-bet)*np.exp(lambm*np.fabs(np.arange(0-int(smaxt),           0)))/Z_m/2
        Ps[int(smaxt)+1:]  =bet*np.exp(lambp*np.fabs(np.arange(           1,int(smaxt)+1)))/Z_p/2
        Ps*=alp
        Ps[int(smaxt)]=(1-alp) #the sole contribution to s=0
    elif func_type=='cent_gauss':   #(alp,sbar_p)
        lambp=-stp/sbar_p
        svec=lambp*np.arange(           -int(smaxt),int(smaxt)+1)
        Ps=np.exp(-svec*svec)
        Ps[int(smaxt)]=0
        Ps*=alp/np.sum(Ps)
        Ps[int(smaxt)]=(1-alp)
    elif func_type=='offcent_gauss':   #(alp,sbar_p,second_pos)
        second_pos=paras[2]
        lambp=-stp/sbar_p
        svec=lambp*np.arange(           -int(smaxt),int(smaxt)+1)
        svec-=lambp*second_pos/stp
        Ps=np.exp(-svec*svec)
        Ps[int(smaxt)]=0
        Ps*=alp/np.sum(Ps)
        Ps[int(smaxt)]=(1-alp)           
    else:
        print('func_Type does not exist!')
    assert np.fabs(np.sum(Ps)-1)<1e-5, "P(s) distribution not normalized! "+str(np.sum(Ps))
    return np.log(Ps)
#@profile
def get_logPn_f(unicounts,Nreads,logfvec,acq_model_type,paras):
    
    if acq_model_type<2:
        m_total=float(np.power(10, paras[3])) 
        r_c=Nreads/m_total
    if acq_model_type<3:
        beta_mv= paras[1]
        alpha_mv=paras[2]
        
    if acq_model_type<2: #for models that include cell counts
        #compute parametrized range (mean-sigma,mean+5*sigma) of m values (number of cells) conditioned on n values (reads) appearing in the data only 
        nsigma=5.
        nmin=300.
        #for each n, get actual range of m to compute around n-dependent mean m
        m_low =np.zeros((len(unicounts),),dtype=int)
        m_high=np.zeros((len(unicounts),),dtype=int)
        for nit,n in enumerate(unicounts):
            mean_m=n/r_c
            dev=nsigma*np.sqrt(mean_m)
            m_low[nit] =int(mean_m-  dev) if (mean_m>dev**2) else 0                         
            m_high[nit]=int(mean_m+5*dev) if (      n>nmin) else int(10*nmin/r_c)
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
    if acq_model_type==0:
        mean_m=m_total*np.exp(logfvec)
        var_m=mean_m+beta_mv*np.power(mean_m,alpha_mv)
        Poisvec=PoisPar(mvec*r_c,unicounts)
        for f_it in range(len(logfvec)):
            NBvec=NegBinPar(mean_m[f_it],var_m[f_it],mvec)
            for n_it,n in enumerate(unicounts):
                Pn_f[f_it,n_it]=np.dot(NBvec[m_low[n_it]:m_high[n_it]+1],Poisvec[m_low[n_it]:m_high[n_it]+1,n_it]) 
    elif acq_model_type==1:  #incorrect! Not a convolution of m Negative binomials is not a Negative binomial with mean multiplied by m (as it was in the NegBin->Poisson case)!
        Poisvec=PoisPar(m_total*np.exp(logfvec),mvec)
        mean_n=r_c*mvec
        NBmtr=NegBinParMtr(mean_n,mean_n+beta_mv*np.power(mean_m,alpha_mv),unicounts)
        for f_it in range(len(logfvec)):
            Poisvectmp=Poisvec[f_it,:]
            for n_it,n in enumerate(unicounts):
                Pn_f[f_it,n_it]=np.dot(Poisvectmp[m_low[n_it]:m_high[n_it]+1],NBmtr[m_low[n_it]:m_high[n_it]+1,n_it]) 
    elif acq_model_type==2:
        mean_n=Nreads*np.exp(logfvec)
        var_n=mean_n+beta_mv*np.power(mean_n,alpha_mv)
        Pn_f=NegBinParMtr(mean_n,var_n,unicounts)
    elif acq_model_type==3:
        mean_n=Nreads*np.exp(logfvec)
        Pn_f=PoisPar(mean_n,unicounts)
    else:
        print('acq_model is 0,1,2, or 3 only')

    return np.log(Pn_f)

#@profile
def get_Pn1n2_s(paras, svec, sparse_rep,acq_model_type,  s_step=0):    #changed ferq_dtype default to float64 
    #svec determines which of 3 run modes is evaluated
    #1) svec is array => compute P(n1,n2|s),           output: Pn1n2_s,unicountvals_1,unicountvals_2,Pn1_f,fvec,Pn2_s,svec
    #2) svec=-1       => null model likelihood,        output: data-averaged loglikelihood
    #3) else          => compute null model, P(n1,n2), output: Pn1n2,unicountvals_1,unicountvals_2,Pn1_f,Pn2_f,fvec
    
    #acq_model_type input is which P(n|f) model to use (see get_logPn_f function): 
    #paras is the list of null model parameters, length depends on the acq_model_type
    #   0:NB->Pois, (alpha_rho, beta_mv, alpha_mv, log_{10}M, log_{10}fmin)
    #   1:Pois->NB, (alpha_rho, beta_mv, alpha_mv, log_{10}M, log_{10}fmin)
    #   2:NBonly,   (alpha_rho, beta_mv, alpha_mv, log_{10}fmin)
    #   3:Poisonly  (alpha_rho, log_{10}fmin)
    #mean variance relation: v = m + beta_mv*m^alpha_mv
    
    indn1,indn2,sparse_rep_counts,unicountvals_1,unicountvals_2,NreadsI,NreadsII=sparse_rep
   
    alpha = paras[0] #power law exponent
    fmin = np.power(10,paras[-1])
    logrhofvec,logfvec = get_rhof(alpha,fmin)
    
    if isinstance(svec,np.ndarray):
        smaxind=(len(svec)-1)/2
        logf_step=logfvec[1] - logfvec[0] 
        f2s_step=int(round(s_step/logf_step)) #rounded number of f-steps in one s-step
        logfmin=logfvec[0 ]-f2s_step*smaxind*logf_step
        logfmax=logfvec[-1]+f2s_step*smaxind*logf_step
        logfvecwide=np.linspace(logfmin,logfmax,len(logfvec)+2*smaxind*f2s_step) #a wider domain for the second frequency f2=f1*exp(s)
    
    logPn1_f=get_logPn_f(unicountvals_1,NreadsI,logfvec,acq_model_type,paras)
    logPn2_f=get_logPn_f(unicountvals_2,NreadsII,logfvecwide if isinstance(svec,np.ndarray) else logfvec,acq_model_type,paras) #for diff expr with shift use shifted range for wide f2
  
    dlogfby2=np.diff(logfvec)/2. #for later integration via trapezoid method
    if isinstance(svec,np.ndarray):  #P(n1,n2|s)
        print('computing P(n1,n2|f,s)')
        Pn1n2_s=np.zeros((len(svec),len(unicountvals_1),len(unicountvals_2))) 
        for s_it,s in enumerate(svec):
            for n1_it,n2_it in zip(indn1,indn2):
                integ=np.exp(logrhofvec+logPn2_f[f2s_step*s_it:(f2s_step*s_it+len(logfvec)),n2_it]+logPn1_f[:,n1_it]+logfvec)
                Pn1n2_s[s_it,n1_it,n2_it] = np.dot(dlogfby2,integ[1:] + integ[:-1])
        Pn0n0_s=np.zeros(svec.shape)
        for s_it,s in enumerate(svec):    
            integ=np.exp(logPn1_f[:,0]+logPn2_f[f2s_step*s_it:(f2s_step*s_it+len(logfvec)),0]+logrhofvec+logfvec)
            Pn0n0_s[s_it]=np.dot(dlogfby2,integ[1:]+integ[:-1])
        Pn2_s=0
        return Pn1n2_s,unicountvals_1,unicountvals_2,np.exp(logPn1_f),np.exp(logfvec),Pn2_s,Pn0n0_s,svec
      
    elif svec==-1: #scalar null model marginal likelihood
        
        integ=np.exp(logrhofvec + logPn2_f[:,0] + logPn1_f[:,0] + logfvec)
        Pn0n0 = np.dot(dlogfby2,integ[1:] + integ[:-1])

        if False: #block to compute and print constraint to check normalization satisfaction
            logPnng0=np.log(1-Pn0n0)
            logPnn0=np.log(Pn0n0)
            logNclones = np.log(np.sum(sparse_rep_counts))-logPnng0
            
            integ=np.exp(logrhofvec + logPn2_f[:,0] + logPn1_f[:,0] + 2*logfvec)
            log_avgf_n0n0 = np.log(np.dot(dlogfby2,integ[1:]+integ[:-1]))
            
            integ=np.exp(logPn1_f[:,indn1]+logPn2_f[:,indn2]+logrhofvec[:,np.newaxis]+logfvec[:,np.newaxis])
            log_Pn1n2=np.log(np.sum(dlogfby2[:,np.newaxis]*(integ[1:,:] + integ[:-1,:]),axis=0))
            integ=np.exp(np.log(integ)+logfvec[:,np.newaxis]) 
            tmp=deepcopy(log_Pn1n2)
            tmp[tmp==-np.Inf]=np.Inf #since subtracted in next line
            avgf_n1n2=    np.exp(np.log(np.sum(dlogfby2[:,np.newaxis]*(integ[1:,:] + integ[:-1,:]),axis=0))-tmp)
            log_sumavgf=np.log(np.dot(sparse_rep_counts,avgf_n1n2))
            
            Z_cond     = np.exp(logNclones + logPnn0 + log_avgf_n0n0) + np.exp(log_sumavgf) 
            
            integ=np.exp(logrhofvec+2*logfvec)
            avgf_ps=np.dot(dlogfby2,integ[:-1]+integ[1:])
            
            avgf_null_pair=np.exp(np.log(1-Pn0n0)-np.log(np.sum(sparse_rep_counts)))
            print(str(np.log(Z_cond))+' '+str(np.log(avgf_ps)-np.log(avgf_null_pair)))

        Pn1n2_s=np.zeros(len(sparse_rep_counts)) #1D representation
        for it,(ind1,ind2) in enumerate(zip(indn1,indn2)):
            integ=np.exp(logPn1_f[:,ind1]+logrhofvec+logPn2_f[:,ind2]+logfvec)
            Pn1n2_s[it] = np.dot(dlogfby2,integ[1:] + integ[:-1])
        Pn1n2_s/=1.-Pn0n0 #renormalize
        return -np.dot(sparse_rep_counts,np.where(Pn1n2_s>0,np.log(Pn1n2_s),0))/float(np.sum(sparse_rep_counts))
    
    else: #s=0 (null model)        
        print('running Null Model, ')        
        Pn1n2_s=np.zeros((len(unicountvals_1),len(unicountvals_2))) #2D representation 
        for n2_it,n2 in enumerate(unicountvals_2): 
            for n1_it,n1 in enumerate(unicountvals_1):
                integ=np.exp(logPn1_f[:,n1_it]+logrhofvec+logPn2_f[:,n2_it]+logfvec)
                Pn1n2_s[n1_it,n2_it] = np.dot(dlogfby2,integ[1:] + integ[-1])
        Pn1n2_s/=1.-Pn1n2_s[0,0] #remove (n1,n2)=(0,0) and renormalize
        Pn1n2_s[0,0]=0.
        return Pn1n2_s,unicountvals_1,unicountvals_2,logPn1_f,logPn2_f,logfvec

#--------------------------Model Sampling-------------------------------

def get_nullmodel_sample_observedonly(paras,acq_model_type,NreadsI,NreadsII,Nsamp):
    '''
    outputs an array of observed clone frequencies and corresponding dataframe of pair counts
    for a null model learned from a dataset pair with NreadsI and NreadsII number of reads, respectively.
    Crucial for RAM efficiency, sampling is conditioned on being observed in each of the three (n,0), (0,n'), and n,n'>0 conditions
    so that only Nsamp clones need to be sampled, rather than the N clones in the repretoire.
    Note that no explicit normalization is applied. It is assumed that the values in paras are consistent with N<f>=1 
    (e.g. were obtained through the learning done in this package).
    '''

    
    alpha = paras[0] #power law exponent
    fmin=np.power(10,paras[-1])
    if acq_model_type<2:
        m_total=float(np.power(10, paras[3])) 
        r_c1=NreadsI/m_total
        r_c2=NreadsII/m_total
        r_cvec=[r_c1,r_c2]
    if acq_model_type<3:
        beta_mv= paras[1]
        alpha_mv=paras[2]
    
    logrhofvec,logfvec = get_rhof(alpha,fmin)
    fvec=np.exp(logfvec)
    dlogf=np.diff(logfvec)/2.
    
    #generate measurement model distribution, Pn_f
    Pn_f=np.empty((len(logfvec),),dtype=object) #len(logfvec) samplers
    
    #get value at n=0 to use for conditioning on n>0 (and get full Pn_f here if acq_model_type=2,3)
    m_max=1e3 #conditioned on n=0, so no edge effects
    
    Nreadsvec=(NreadsI,NreadsII)
    for it in range(2):
        Pn_f=np.empty((len(fvec),),dtype=object)
        if acq_model_type==3:
            m1vec=Nreadsvec[it]*fvec
            for find,m1 in enumerate(m1vec):
                Pn_f[find]=poisson(m1)
            logPn0_f=-m1vec
        elif acq_model_type==2:
            m1=Nreadsvec[it]*fvec
            v1=m1+beta_mv*np.power(m1,alpha_mv)
            p=1-m1/v1
            n=m1*m1/v1/p
            for find,(n,p) in enumerate(zip(n,p)):
                Pn_f[find]=nbinom(n,1-p)
            Pn0_f=np.asarray([Pn_find.pmf(0) for Pn_find in Pn_f])
            logPn0_f=np.log(Pn0_f)
        elif acq_model_type==1:
            m1=r_cvec[it]*np.arange(m_max+1)
            v1=m1+beta_mv*np.power(m1,alpha_mv)
            p=1-m1/v1
            p[0]=1.
            n=m1*m1/v1/p
            n[0]=0.
            Pn0_f=np.zeros((len(fvec),))
            for find in range(len(Pn0_f)):
                Pn0_f[find]=np.dot(poisson(m_total*fvec[find]).pmf(np.arange(m_max+1)),np.insert(nbinom(n[1:],1-p[1:]).pmf(0),0,1.))
            logPn0_f=np.log(Pn0_f)
        elif acq_model_type==0:
            m1=m_total*fvec
            v1=m1+beta_mv*np.power(m1,alpha_mv)
            p=1-m1/v1
            n=m1*m1/v1/p
            Pn0_f=np.zeros((len(fvec),))
            for find in range(len(Pn0_f)):
                nbtmp=nbinom(n[find],1-p[find]).pmf(np.arange(m_max+1))
                ptmp=poisson(r_cvec[it]*np.arange(m_max+1)).pmf(0)
                Pn0_f[find]=np.sum(np.exp(np.log(nbtmp)+np.log(ptmp)))
            logPn0_f=np.log(Pn0_f)
        else:
            print('acq_model is 0,1,2, or 3 only')
            
        if it==0:
            Pn1_f=Pn_f
            logPn10_f=logPn0_f
        else:
            Pn2_f=Pn_f
            logPn20_f=logPn0_f

    #3-quadrant q|f conditional distribution (qx0:n1>0,n2=0;q0x:n1=0,n2>0;qxx:n1,n2>0)
    logPqx0_f=np.log(1-np.exp(logPn10_f))+logPn20_f
    logPq0x_f=logPn10_f+np.log(1-np.exp(logPn20_f))
    logPqxx_f=np.log(1-np.exp(logPn10_f))+np.log(1-np.exp(logPn20_f))
    #3-quadrant q,f joint distribution
    logPfqx0=logPqx0_f+logrhofvec
    logPfq0x=logPq0x_f+logrhofvec
    logPfqxx=logPqxx_f+logrhofvec
    #3-quadrant q marginal distribution 
    Pqx0=np.trapz(np.exp(logPfqx0+logfvec),x=logfvec)
    Pq0x=np.trapz(np.exp(logPfq0x+logfvec),x=logfvec)
    Pqxx=np.trapz(np.exp(logPfqxx+logfvec),x=logfvec)
    
    #3 quadrant conditional f|q distribution
    Pf_qx0=np.where(Pqx0>0,np.exp(logPfqx0-np.log(Pqx0)),0)
    Pf_q0x=np.where(Pq0x>0,np.exp(logPfq0x-np.log(Pq0x)),0)
    Pf_qxx=np.where(Pqxx>0,np.exp(logPfqxx-np.log(Pqxx)),0)
    
    #3-quadrant q marginal distribution
    newPqZ=Pqx0 + Pq0x + Pqxx
    Pqx0/=newPqZ
    Pq0x/=newPqZ
    Pqxx/=newPqZ

    Pfqx0=np.exp(logPfqx0)
    Pfq0x=np.exp(logPfq0x)
    Pfqxx=np.exp(logPfqxx)
    
    print('Model probs: '+str(Pqx0)+' '+str(Pq0x)+' '+str(Pqxx))

    #get samples 
    num_samples=Nsamp
    q_samples=np.random.choice(range(3), num_samples, p=(Pqx0,Pq0x,Pqxx))
    vals,counts=np.unique(q_samples,return_counts=True)
    num_qx0=counts[0]
    num_q0x=counts[1]
    num_qxx=counts[2]
    print('q samples: '+str(sum(counts))+' '+str(num_qx0)+' '+str(num_q0x)+' '+str(num_qxx))
    print('q sampled probs: '+str(num_qx0/float(sum(counts)))+' '+str(num_q0x/float(sum(counts)))+' '+str(num_qxx/float(sum(counts))))
    
    #x0
    integ=np.exp(np.log(Pf_qx0)+logfvec)
    f_samples_inds=get_distsample(dlogf*(integ[1:] + integ[:-1]),num_qx0).flatten()
    f_sorted_inds=np.argsort(f_samples_inds)
    f_samples_inds=f_samples_inds[f_sorted_inds] 
    qx0_f_samples=fvec[f_samples_inds]
    find_vals,f_start_ind,f_counts=np.unique(f_samples_inds,return_counts=True,return_index=True)
    qx0_samples=np.zeros((num_qx0,))
    if acq_model_type<2:
        qx0_m_samples=np.zeros((num_qx0,))
        #conditioning on n>0 applies an m-dependent factor to Pm_f, which can't be incorporated into the ppf method used for acq_model_type 2 and 3. 
        #We handle that here by using a custom finite range sampler, which has the drawback of having to define an upper limit. 
        #This works so long as n_max/r_c<<m_max, so depends on highest counts in data (n_max). My data had max counts of 1e3-1e4.
        #Alternatively, could define a custom scipy RV class by defining it's PMF, but has to be array-compatible which requires care. 
        m_samp_max=int(1e5) 
        mvec=np.arange(m_samp_max)   
    
    for it,find in enumerate(find_vals):
        if acq_model_type==0:      
            m1=m_total*fvec[find]
            v1=m1+beta_mv*np.power(m1,alpha_mv)
            p=1-m1/v1
            n=m1*m1/v1/p
            Pm1_f=nbinom(n,1-p)
            
            Pm1_f_adj=np.exp(np.log(1-np.exp(-r_c1*mvec))+np.log(Pm1_f.pmf(mvec))-np.log((1-np.power(np.exp(r_c1+np.log(1-p))/(np.exp(r_c1)-p),n)))) #adds m-dependent factor due to conditioning on n>0...
            Pm1_f_adj_obj=rv_discrete(name='nbinom_adj',values=(mvec,Pm1_f_adj/np.sum(Pm1_f_adj)))
            qx0_m_samples[f_start_ind[it]:f_start_ind[it]+f_counts[it]]=Pm1_f_adj_obj.rvs(size=f_counts[it])
            
            mvals,minds,m_counts=np.unique(qx0_m_samples[f_start_ind[it]:f_start_ind[it]+f_counts[it]],return_inverse=True,return_counts=True)
            for mit,m in enumerate(mvals):
                Pn1_m1=poisson(r_c1*m)
                samples=np.random.random(size=m_counts[mit]) * (1-Pn1_m1.cdf(0)) + Pn1_m1.cdf(0)
                qx0_samples[f_start_ind[it]:f_start_ind[it]+f_counts[it]][minds==mit]=Pn1_m1.ppf(samples)
 
        elif acq_model_type==1:
            Pm1_f=poisson(m_total*fvec[find])
            
            m1=r_c1*mvec
            v1=m1+beta_mv*np.power(m1,alpha_mv)
            p=1.-m1/v1
            n=m1*m1/v1/p
            p[0]=1.
            n[0]=0.
            Pn10_m1=nbinom(n,1-p).pmf(0)
            Pn10_m1[0]=1.
            Pm1_f_adj=(1-Pn10_m1)/(1-np.sum(Pm1_f.pmf(mvec)*Pn10_m1))*Pm1_f.pmf(mvec) #adds m-dependent factor due to conditioning on n>0...
            Pm1_f_adj_obj=rv_discrete(name='nbinom_adj',values=(mvec,Pm1_f_adj/np.sum(Pm1_f_adj)))
            qx0_m_samples[f_start_ind[it]:f_start_ind[it]+f_counts[it]]=Pm1_f_adj_obj.rvs(size=f_counts[it])
            
            mvals,minds,m_counts=np.unique(qx0_m_samples[f_start_ind[it]:f_start_ind[it]+f_counts[it]],return_inverse=True,return_counts=True)
            for mit,m in enumerate(mvals):
                m1=r_c1*m
                v1=m1+beta_mv*np.power(m1,alpha_mv)
                if m==0:
                  p=1.
                  n=0.
                else:
                  p=1-m1/v1
                  n=m1*m1/v1/p
                  Pn1_m1=nbinom(n,1-p)
                samples=np.random.random(size=m_counts[mit]) * (1-Pn1_m1.cdf(0)) + Pn1_m1.cdf(0)                        
                qx0_samples[f_start_ind[it]:f_start_ind[it]+f_counts[it]][minds==mit]=Pn1_m1.ppf(samples)
        
        elif acq_model_type>1:
            samples=np.random.random(size=f_counts[it]) * (1-Pn1_f[find].cdf(0)) + Pn1_f[find].cdf(0)
            qx0_samples[f_start_ind[it]:f_start_ind[it]+f_counts[it]]=Pn1_f[find].ppf(samples)
        else:
            print('acq_model is 0,1,2, or 3 only')
    qx0_pair_samples=np.hstack((qx0_samples[:,np.newaxis],np.zeros((num_qx0,1)))) 
    
    #0x
    integ=np.exp(np.log(Pf_q0x)+logfvec)
    f_samples_inds=get_distsample(dlogf*(integ[1:] + integ[:-1]),num_q0x).flatten()
    f_sorted_inds=np.argsort(f_samples_inds)
    f_samples_inds=f_samples_inds[f_sorted_inds] 
    q0x_f_samples=fvec[f_samples_inds]
    find_vals,f_start_ind,f_counts=np.unique(f_samples_inds,return_counts=True,return_index=True)
    q0x_samples=np.zeros((num_q0x,))
    if acq_model_type<2:
        q0x_m_samples=np.zeros((num_q0x,))
    for it,find in enumerate(find_vals):
        if acq_model_type==0:
            m2=m_total*fvec[find]
            v2=m2+beta_mv*np.power(m2,alpha_mv)
            p=1-m2/v2
            n=m2*m2/v2/p
            Pm2_f=nbinom(n,1-p)
            
            Pm2_f_adj=np.exp(np.log(1-np.exp(-r_c2*mvec))+np.log(Pm2_f.pmf(mvec))-np.log((1-np.power(np.exp(r_c2+np.log(1-p))/(np.exp(r_c2)-p),n)))) #adds m-dependent factor due to conditioning on n>0...
            Pm2_f_adj_obj=rv_discrete(name='nbinom_adj',values=(mvec,Pm2_f_adj/np.sum(Pm2_f_adj)))
            q0x_m_samples[f_start_ind[it]:f_start_ind[it]+f_counts[it]]=Pm2_f_adj_obj.rvs(size=f_counts[it])

            mvals,minds,m_counts=np.unique(q0x_m_samples[f_start_ind[it]:f_start_ind[it]+f_counts[it]],return_inverse=True,return_counts=True)
            for mit,m in enumerate(mvals):
                Pn2_m2=poisson(r_c2*m)
                samples=np.random.random(size=m_counts[mit]) * (1-Pn2_m2.cdf(0)) + Pn2_m2.cdf(0)
                q0x_samples[f_start_ind[it]:f_start_ind[it]+f_counts[it]][minds==mit]=Pn2_m2.ppf(samples)
        
        elif acq_model_type==1:
            Pm2_f=poisson(m_total*fvec[find])
            
            m2=r_c2*mvec
            v2=m2+beta_mv*np.power(m2,alpha_mv)
            p=1-m2/v2
            n=m2*m2/v2/p
            p[0]=1
            n[0]=0
            Pn20_m2=nbinom(n,1-p).pmf(0)
            Pn20_m2[0]=1.
            Pm2_f_adj=np.exp(np.log(1-Pn20_m2)-np.log(1-np.dot(Pm2_f.pmf(mvec),Pn20_m2))+np.log(Pm2_f.pmf(mvec))) #adds m-dependent factor due to conditioning on n>0...
            Pm2_f_adj_obj=rv_discrete(name='nbinom_adj',values=(mvec,Pm2_f_adj/np.sum(Pm2_f_adj)))
            q0x_m_samples[f_start_ind[it]:f_start_ind[it]+f_counts[it]]=Pm2_f_adj_obj.rvs(size=f_counts[it])

            mvals,minds,m_counts=np.unique(q0x_m_samples[f_start_ind[it]:f_start_ind[it]+f_counts[it]],return_inverse=True,return_counts=True)
            for mit,m in enumerate(mvals):
                m2=r_c2*m
                v2=m2+beta_mv*np.power(m2,alpha_mv)
                if m==0:
                  p=1.
                  n=0.
                else:
                  p=1-m2/v2
                  n=m2*m2/v2/p
                Pn2_m2=nbinom(n,1-p)
                samples=np.random.random(size=m_counts[mit]) * (1-Pn2_m2.cdf(0)) + Pn2_m2.cdf(0)                        
                q0x_samples[f_start_ind[it]:f_start_ind[it]+f_counts[it]][minds==mit]=Pn2_m2.ppf(samples)            
        
        elif acq_model_type > 1:
            samples=np.random.random(size=f_counts[it]) * (1-Pn2_f[find].cdf(0)) + Pn2_f[find].cdf(0)
            q0x_samples[f_start_ind[it]:f_start_ind[it]+f_counts[it]]=Pn2_f[find].ppf(samples)
        else:
            print('acq_model is 0,1,2, or 3 only')
    q0x_pair_samples=np.hstack((np.zeros((num_q0x,1)),q0x_samples[:,np.newaxis]))
    
    #qxx
    integ=np.exp(np.log(Pf_qxx)+logfvec)
    f_samples_inds=get_distsample(dlogf*(integ[1:] + integ[:-1]),num_qxx).flatten()        
    f_sorted_inds=np.argsort(f_samples_inds)
    f_samples_inds=f_samples_inds[f_sorted_inds] 
    qxx_f_samples=fvec[f_samples_inds]
    find_vals,f_start_ind,f_counts=np.unique(f_samples_inds,return_counts=True,return_index=True)
    qxx_n1_samples=np.zeros((num_qxx,))
    qxx_n2_samples=np.zeros((num_qxx,))
    if acq_model_type<2:
        qxx_m1_samples=np.zeros((num_qxx,))
        qxx_m2_samples=np.zeros((num_qxx,))
    for it,find in enumerate(find_vals):
        if acq_model_type==0:
            m1=m_total*fvec[find]
            v1=m1+beta_mv*np.power(m1,alpha_mv)
            p=1-m1/v1
            n=m1*m1/v1/p
            Pm1_f=nbinom(n,1-p)
            
            Pm1_f_adj=np.exp(np.log(1-np.exp(-r_c1*mvec))+np.log(Pm1_f.pmf(mvec))-np.log((1-np.power(np.exp(r_c1+np.log(1-p))/(np.exp(r_c1)-p),n)))) #adds m-dependent factor due to conditioning on n>0...
            if np.sum(Pm1_f_adj)==0:
                qxx_m1_samples[f_start_ind[it]:f_start_ind[it]+f_counts[it]]=1
            else:
                Pm1_f_adj_obj=rv_discrete(name='nbinom_adj',values=(mvec,Pm1_f_adj/np.sum(Pm1_f_adj)))
                qxx_m1_samples[f_start_ind[it]:f_start_ind[it]+f_counts[it]]=Pm1_f_adj_obj.rvs(size=f_counts[it])

            mvals,minds,m_counts=np.unique(qxx_m1_samples[f_start_ind[it]:f_start_ind[it]+f_counts[it]],return_inverse=True,return_counts=True)
            for mit,m in enumerate(mvals):
                Pn1_m1=poisson(r_c1*m)
                samples=np.random.random(size=m_counts[mit]) * (1-Pn1_m1.cdf(0)) + Pn1_m1.cdf(0)
                qxx_n1_samples[f_start_ind[it]:f_start_ind[it]+f_counts[it]][minds==mit]=Pn1_m1.ppf(samples)
                
            m2=m_total*fvec[find]
            v2=m2+beta_mv*np.power(m2,alpha_mv)
            p=1-m2/v2
            n=m2*m2/v2/p
            Pm2_f=nbinom(n,1-p)
            
            Pm2_f_adj=np.exp(np.log(1-np.exp(-r_c2*mvec))+np.log(Pm2_f.pmf(mvec))-np.log((1-np.power(np.exp(r_c2+np.log(1-p))/(np.exp(r_c2)-p),n)))) #adds m-dependent factor due to conditioning on n>0...
            if np.sum(Pm1_f_adj)==0:
                qxx_m2_samples[f_start_ind[it]:f_start_ind[it]+f_counts[it]]=1
            else:
                Pm2_f_adj_obj=rv_discrete(name='nbinom_adj',values=(mvec,Pm2_f_adj/np.sum(Pm2_f_adj)))
                qxx_m2_samples[f_start_ind[it]:f_start_ind[it]+f_counts[it]]=Pm2_f_adj_obj.rvs(size=f_counts[it])

            mvals,minds,m_counts=np.unique(qxx_m2_samples[f_start_ind[it]:f_start_ind[it]+f_counts[it]],return_inverse=True,return_counts=True)
            for mit,m in enumerate(mvals):
                Pn2_m2=poisson(r_c2*m)
                samples=np.random.random(size=m_counts[mit]) * (1-Pn2_m2.cdf(0)) + Pn2_m2.cdf(0)
                qxx_n2_samples[f_start_ind[it]:f_start_ind[it]+f_counts[it]][minds==mit]=Pn2_m2.ppf(samples)    

        elif acq_model_type==1:
            Pm1_f=poisson(m_total*fvec[find])
            
            m1=r_c1*mvec
            v1=m1+beta_mv*np.power(m1,alpha_mv)
            p=1-m1/v1
            n=m1*m1/v1/p
            p[0]=1
            n[0]=0
            Pn10_m1=nbinom(n,1-p).pmf(0)
            Pn10_m1[0]=1.
            Pm1_f_adj=np.exp(np.log(1-Pn10_m1)-np.log(1-np.dot(Pm1_f.pmf(mvec),Pn10_m1))+np.log(Pm1_f.pmf(mvec))) #adds m-dependent factor due to conditioning on n>0...
            if np.sum(Pm1_f_adj)==0:
                qxx_m1_samples[f_start_ind[it]:f_start_ind[it]+f_counts[it]]=1 #minimum cell size conditional on n>0
            else:
                Pm1_f_adj_obj=rv_discrete(name='nbinom_adj',values=(mvec,Pm1_f_adj/np.sum(Pm1_f_adj)))
                qxx_m1_samples[f_start_ind[it]:f_start_ind[it]+f_counts[it]]=Pm1_f_adj_obj.rvs(size=f_counts[it])

            mvals,minds,m_counts=np.unique(qxx_m1_samples[f_start_ind[it]:f_start_ind[it]+f_counts[it]],return_inverse=True,return_counts=True)
            for mit,m in enumerate(mvals):
                m1=r_c1*m
                v1=m1+beta_mv*np.power(m1,alpha_mv)
                if m==0:
                  p=1.
                  n=0.
                else:
                  p=1-m1/v1
                  n=m1*m1/v1/p
                Pn1_m1=nbinom(n,1-p)
                samples=np.random.random(size=m_counts[mit]) * (1-Pn1_m1.cdf(0)) + Pn1_m1.cdf(0)                        
                qxx_n1_samples[f_start_ind[it]:f_start_ind[it]+f_counts[it]][minds==mit]=Pn1_m1.ppf(samples)            
                
            Pm2_f=poisson(m_total*fvec[find])
            
            m2=r_c2*mvec
            v2=m2+beta_mv*np.power(m2,alpha_mv)
            p=1-m2/v2
            n=m2*m2/v2/p
            p[0]=1
            n[0]=0
            Pn20_m2=nbinom(n,1-p).pmf(0)
            Pn20_m2[0]=1.
            Pm2_f_adj=np.exp(np.log(1-Pn20_m2)-np.log(1-np.dot(Pm2_f.pmf(mvec),Pn20_m2))+np.log(Pm2_f.pmf(mvec))) #adds m-dependent factor due to conditioning on n>0...
            if np.sum(Pm1_f_adj)==0:
                qxx_m2_samples[f_start_ind[it]:f_start_ind[it]+f_counts[it]]=1 #minimum cell size conditional on n>0
            else:
                Pm2_f_adj_obj=rv_discrete(name='nbinom_adj',values=(mvec,Pm2_f_adj/np.sum(Pm2_f_adj)))
                qxx_m2_samples[f_start_ind[it]:f_start_ind[it]+f_counts[it]]=Pm2_f_adj_obj.rvs(size=f_counts[it])

            mvals,minds,m_counts=np.unique(qxx_m2_samples[f_start_ind[it]:f_start_ind[it]+f_counts[it]],return_inverse=True,return_counts=True)
            for mit,m in enumerate(mvals):
                m2=r_c2*m
                v2=m2+beta_mv*np.power(m2,alpha_mv)
                if m==0:
                  p=1.
                  n=0.
                else:
                  p=1-m2/v2
                  n=m2*m2/v2/p
                Pn2_m2=nbinom(n,1-p)
                samples=np.random.random(size=m_counts[mit]) * (1-Pn2_m2.cdf(0)) + Pn2_m2.cdf(0)                        
                qxx_n2_samples[f_start_ind[it]:f_start_ind[it]+f_counts[it]][minds==mit]=Pn2_m2.ppf(samples)                
        elif acq_model_type>1:
            samples=np.random.random(size=f_counts[it]) * (1-Pn1_f[find].cdf(0)) + Pn1_f[find].cdf(0)
            qxx_n1_samples[f_start_ind[it]:f_start_ind[it]+f_counts[it]]=Pn1_f[find].ppf(samples)
            samples=np.random.random(size=f_counts[it]) * (1-Pn2_f[find].cdf(0)) + Pn2_f[find].cdf(0)
            qxx_n2_samples[f_start_ind[it]:f_start_ind[it]+f_counts[it]]=Pn2_f[find].ppf(samples)
        else:
            print('acq_model is 0,1,2, or 3 only')
            
    qxx_pair_samples=np.hstack((qxx_n1_samples[:,np.newaxis],qxx_n2_samples[:,np.newaxis]))
    
    pair_samples=np.vstack((q0x_pair_samples,qx0_pair_samples,qxx_pair_samples))
    f_samples=np.concatenate((q0x_f_samples,qx0_f_samples,qxx_f_samples))
    output_m_samples=False
    if acq_model_type<2 and output_m_samples:                
        m1_samples=np.concatenate((q0x_m1_samples,qx0_m1_samples,qxx_m1_samples))
        m2_samples=np.concatenate((q0x_m2_samples,qx0_m2_samples,qxx_m2_samples))
    
    pair_samples_df=pd.DataFrame({'Clone_count_1':pair_samples[:,0],'Clone_count_2':pair_samples[:,1]})
    
    return f_samples,pair_samples_df

def get_diffexpr_model_sample_all(acq_model_type,logPsvec,svec,logfvecwide,f2s_step,null_paras,Nreads,Nclones,seed,nofmin):
    '''
    outputs an array of observed clone frequencies and corresponding dataframe of pair counts.
    Optionally Sets fmin such that normalization condition satisfied, so can choose any values of other null model parameters
    Sampling is of the full repertoire, after which unseen clones are removed.
    Sampling is done efficiently by creating a single index over f and s, and batch sampling clones with the same index 
    TODO add 2-step null models, currently only type 2 (NegBinonly)
    '''
    
    logrhofvec,logfvec = get_rhof(null_paras[0],np.power(10,null_paras[-1]),freq_dtype='float32')
    
    alpha_rho=null_paras[0]    
    
    if nofmin:#given null model paras, except for fmin, obtain fmin as consistent with N<f>=1 constraint (n.b. thif only address null model...)
        def fmin_func(logfmin,alpha_rho,Nclones):
            fmin=np.power(10.,logfmin)
            logrhofvec,logfvec = get_rhof(alpha_rho,fmin,freq_dtype='float32') #low precision here to be consistent with low precision freq sampling below
            dlogfbby2=np.diff(logfvec)/2.
            integ=np.exp(logrhofvec+2*logfvec,dtype='float64') #map back to regular precision
            return np.exp(np.log(Nclones)+np.log(np.sum(dlogfby2*(integ[1:] + integ[:-1]))))-1   # N*<f> = 1 constraint
        fmin_func_part=partial(fmin_func,alpha_rho=alpha_rho,Nclones=Nclones)
        from scipy.optimize import fsolve
        logfmin_guess = -9
        logfmin_sol= fsolve(fmin_func_part, logfmin_guess)
        fmin=np.power(10.,logfmin_sol[0])
        print('logfmin: '+str(logfmin_sol))
        paras=null_paras+[np.log10(fmin)]
    else:
        paras=deepcopy(null_paras)
        fmin=np.power(10,paras[-1])
    dlogfby2=np.asarray(np.diff(logfvec)/2.,dtype='float64')
    
    np.random.seed(seed+1)
    n1_samples=np.zeros(Nclones)
    n2_samples=np.zeros(Nclones)  
    
    #sample in f space and s space
    integ=np.exp(logrhofvec[np.newaxis,:]+logfvec[np.newaxis,:])
    f_samples_inds=get_distsample(np.asarray((dlogfby2[np.newaxis,:]*(integ[:,1:]+integ[:,:-1])).flatten(),dtype='float64'),Nclones,dtype='uint32').flatten() #n.b. input flattened; output sorted
    s_samples_inds=np.random.permutation(get_distsample(np.exp(logPsvec),Nclones,dtype='uint32').flatten()) #n.b. input flattened; output sorted
    
    #implement normalization directly:
    shift=np.log(np.sum(np.exp(logfvec[f_samples_inds]+svec[s_samples_inds]))) - np.log(np.sum(np.exp(logfvec[f_samples_inds])))
    logfvecwide_shift=logfvecwide- shift
    
    Pn_f=np.empty((len(logfvecwide),),dtype=object) #define a new Pn2_f on shifted domain at each iteration        
    if acq_model_type==3:
        #n1_samples=np.array(np.random.poisson(lam=NreadsI*np.exp(logfvec[f_samples_inds])),dtype='uint32')
        meannvec=np.exp(logfvecwide_shift)*Nreads
        for find,mean_n in enumerate(meannvec):
            Pn_f[find]=poisson(mean_n)
    elif acq_model_type==2:
        beta_mv=paras[1]
        alpha_mv=paras[2]
        m=float(Nreads)*np.exp(logfvecwide_shift)
        v=m+beta_mv*np.power(m,alpha_mv)
        pvec=1-m/v
        nvec=m*m/v/pvec
        for find,(n,p) in enumerate(zip(nvec,pvec)):
            Pn_f[find]=nbinom(n,1-p)
    elif acq_model_type==1:
        print('not implemented yet!')
    elif acq_model_type==0:
        print('not implemented yet!')
        #not sure if the memory requirements of Nclones=1e9 are feasible here since acq_model_type==2 almost maxed out my 32GB machine, 
        #and here we would need the intermediate m variable. I guess we could always clear the preceding variable in the chain...
         
    fs_samples_inds=s_samples_inds*len(logfvec)+f_samples_inds
    fsind_vals,fs_start_ind,fs_counts=np.unique(fs_samples_inds,return_counts=True,return_index=True)
    smaxind=(len(svec)-1)/2
    for it,fsind in enumerate(fsind_vals):
        sind,find=np.unravel_index(fsind,(len(svec),len(logfvec)))
        n1_samples[fs_start_ind[it]:fs_start_ind[it]+fs_counts[it]]=np.array(Pn_f[int(smaxind*f2s_step+find)].rvs(fs_counts[it]),dtype='uint32')
        n2_samples[fs_start_ind[it]:fs_start_ind[it]+fs_counts[it]]=np.array(Pn_f[int(   sind*f2s_step+find)].rvs(fs_counts[it]),dtype='uint32')
    
    #reduce to observed samples only
    obs=np.logical_or(n1_samples>0,n2_samples>0)
    n1_samples=n1_samples[obs]
    n2_samples=n2_samples[obs]
    f1_samples=np.exp(logfvec[f_samples_inds[obs]])

    fs_sample_arr=np.exp(logfvec[np.newaxis,:]+svec[:,np.newaxis]).flatten() #untested
    f2_samples=fs_sample_arr[fs_samples_inds[obs]]
    
    pair_samples_df=pd.DataFrame({'Clone_count_1':n1_samples,'Clone_count_2':n2_samples})

    return f1_samples,f2_samples,pair_samples_df




































