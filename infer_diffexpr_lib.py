import numpy as np
import math
import pandas as pd
from functools import partial
from copy import deepcopy
import ctypes
mkl_rt = ctypes.CDLL('libmkl_rt.so')
num_threads=4
mkl_set_num_threads = mkl_rt.MKL_Set_Num_Threads(num_threads)

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
    NBvec=np.arange(nmax+1,dtype=float)
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
    assert Mvec[0]==0, "first element needs to be zero"
    nmax=unicountvals[-1]
    nlen=len(unicountvals)
    mlen=len(Mvec)
    Nvec=unicountvals
    logNvec=-np.insert(np.cumsum(np.log(np.arange(1,nmax+1))),0,0.)[unicountvals] #avoid n=0 nans  
    Nmtr=np.exp(Nvec[np.newaxis,:]*np.log(Mvec)[:,np.newaxis]+logNvec[np.newaxis,:]-Mvec[:,np.newaxis]) # np.log(Mvec) throws warning: since log(0)=-inf
    if Mvec[0]==0:
        Nmtr[0,:]=np.zeros((nlen,)) #when m=0, n=0, and so get rid of nans from log(0)
        Nmtr[0,0]=1. #handled below
    if unicountvals[0]==0: #if n=0 included get rid of nans from log(0)
        Nmtr[:,0]=np.exp(-Mvec)
    return Nmtr
  
def get_distsample(pmf,Nsamp,dtype='uint16'):
    '''
    generates Nsamp index samples of dtype (e.g. uint16 handles up to 65535 indices) from discrete probability mass function pmf
    '''
    shape = np.shape(pmf)
    sortindex = np.argsort(pmf, axis=None)#uses flattened array
    pmf = pmf.flatten()
    pmf = pmf[sortindex]
    cmf = np.cumsum(pmf)
    choice = np.random.uniform(high = cmf[-1], size = int(float(Nsamp)))
    index = np.searchsorted(cmf, choice)
    index = sortindex[index]
    index = np.unravel_index(index, shape)
    index = np.transpose(np.vstack(index))
    sampled_pairs = np.array(index[np.argsort(index[:,0])],dtype=dtype)
    return sampled_pairs
  
def get_sparserep(counts):
    '''
    Tranforms {(n1,n2)} data stored in pandas dataframe to a sparse 1D representation.
    unicountvals_1(2) are the unique values of n1(2).
    clonecountpair_counts gives the counts of unique pairs.
    indn1(2) is the index of unicountvals_1(2) giving the value of n1(2) in that unique pair.
    len(indn1)=len(indn2)=len(clonecountpair_counts)
    '''
    counts['paircount']=1 #gives a weight of 1 to each observed clone
    clone_counts=counts.groupby(['Clone_count_1','Clone_count_2']).sum()
    clonecountpair_counts=np.asarray(clone_counts.values.flatten(),dtype=int)
    clonecountpair_vals=clone_counts.index.values
    indn1=np.asarray([clonecountpair_vals[it][0] for it in range(len(clonecountpair_counts))],dtype=int)
    indn2=np.asarray([clonecountpair_vals[it][1] for it in range(len(clonecountpair_counts))],dtype=int)
    NreadsI=counts.Clone_count_1.sum()
    NreadsII=counts.Clone_count_2.sum()

    unicountvals_1,indn1=np.unique(indn1,return_inverse=True)
    unicountvals_2,indn2=np.unique(indn2,return_inverse=True)
  
    return indn1,indn2,clonecountpair_counts,unicountvals_1,unicountvals_2,NreadsI,NreadsII

def import_data(path,filename1,filename2,mincount,maxcount,colnames1,colnames2):
    '''
    Reads in Yellow fever data from two datasets and merges based on nt sequence.
    Outputs dataframe of pair counts for all clones.
    Considers clones with counts between mincount and maxcount
    Uses specified column names and headerline in stored fasta file.
    '''
    
    headerline=0 #line number of headerline
    newnames=['Clone_fraction','Clone_count','ntCDR3','AACDR3']    
    with open(path+filename1, 'r') as f:
        F1Frame_chunk=pd.read_csv(f,delimiter='\t',usecols=colnames1,header=headerline)[colnames1]
    with open(path+filename2, 'r') as f:
        F2Frame_chunk=pd.read_csv(f,delimiter='\t',usecols=colnames2,header=headerline)[colnames2]
    F1Frame_chunk.columns=newnames
    F2Frame_chunk.columns=newnames
    suffixes=('_1','_2')
    mergedFrame=pd.merge(F1Frame_chunk,F2Frame_chunk,on=newnames[2],suffixes=suffixes,how='outer')
    for nameit in [0,1]:
        for labelit in suffixes:
            mergedFrame.loc[:,newnames[nameit]+labelit].fillna(int(0),inplace=True)
            if nameit==1:
                mergedFrame.loc[:,newnames[nameit]+labelit].astype(int)
    def dummy(x):
        val=x[0]
        if pd.isnull(val):
            val=x[1]    
        return val
    mergedFrame.loc[:,newnames[3]+suffixes[0]]=mergedFrame.loc[:,[newnames[3]+suffixes[0],newnames[3]+suffixes[1]]].apply(dummy,axis=1) #assigns AA sequence to clones, creates duplicates
    mergedFrame.drop(newnames[3]+suffixes[1], 1,inplace=True) #removes duplicates
    mergedFrame.rename(columns = {newnames[3]+suffixes[0]:newnames[3]}, inplace = True)
    mergedFrame=mergedFrame[[newname+suffix for newname in newnames[:2] for suffix in suffixes]+[newnames[2],newnames[3]]]
    filterout=((mergedFrame.Clone_count_1<mincount) & (mergedFrame.Clone_count_2==0)) | ((mergedFrame.Clone_count_2<mincount) & (mergedFrame.Clone_count_1==0)) #has effect only if mincount>0
    number_clones=len(mergedFrame)
    return number_clones,mergedFrame.loc[((mergedFrame.Clone_count_1<=maxcount) & (mergedFrame.Clone_count_2<=maxcount)) & ~filterout]

def get_rhof(alpha_rho,freq_nbins,fmin,freq_dtype):
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
  
def get_Ps_pm(alp,bet,m_sbar,p_sbar,smax,stp):
    '''
    generates asymmetric exponential distribution over log fold change
    with contraction effect size m_sbar expansion effect size p_sbar and responding fraction alp.
    computed over discrete range of s from -smax to smax in steps of size stp.
    note that the responding fraction has no s=0 contribution.
    '''
    lambp=-stp/p_sbar
    lambm=-stp/m_sbar
    smaxt=round(smax/stp)
    if m_sbar==0:
        Z_p=(np.exp((smaxt+1)*lambp)-1)/(np.exp(lambp)-1)-1 #no s=0 contribution
        Ps=np.zeros(2*int(smaxt)+1)
        Ps[int(smaxt)+1:] = np.exp(lambp*np.fabs(np.arange(           1,int(smaxt)+1)))/Z_p
    else:
        Z_m=(np.exp((smaxt+1)*lambm)-1)/(np.exp(lambm)-1)-1 #no s=0 contribution
        #Z_m=(np.exp((smaxt+1)*lambm)-1)/(np.exp(lambm)-1) #no s=0 contribution
        Z_p=(np.exp((smaxt+1)*lambp)-1)/(np.exp(lambp)-1)-1 #no s=0 contribution
        Ps=np.zeros(2*int(smaxt)+1)
        #Ps[:int(smaxt)]=(1-bet)*np.exp(lambm*np.fabs(np.arange(0-int(smaxt),           0)))/Z_m
        Ps[int(smaxt)+1:]  =bet*np.exp(lambp*np.fabs(np.arange(           1,int(smaxt)+1)))/Z_p
        
        #Ps[:int(smaxt)+1]=(1-bet)*np.exp(lambm*np.fabs(np.arange(0-int(smaxt),           1)))/Z_m
        
        #Ps[:int(smaxt)]=(1-bet)*np.exp(lambm*np.fabs(np.arange(0-int(smaxt),           0)))#/Z_m
        #Ps[int(smaxt)+1:]  =bet*np.exp(lambp*np.fabs(np.arange(           1,int(smaxt)+1)))#/Z_p
        #Ps/=(Zp+Zm)/2
    Ps*=alp
    Ps[int(smaxt)]=(1-alp) #the sole contribution to s=0
    return Ps
  
def constr_fn(paras,NreadsI_d,NreadsII_d,unicountvals_1_d,unicountvals_2_d,indn1_d,indn2_d,countpaircounts_d,case,freq_dtype):
    '''
    function that outputs the <f>=1/N contraint value to be used with scipy.minimize
    total number of clones in repertoire, N, is estimated as N=Nsamp/(1-P(0,0))
    '''
    NreadsI=NreadsI_d
    NreadsII=NreadsII_d
    repfac=NreadsII/NreadsI
    alpha_rho = paras[0]
    Nreadsvec=(NreadsI,NreadsII)
    if case<2:
        m_total=np.power(10,paras[3])
        r_c1=NreadsI/m_total 
        r_c2=repfac*r_c1     
        r_cvec=(r_c1,r_c2)
        fmin=np.power(10,paras[4])
    else:
        fmin=np.power(10,paras[3])
    beta_mv= paras[1]
    alpha_mv=paras[2]
    nfbins=800
    logrhofvec,logfvec = get_rhof(alpha_rho,nfbins,fmin,freq_dtype)
    fvec=np.exp(logfvec)
    dlogf=np.diff(logfvec)/2.
    
    for it in range(2):
        Pn_f=np.zeros((len(fvec),))
        if case==0:
            mvec=np.arange(500)
            mean_m=m_total*fvec
            var_m=mean_m+beta_mv*np.power(mean_m,alpha_mv)
            Poisvec=PoisPar(mvec*r_cvec[it],np.arange(2))[:,0]
            for f_it in range(len(fvec)):
                NBvec=NegBinPar(mean_m[f_it],var_m[f_it],mvec)
                Pn_f[f_it]=np.dot(NBvec,Poisvec) 
        elif case==2:
            mean_m=Nreadsvec[it]*fvec
            var_m=mean_m+beta_mv*np.power(mean_m,alpha_mv)
            m=mean_m
            v=var_m
            p = 1-m/v
            r = m*m/v/p
            Pn_f=np.exp(r*np.log(m/v))
        
        if it==0:
            logPn1_f=np.log(Pn_f) #throws warning
        else:
            logPn2_f=np.log(Pn_f) #throws warning
    
    integ=np.exp(logrhofvec+2*logfvec)
    avgf_pf=np.dot(dlogf,integ[:-1]+integ[1:])
    
    integ=np.exp(logPn1_f+logPn2_f+logrhofvec+logfvec)
    Pn0n0=np.dot(dlogf,integ[1:]+integ[:-1])
    avgf_null_pair=np.exp(np.log(1-Pn0n0)-np.log(np.sum(countpaircounts_d)))
    return avgf_pf-avgf_null_pair
 
#@profile
def get_Pn1n2_s(paras, svec, unicountvals_1, unicountvals_2, NreadsI, NreadsII, nfbins, repfac, indn1=None ,indn2=None,countpaircounts_d=None,case=0,freq_dtype='float32',  s_step=0):    
    #svec determines which of 3 run modes is evaluated
    #1) svec is array => compute P(n1,n2|s),           output: Pn1n2_s,unicountvals_1,unicountvals_2,Pn1_f,fvec,Pn2_s,svec
    #2) svec=-1       => null model likelihood,        output: data-averaged loglikelihood
    #3) else          => compute null model, P(n1,n2), output: Pn1n2_s,unicountvals_1,unicountvals_2,Pn1_f,Pn2_f,fvec
    
    #case input is which P(n|f) model to use. 0:NB->Pois,1:Pois->NB,2:NBonly,3:Poisonly
    #repfac is a factor that scales the average number of reads/cell. Often set to NreadsII/NreadsI
    #paras is the list of null model parameters, length depends on the case
    #e.g. for case=0, paras=(alpha,beta_mv,alpha_mv,log_m_total,log_fmin)
    #mean variance relation: v = m + beta_mv*m^alpha_mv
    alpha = paras[0] #power law exponent
    if case<2:
        m_total=float(np.power(10, paras[3])) 
        r_c1=NreadsI/m_total 
        r_c2=repfac*r_c1  
        fmin=np.power(10,paras[4])
    else:
        fmin=np.power(10,paras[3])

    beta_mv= paras[1]
    alpha_mv=paras[2]
    
    logrhofvec,logfvec = get_rhof(alpha,nfbins,fmin,freq_dtype)
    
    dlogfby2=np.diff(logfvec)/2.

    logf_step=logfvec[1] - logfvec[0] #use natural log here since f2 increments in increments in exp().  
    
    if isinstance(svec,np.ndarray):
        smaxind=(len(svec)-1)/2
        f2s_step=int(round(s_step/logf_step)) #rounded number of f-steps in one s-step
        logfmin=logfvec[0 ]-f2s_step*smaxind*logf_step
        logfmax=logfvec[-1]+f2s_step*smaxind*logf_step
        logfvecwide=np.linspace(logfmin,logfmax,len(logfvec)+2*smaxind*f2s_step) #a wider domain for the second frequency f2=f1*exp(s)
        
    #compute P(n1|f) and P(n2|f), each in an iteration of the following loop
    Nreadsvec=(NreadsI,NreadsII)
    r_cvec=(r_c1,r_c2)
    for it in range(2):
        if it==0:
            unicounts=unicountvals_1
            logfvec_tmp=deepcopy(logfvec)
        else:
            unicounts=unicountvals_2
            if isinstance(svec,np.ndarray): #for diff expr with shift use shifted range for wide f2
                logfvec_tmp=deepcopy(logfvecwide) #contains s-shift for sampled data method
        if case<2:
            #compute range of m values (number of cells) over which to sum for a given n value (reads) in the data 
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

        Pn_f=np.zeros((len(logfvec_tmp),len(unicounts)))
        if case==0:
            mean_m=m_total*np.exp(logfvec_tmp)
            var_m=mean_m+beta_mv*np.power(mean_m,alpha_mv)
            Poisvec=PoisPar(mvec*r_cvec[it],unicounts)
            for f_it in range(len(logfvec)):
                NBvec=NegBinPar(mean_m[f_it],var_m[f_it],mvec)
                for n_it,n in enumerate(unicounts):
                    Pn_f[f_it,n_it]=np.dot(NBvec[m_low[n_it]:m_high[n_it]+1],Poisvec[m_low[n_it]:m_high[n_it]+1,n_it]) 
        elif case==1:
            Poisvec=PoisPar(m_total*np.exp(logfvec_tmp),mvec)
            mean_n=r_cvec[it]*mvec
            NBmtr=NegBinParMtr(mean_n,mean_n+beta_mv*np.power(mean_m,alpha_mv),unicounts)
            for f_it in range(len(logfvec)):
                Poisvectmp=Poisvec[f_it,:]
                for n_it,n in enumerate(unicounts):
                    Pn_f[f_it,n_it]=np.dot(Poisvectmp[m_low[n_it]:m_high[n_it]+1],NBmtr[m_low[n_it]:m_high[n_it]+1,n_it]) 
        elif case==2:
            mean_n=Nreadsvec[it]*np.exp(logfvec_tmp)
            var_n=mean_n+beta_mv*np.power(mean_n,alpha_mv)
            Pn_f=NegBinParMtr(mean_n,var_n,unicounts)
        else:# case==3:
            mean_n=Nreadsvec[it]*np.exp(logfvec_tmp)
            Pn_f=PoisPar(mean_n,unicounts)

        if it==0:
            logPn1_f=np.log(Pn_f)
        else:
            logPn2_f=Pn_f
            logPn2_f=np.log(logPn2_f) #throws warning  
            
    
    if isinstance(svec,np.ndarray):  #diffexpr model

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
      
    elif svec==-1: #scalar marginal likelihood
        
        integ=np.exp(logrhofvec + logPn2_f[:,0] + logPn1_f[:,0] + logfvec)
        #Pn0n0 = dlogfby2[0]*np.sum(integ[1:] + integ[:-1])
        Pn0n0 = np.dot(dlogfby2,integ[1:] + integ[:-1])

        Pn1n2_s=np.zeros(len(countpaircounts_d)) #1D representation
        for it,(ind1,ind2) in enumerate(zip(indn1,indn2)):
            integ=np.exp(logPn1_f[:,ind1]+logrhofvec+logPn2_f[:,ind2]+logfvec)
            #Pn1n2_s[it] = dlogfby2[0]*np.sum(integ[1:] + integ[:-1])
            Pn1n2_s[it] = np.dot(dlogfby2,integ[1:] + integ[:-1])
        Pn1n2_s/=1.-Pn0n0 #renormalize
        return -np.dot(countpaircounts_d,np.where(Pn1n2_s>0,np.log(Pn1n2_s),0))/float(np.sum(countpaircounts_d))
    
    else: #s=0 (null model)        
        print('running Null Model, ')        
        Pn1n2_s=np.zeros((len(unicountvals_1),len(unicountvals_2))) #2D representation 
        for n2_it,n2 in enumerate(unicountvals_2): 
            for n1_it,n1 in enumerate(unicountvals_1):
                integ=np.exp(logPn1_f[:,n1_it]+logrhofvec+logPn2_f[:,n2_it]+logfvec)
                Pn1n2_s[n1_it,n2_it] = np.dot(dlogfby2,integ[1:] + integ[-1])
        Pn1n2_s/=1.-Pn1n2_s[0,0] #remove (n1,n2)=(0,0) and renormalize
        Pn1n2_s[0,0]=0.
        return Pn1n2_s,unicountvals_1,unicountvals_2,Pn1_f,Pn2_f,logfvec

def get_likelihood(paras,null_paras,svec,smax,s_step,indn1_d,indn2_d,fvec,fvecwide,rhofvec,\
                                 unicountvals_1_d,unicountvals_2_d,countpaircounts_d,\
                                 NreadsI, NreadsII, nfbins,f2s_step,\
                                 m_low,m_high,mvec,Nsamp,r_cvec,logPn1_f,case):
    logfvec=np.log(fvec)
    dlogf=np.diff(logfvec)/2.
    logrhofvec=np.log(rhofvec)
    alpha_rho = null_paras[0]
    if case<2: #case: 0=NB->Pois, 1=Pois->NB, 2=NB, 3=Pois 
        m_total=float(np.power(10, null_paras[3]))
        r_c1=NreadsI/m_total 
        r_c2=NreadsII/m_total     
        r_cvec=(r_c1,r_c2)
        fmin=np.power(10,null_paras[4])
    else:
        fmin=np.power(10,null_paras[3])

    if case<3:
        beta_mv= null_paras[1]
        alpha_mv=null_paras[2]
    
    Ps = get_Ps_pm(np.power(10,paras[0]),0,0,paras[1],smax,s_step)

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
    return np.dot(countpaircounts_d/float(Nsamp),np.where(tmp>0,np.log(tmp),0))

      
def callbackFnull(Xi): #case dependent
    '''prints iteration info. called scipy.minimize'''
    print('{0: 3.6f}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}  {4: 3.6f}'.format(Xi[0], Xi[1], Xi[2], Xi[3],Xi[4])+'\n')    #case=0
    #print('{0: 3.6f}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f} '.format(Xi[0], Xi[1], Xi[2], Xi[3])+'\n')                    #case=1
    #print('{0: 3.6f}   {1: 3.6f}   {2: 3.6f}   '.format(np.power(10,Xi[0]), np.power(10,Xi[1]), Xi[2])+'\n')           #case=2
    #print('{0: 3.6f}   {1: 3.6f}   '.format(np.power(10,Xi[0]), np.power(10,Xi[1]))+'\n')                               #case=3
def callbackFdiffexpr(Xi): #case dependent
    '''prints iteration info. called scipy.minimize'''
    print('{0: 3.6f}   {1: 3.6f}   {2: 3.6f}   '.format(np.power(10,Xi[0]), np.power(10,Xi[1]), Xi[2])+'\n')           
    #print('{0: 3.6f}   {1: 3.6f}   '.format(np.power(10,Xi[0]), np.power(10,Xi[1]))+'\n')                             
    
def save_table(outpath, svec, Ps,Pn1n2_s, Pn0n0_s,  subset, unicountvals_1_d, unicountvals_2_d,indn1_d,indn2_d,print_expanded=True, pthresh=0.1, smedthresh=3.46):
    '''
    takes learned diffexpr model, Pn1n2_s*Ps, computes posteriors over (n1,n2) pairs, and writes to file a table of data with clones as rows and columns as measures of thier posteriors 
    print_expanded=True orders table as ascending by , else descending
    pthresh is the threshold in 'p-value'-like (null hypo) probability, 1-P(s>0|n1_i,n2_i), where i is the row (i.e. the clone) n.b. lower null prob implies larger probability of expansion
    smedthresh is the threshold on the posterior median, below which clones are discarded
    '''

    Psn1n2_ps=Pn1n2_s*Ps[:,np.newaxis,np.newaxis] 
    
    #compute marginal likelihood (neglect renormalization , since it cancels in conditional below) 
    Pn1n2_ps=np.sum(Psn1n2_ps,0)

    Ps_n1n2ps=Pn1n2_s*Ps[:,np.newaxis,np.newaxis]/Pn1n2_ps[np.newaxis,:,:]
    #compute cdf to get p-value to threshold on to reduce output size
    cdfPs_n1n2ps=np.cumsum(Ps_n1n2ps,0)
    

    def dummy(row,cdfPs_n1n2ps,unicountvals_1_d,unicountvals_2_d):
        '''
        when applied to dataframe, generates 'p-value'-like (null hypo) probability, 1-P(s>0|n1_i,n2_i), where i is the row (i.e. the clone)
        '''
        return cdfPs_n1n2ps[np.argmin(np.fabs(svec)),row['Clone_count_1']==unicountvals_1_d,row['Clone_count_2']==unicountvals_2_d][0]
    dummy_part=partial(dummy,cdfPs_n1n2ps=cdfPs_n1n2ps,unicountvals_1_d=unicountvals_1_d,unicountvals_2_d=unicountvals_2_d)
    
    cdflabel=r'$1-P(s>0)$'
    subset[cdflabel]=subset.apply(dummy_part, axis=1)
    subset=subset[subset[cdflabel]<pthresh].reset_index(drop=True)

    #go from clone count pair (n1,n2) to index in unicountvals_1_d and unicountvals_2_d
    data_pairs_ind_1=np.zeros((len(subset),),dtype=int)
    data_pairs_ind_2=np.zeros((len(subset),),dtype=int)
    for it in range(len(subset)):
        data_pairs_ind_1[it]=np.where(int(subset.iloc[it].Clone_count_1)==unicountvals_1_d)[0]
        data_pairs_ind_2[it]=np.where(int(subset.iloc[it].Clone_count_2)==unicountvals_2_d)[0]   
    #posteriors over data clones
    Ps_n1n2ps_datpairs=Ps_n1n2ps[:,data_pairs_ind_1,data_pairs_ind_2]
    
    #compute posterior metrics
    mean_est=np.zeros((len(subset),))
    max_est= np.zeros((len(subset),))
    slowvec= np.zeros((len(subset),))
    smedvec= np.zeros((len(subset),))
    shighvec=np.zeros((len(subset),))
    pval=0.025 #double-sided comparison statistical test
    pvalvec=[pval,0.5,1-pval] #bound criteria defining slow, smed, and shigh, respectively
    for it,column in enumerate(np.transpose(Ps_n1n2ps_datpairs)):
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
    
    colnames=(r'$\bar{s}$',r'$s_{max}$',r'$s_{3,high}$',r'$s_{2,med}$',r'$s_{1,low}$')
    for it,coldata in enumerate((mean_est,max_est,shighvec,smedvec,slowvec)):
        subset.insert(0,colnames[it],coldata)
    oldcolnames=( 'AACDR3',  'ntCDR3', 'Clone_count_1', 'Clone_count_2', 'Clone_fraction_1', 'Clone_fraction_2')
    newcolnames=('CDR3_AA', 'CDR3_nt',        r'$n_1$',        r'$n_2$',           r'$f_1$',           r'$f_2$')
    subset=subset.rename(columns=dict(zip(oldcolnames, newcolnames)))
    
    #select only clones whose posterior median pass the given threshold
    subset=subset[subset[r'$s_{2,med}$']>smedthresh]
    
    print("writing to: "+outpath)
    if print_expanded:
        subset=subset.sort_values(by=cdflabel,ascending=True)
        strout='expanded'
    else:
        subset=subset.sort_values(by=cdflabel,ascending=False)
        strout='contracted'
    subset.to_csv(outpath+'top_'+strout+'.csv',sep='\t',index=False)
    
#-------fucntions for polishing P(s) parameter estimates----------
def get_likelihood(paras,null_paras,svec,smax,s_step,indn1_d,indn2_d,fvec,fvecwide,rhofvec,\
                                unicountvals_1_d,unicountvals_2_d,countpaircounts_d,\
                                NreadsI, NreadsII, nfbins,f2s_step,\
                                m_low,m_high,mvec,Nsamp,logPn1_f,case):
    logfvec=np.log(fvec)
    dlogf=np.diff(logfvec)/2.
    logrhofvec=np.log(rhofvec)
    alpha_rho = null_paras[0]
    if case<2: #case: 0=NB->Pois, 1=Pois->NB, 2=NB, 3=Pois 
        m_total=float(np.power(10, null_paras[3]))
        r_c1=NreadsI/m_total 
        r_c2=NreadsII/m_total     
        r_cvec=(r_c1,r_c2)
        fmin=np.power(10,null_paras[4])
    else:
        fmin=np.power(10,null_paras[3])

    if case<3:
        beta_mv= null_paras[1]
        alpha_mv=null_paras[2]
    
    #Ps = get_Ps_pm(np.power(10,paras[0]),np.power(10,paras[1]),paras[2],paras[3],smax,s_step)
    Ps = get_Ps_pm(np.power(10,paras[0]),0,0,paras[1],smax,s_step)

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
    return np.dot(countpaircounts_d/float(Nsamp),np.where(tmp>0,np.log(tmp),0))


def constr_fn_diffexpr(paras,null_paras,svec,smax,s_step,indn1_d,indn2_d,fvec,fvecwide,rhofvec,\
                            unicountvals_1_d,unicountvals_2_d,countpaircounts_d,\
                            NreadsI, NreadsII, nfbins,f2s_step,\
                            m_low,m_high,mvec,Nsamp,logPn1_f,case):
    #Ps = get_Ps_pm(np.power(10,paras[0]),np.power(10,paras[1]),paras[2],paras[3],smax,s_step)
    Ps = get_Ps_pm(np.power(10,paras[0]),0,0,paras[1],smax,s_step)

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
        fmin=np.power(10,null_paras[3])

    if case<3:
        beta_mv= null_paras[1]
        alpha_mv=null_paras[2]

    logfvec=np.log(fvec)
    dlogf=np.diff(logfvec)/2.
    logrhofvec=np.log(rhofvec)
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

    Pn2_f=np.zeros((len(fvec),len(unicountvals_2_d)))
    for s_it in range(len(svec)):
        Pn2_f+=np.exp(logPn2_f[f2s_step*s_it:f2s_step*s_it+nfbins,:]+logPsvec[s_it,np.newaxis,np.newaxis])
    log_Pn2_f=np.log(Pn2_f)
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

    log_expsavg_Pn2_f=np.zeros((len(fvec),len(unicountvals_2_d)))
    for s_it in range(len(svec)):
        log_expsavg_Pn2_f+=np.exp(svec_shift[s_it,np.newaxis,np.newaxis]+logPn2_f[f2s_step*s_it:f2s_step*s_it+nfbins,:]+logPsvec[s_it,np.newaxis,np.newaxis]) #cuts down on memory constraints
    log_expsavg_Pn2_f=np.log(log_expsavg_Pn2_f)
    #log_expsavg_Pn2_f=np.log(np.sum(np.exp(svec[:,np.newaxis,np.newaxis]+logPn2_s+logPsvec[:,np.newaxis,np.newaxis]),axis=0))
    integ=np.exp(log_expsavg_Pn2_f[:,indn2_d]+logPn1_f[:,indn1_d]+logrhofvec[:,np.newaxis]+2*logfvec[:,np.newaxis])
    avgfexps_n1n2=np.exp(np.log(np.sum(dlogf[:,np.newaxis]*(integ[1:,:] + integ[:-1,:]),axis=0))-tmp)
    log_avgfexps=np.log(np.dot(countpaircounts_d,avgfexps_n1n2))

    logPn20_s=np.zeros((len(svec),len(fvec))) #svec is svec_shift
    for s_it in range(len(svec)):
        logPn20_s[s_it,:]=logPn2_f[f2s_step*s_it:f2s_step*s_it+len(fvec),0]   #note here this is fvec long on shifted s
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
