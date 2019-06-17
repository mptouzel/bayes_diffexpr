import numpy as np
from lib.model import *
from scipy.optimize import minimize

def constr_fn(paras,svec,sparse_rep,acq_model_type,constr_type): #additional inputs???
    '''
    returns the two level-set functions: log<f>-log(1/N), with N=Nclones/(1-P(0,0)) and log(Z_f), with Z_f=N<f>_{n+n'=0} + sum_i^Nclones <f>_{f|n,n'}
    '''
    
    indn1,indn2,sparse_rep_counts,unicountvals_1,unicountvals_2,NreadsI,NreadsII=sparse_rep

    alpha = paras[0] #power law exponent
    fmin = np.power(10,paras[4]) if acq_model_type<2 else np.power(10,paras[3]) 
    logrhofvec,logfvec = get_rhof(paras[0],fmin)
    dlogfby2=np.diff(logfvec)/2. #1/2 comes from trapezoid integration below
    
    if isinstance(svec,np.ndarray):
        logfvecwide=deepcopy(logfvec) #TODO add fvecwide calc here
 
    integ=np.exp(logrhofvec+2*logfvec)
    avgf_ps=np.dot(dlogfby2,integ[:-1]+integ[1:])

    logPn1_f=get_logPn_f(unicountvals_1,NreadsI,logfvec,acq_model_type,paras)
    logPn2_f=get_logPn_f(unicountvals_2,NreadsII,logfvecwide if isinstance(svec,np.ndarray) else logfvec,acq_model_type,paras) #for diff expr with shift use shifted range for wide f2, #contains s-shift for sampled data method
  
    integ=np.exp(logPn1_f[:,0]+logPn2_f[:,0]+logrhofvec+logfvec)
    Pn0n0=np.dot(dlogfby2,integ[1:]+integ[:-1])
    logPnng0=np.log(1-Pn0n0)
    avgf_null_pair=np.exp(logPnng0-np.log(np.sum(sparse_rep_counts)))

    C1=np.log(avgf_ps)-np.log(avgf_null_pair)

    integ = np.exp(logPn1_f[:,0]+logPn2_f[:,0]+logrhofvec+2*logfvec)
    log_avgf_n0n0 = np.log(np.dot(dlogfby2,integ[1:]+integ[:-1]))

    integ=np.exp(logPn1_f[:,indn1]+logPn2_f[:,indn2]+logrhofvec[:,np.newaxis]+logfvec[:,np.newaxis])
    log_Pn1n2=np.log(np.sum(dlogfby2[:,np.newaxis]*(integ[1:,:] + integ[:-1,:]),axis=0))
    integ=np.exp(np.log(integ)+logfvec[:,np.newaxis]) 
    tmp=deepcopy(log_Pn1n2)
    tmp[tmp==-np.Inf]=np.Inf #since subtracted in next line
    avgf_n1n2=    np.exp(np.log(np.sum(dlogfby2[:,np.newaxis]*(integ[1:,:] + integ[:-1,:]),axis=0))-tmp)
    log_sumavgf=np.log(np.dot(sparse_rep_counts,avgf_n1n2))

    logNclones = np.log(np.sum(sparse_rep_counts))-logPnng0
    Z     = np.exp(logNclones + np.log(Pn0n0) + log_avgf_n0n0) + np.exp(log_sumavgf)
    
    C2=np.log(Z)
    #print('C1:'+str(C1)+' C2:'+str(C2))
    if constr_type==0:
        return C1
    elif constr_type==1:
        return C2
    else:
        return C1,C2

def callback(paras,prtfn,nparas): #acq_model_type dependent
    '''prints iteration info. called by scipy.minimize'''
    global curr_iter
    prtfn(''.join(['{0:d} ']+['{'+str(it)+':3.6f} ' for it in range(1,len(paras)+1)]).format(*([curr_iter]+list(paras))))
    curr_iter += 1

def learn_null_model(sparse_rep,acq_model_type,init_paras,constr_type=2,prtfn=print):
    '''
    performs constrained maximization of null model likelihood
    '''
    if acq_model_type<2:
        parameter_labels=['alph_rho', 'beta','alpha','m_total','fmin']
    elif acq_model_type==2:
        parameter_labels=['alph_rho', 'beta','alpha','fmin']
    else:
        parameter_labels=['alph_rho', 'fmin']
    assert len(parameter_labels)==len(init_paras), "number of model and initial paras differ!"
    condict={'type':'eq','fun':constr_fn,'args': (-1,sparse_rep,acq_model_type,constr_type)}
    partialobjfunc=partial(get_Pn1n2_s,svec=-1, sparse_rep=sparse_rep,acq_model_type=acq_model_type)
    nullfunctol=1e-6
    nullmaxiter=200
    header=['Iter']+parameter_labels
    prtfn(''.join(['{'+str(it)+':9s} ' for it in range(len(init_paras)+1)]).format(*header))
    global curr_iter
    curr_iter = 1
    callbackp=partial(callback,prtfn=prtfn,nparas=len(init_paras))
    outstruct = minimize(partialobjfunc, init_paras, method='SLSQP', callback=callbackp, constraints=condict,options={'ftol':nullfunctol ,'disp': True,'maxiter':nullmaxiter})
    constr_value=constr_fn(outstruct.x,-1,sparse_rep,acq_model_type,constr_type)
    prtfn(outstruct)
    return outstruct,constr_value
   
#-------fucntions for polishing P(s) parameter estimates after grid search----------

def get_likelihood(paras,null_paras,svec,smax,s_step,indn1_d,indn2_d,fvec,fvecwide,rhofvec,\
                                 unicountvals_1_d,unicountvals_2_d,countpaircounts_d,\
                                 NreadsI, NreadsII, nfbins,f2s_step,\
                                 m_low,m_high,mvec,Nsamp,logPn1_f,acq_model_type):
    logfvec=np.log(fvec)
    dlogf=np.diff(logfvec)/2.
    logrhofvec=np.log(rhofvec)
    alpha_rho = null_paras[0]
    if acq_model_type<2: #acq_model_type: 0=NB->Pois, 1=Pois->NB, 2=NB, 3=Pois 
        m_total=float(np.power(10, null_paras[3]))
        r_c1=NreadsI/m_total 
        r_c2=NreadsII/m_total     
        r_cvec=(r_c1,r_c2)
        fmin=np.power(10,null_paras[4])
    else:
        fmin=np.power(10,null_paras[3])

    if acq_model_type<3:
        beta_mv= null_paras[1]
        alpha_mv=null_paras[2]
    
    Ps = get_Ps_pm(np.power(10,paras[0]),0,0,paras[1],smax,s_step)

    logPsvec=np.log(Ps)
    shift=paras[-1]
    

    fvecwide_shift=np.exp(np.log(fvecwide)-shift) #implements shift in Pn2_fs
    svec_shift=svec-shift
    unicounts=unicountvals_2_d
    Pn2_f=np.zeros((len(fvecwide_shift),len(unicounts)))
    if acq_model_type==0:
        mean_m=m_total*fvecwide_shift
        var_m=mean_m+beta_mv*np.power(mean_m,alpha_mv)
        Poisvec=PoisPar(mvec*r_cvec[1],unicounts)
        for f_it in range(len(fvecwide_shift)):
            NBvec=NegBinPar(mean_m[f_it],var_m[f_it],mvec)
            for n_it,n in enumerate(unicounts):
                Pn2_f[f_it,n_it]=np.dot(NBvec[m_low[n_it]:m_high[n_it]+1],Poisvec[m_low[n_it]:m_high[n_it]+1,n_it]) 
    elif acq_model_type==1:
        Poisvec=PoisPar(m_total*fvecwide_shift,mvec)
        mean_n=r_cvec[1]*mvec
        NBmtr=NegBinParMtr(mean_n,mean_n+beta_mv*np.power(mean_m,alpha_mv),unicounts)
        for f_it in range(len(fvecwide_shift)):
            Poisvectmp=Poisvec[f_it,:]
            for n_it,n in enumerate(unicounts):
                Pn2_f[f_it,n_it]=np.dot(Poisvectmp[m_low[n_it]:m_high[n_it]+1],NBmtr[m_low[n_it]:m_high[n_it]+1,n_it]) 
    elif acq_model_type==2:
        mean_n=Nreadsvec[1]*fvecwide_shift
        var_n=mean_n+beta_mv*np.power(mean_n,alpha_mv)
        Pn2_f=NegBinParMtr(mean_n,var_n,unicounts)
    else:# acq_model_type==3:
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
                            m_low,m_high,mvec,Nsamp,logPn1_f,acq_model_type):
    #Ps = get_Ps_pm(np.power(10,paras[0]),np.power(10,paras[1]),paras[2],paras[3],smax,s_step)
    Ps = get_Ps_pm(np.power(10,paras[0]),0,0,paras[1],smax,s_step)

    logPsvec=np.log(Ps) 
    shift=paras[-1]

    alpha_rho = null_paras[0]
    if acq_model_type<2: #acq_model_type: 0=NB->Pois, 1=Pois->NB, 2=NB, 3=Pois 
        m_total=float(np.power(10, null_paras[3]))
        r_c1=NreadsI/m_total 
        r_c2=NreadsII/m_total      
        r_cvec=(r_c1,r_c2)
        fmin=np.power(10,null_paras[4])
    else:
        fmin=np.power(10,null_paras[3])

    if acq_model_type<3:
        beta_mv= null_paras[1]
        alpha_mv=null_paras[2]

    logfvec=np.log(fvec)
    dlogf=np.diff(logfvec)/2.
    logrhofvec=np.log(rhofvec)
    fvecwide_shift=np.exp(np.log(fvecwide)-shift) #implements shift in Pn2_fs
    svec_shift=svec-shift
    unicounts=unicountvals_2_d
    Pn2_f=np.zeros((len(fvecwide_shift),len(unicounts)))
    if acq_model_type==0:
        mean_m=m_total*fvecwide_shift
        var_m=mean_m+beta_mv*np.power(mean_m,alpha_mv)
        Poisvec=PoisPar(mvec*r_cvec[1],unicounts)
        for f_it in range(len(fvecwide_shift)):
            NBvec=NegBinPar(mean_m[f_it],var_m[f_it],mvec)
            for n_it,n in enumerate(unicounts):
                Pn2_f[f_it,n_it]=np.dot(NBvec[m_low[n_it]:m_high[n_it]+1],Poisvec[m_low[n_it]:m_high[n_it]+1,n_it]) 
    elif acq_model_type==1:
        Poisvec=PoisPar(m_total*fvecwide_shift,mvec)
        mean_n=r_cvec[1]*mvec
        NBmtr=NegBinParMtr(mean_n,mean_n+beta_mv*np.power(mean_m,alpha_mv),unicounts)
        for f_it in range(len(fvecwide_shift)):
            Poisvectmp=Poisvec[f_it,:]
            for n_it,n in enumerate(unicounts):
                Pn2_f[f_it,n_it]=np.dot(Poisvectmp[m_low[n_it]:m_high[n_it]+1],NBmtr[m_low[n_it]:m_high[n_it]+1,n_it]) 
    elif acq_model_type==2:
        mean_n=Nreadsvec[1]*fvecwide_shift
        var_n=mean_n+beta_mv*np.power(mean_n,alpha_mv)
        Pn2_f=NegBinParMtr(mean_n,var_n,unicounts)
    else:# acq_model_type==3:
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