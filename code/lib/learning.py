import numpy as np
from lib.model import *
from scipy.optimize import minimize
import numpy.polynomial.polynomial as poly

def callback(paras,prtfn,nparas): 
    '''prints iteration info. called by scipy.minimize'''
    global curr_iter
    prtfn(''.join(['{0:d} ']+['{'+str(it)+':3.6f} ' for it in range(1,len(paras)+1)]).format(*([curr_iter]+list(paras))))
    curr_iter += 1

#---------null model--------------

def nullmodel_constr_fn(paras,svec,sparse_rep,acq_model_type,constr_type):
    '''
    returns either or both of the two level-set functions: log<f>-log(1/N), with N=Nclones/(1-P(0,0)) and log(Z_f), with Z_f=N<f>_{n+n'=0} + sum_i^Nclones <f>_{f|n,n'}
    '''
    
    indn1,indn2,sparse_rep_counts,unicountvals_1,unicountvals_2,NreadsI,NreadsII=sparse_rep

    alpha = paras[0] #power law exponent
    fmin = np.power(10,paras[-1])
    logrhofvec,logfvec = get_rhof(alpha,fmin)
    dlogfby2=np.diff(logfvec)/2. #1/2 comes from trapezoid integration below
 
    integ=np.exp(logrhofvec+2*logfvec)
    avgf_ps=np.dot(dlogfby2,integ[:-1]+integ[1:])

    logPn1_f=get_logPn_f(unicountvals_1,NreadsI, logfvec,acq_model_type,paras)
    logPn2_f=get_logPn_f(unicountvals_2,NreadsII,logfvec,acq_model_type,paras)
  
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

def learn_null_model(sparse_rep,acq_model_type,init_paras,constr_type=1,prtfn=print): #constraint type 1 gives only low error modes, see paper for details.
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
    condict={'type':'eq','fun':nullmodel_constr_fn,'args': (-1,sparse_rep,acq_model_type,constr_type)}
    partialobjfunc=partial(get_Pn1n2_s,svec=-1, sparse_rep=sparse_rep,acq_model_type=acq_model_type)
    nullfunctol=1e-6
    nullmaxiter=200
    header=['Iter']+parameter_labels
    prtfn(''.join(['{'+str(it)+':9s} ' for it in range(len(init_paras)+1)]).format(*header))
    global curr_iter
    curr_iter = 1
    callbackp=partial(callback,prtfn=prtfn,nparas=len(init_paras))
    outstruct = minimize(partialobjfunc, init_paras, method='SLSQP', callback=callbackp, constraints=condict,options={'ftol':nullfunctol ,'disp': True,'maxiter':nullmaxiter})
    constr_value=nullmodel_constr_fn(outstruct.x,-1,sparse_rep,acq_model_type,constr_type)
    
    prtfn(outstruct)
    return outstruct,constr_value


#-------diffexpr model----------

def get_diffexpr_likelihood(diffexpr_paras,null_paras,sparse_rep,acq_model_type,logfvec,logfvecwide,svec,smax,s_step,Ps_type,f2s_step,logPn1_f,logrhofvec,output_unseen=False,maxL=0):
    
    logPsvec = get_logPs_pm(diffexpr_paras[:-1],smax,s_step,Ps_type)
    shift=diffexpr_paras[-1]
    
    dlogfby2=np.diff(logfvec)/2. #1/2 comes from trapezoid integration below
    
    indn1,indn2,sparse_rep_counts,_,unicountvals_2,_,NreadsII=sparse_rep
    
    logfvecwide_shift=logfvecwide-shift #implements shift in Pn2_fs
    svec_shift=svec-shift
    logPn2_f=get_logPn_f(unicountvals_2,NreadsII,logfvecwide_shift,acq_model_type,null_paras)

    logPn2_f_ints=np.zeros((len(logfvec),len(unicountvals_2))) #n.b. underscore
    for s_it in range(len(svec)):
        logPn2_f_ints+=np.exp(logPn2_f[f2s_step*s_it:f2s_step*s_it+len(logfvec),:]+logPsvec[s_it,np.newaxis,np.newaxis])
    logPn2_f_ints=np.log(logPn2_f_ints)
    integ=np.exp(logPn2_f_ints[:,indn2]+logPn1_f[:,indn1]+logrhofvec[:,np.newaxis]+logfvec[:,np.newaxis])
    log_Pn1n2=np.log(np.sum(dlogfby2[:,np.newaxis]*(integ[1:,:] + integ[:-1,:]),axis=0))
    
    integ=np.exp(logPn1_f[:,0]+logPn2_f_ints[:,0]+logrhofvec+logfvec)
    Pn0n0=np.dot(dlogfby2,integ[1:]+integ[:-1])
    if output_unseen:
        return np.dot(sparse_rep_counts/float(np.sum(sparse_rep_counts)),log_Pn1n2-np.log(1-Pn0n0)), Pn0n0
    else:
        return np.log10(-np.dot(sparse_rep_counts/float(np.sum(sparse_rep_counts)),log_Pn1n2-np.log(1-Pn0n0)))
    
def diffexpr_constr_fn(diffexpr_paras,null_paras,sparse_rep,acq_model_type,logfvec,logfvecwide,svec,smax,s_step,Ps_type,f2s_step,logPn1_f,logrhofvec):
    
    logPsvec = get_logPs_pm(diffexpr_paras[:-1],smax,s_step,Ps_type)
    shift=diffexpr_paras[-1]

    indn1,indn2,sparse_rep_counts,_,unicountvals_2,_,NreadsII=sparse_rep
    dlogfby2=np.diff(logfvec)/2. #1/2 comes from trapezoid integration below
    
    #logPn2_f
    logfvecwide_shift=logfvecwide-shift #implements shift in Pn2_fs
    svec_shift=svec-shift
    logPn2_f=get_logPn_f(unicountvals_2,NreadsII,logfvecwide_shift,acq_model_type,null_paras)
    
    #logPn2_f_ints (i.e. after marginalizing s)
    logPn2_f_ints=np.zeros((len(logfvec),len(unicountvals_2)))
    for s_it in range(len(svec)):
        logPn2_f_ints+=np.exp(logPn2_f[f2s_step*s_it:f2s_step*s_it+len(logfvec),:]+logPsvec[s_it,np.newaxis,np.newaxis])
    logPn2_f_ints=np.log(logPn2_f_ints)
    
    #log_avgf
    integ=np.exp(logPn2_f_ints[:,indn2]+logPn1_f[:,indn1]+logrhofvec[:,np.newaxis]+logfvec[:,np.newaxis])
    log_Pn1n2=np.log(np.sum(dlogfby2[:,np.newaxis]*(integ[1:,:] + integ[:-1,:]),axis=0)) #is this low precision?
    integ=np.exp(np.log(integ)+logfvec[:,np.newaxis]) 
    tmp=deepcopy(log_Pn1n2)
    tmp[tmp==-np.Inf]=np.Inf #since subtracted in next line
    avgf_n1n2=    np.exp(np.log(np.sum(dlogfby2[:,np.newaxis]*(integ[1:,:] + integ[:-1,:]),axis=0))-tmp)
    log_avgf=np.log(np.dot(sparse_rep_counts,avgf_n1n2))
    
    #log_avgfexps
    log_expsavg_Pn2_f=np.zeros((len(logfvec),len(unicountvals_2)))
    for s_it in range(len(svec)): #loop over s instead of adding it as a dimension: cuts down significantly on memory constraints
        log_expsavg_Pn2_f+=np.exp(svec_shift[s_it,np.newaxis,np.newaxis]+logPn2_f[f2s_step*s_it:f2s_step*s_it+len(logfvec),:]+logPsvec[s_it,np.newaxis,np.newaxis]) 
    log_expsavg_Pn2_f=np.log(log_expsavg_Pn2_f)
    integ=np.exp(log_expsavg_Pn2_f[:,indn2]+logPn1_f[:,indn1]+logrhofvec[:,np.newaxis]+2*logfvec[:,np.newaxis])
    avgfexps_n1n2=np.exp(np.log(np.sum(dlogfby2[:,np.newaxis]*(integ[1:,:] + integ[:-1,:]),axis=0))-tmp)
    log_avgfexps=np.log(np.dot(sparse_rep_counts,avgfexps_n1n2))
    
    #00
    integ=np.exp(logPn1_f[:,0]+logPn2_f_ints[:,0]+logrhofvec+logfvec)
    logPnn0=np.log(np.dot(dlogfby2,integ[1:]+integ[:-1]))
    log_Pnng0=np.log(1-np.exp(logPnn0))
    logNclones=np.log(np.sum(sparse_rep_counts))-log_Pnng0
    
    #decomposed f averages
    integ = np.exp(logPn1_f[:,0]+logrhofvec+2*logfvec+logPn2_f_ints[:,0])
    log_avgf_n0n0 = np.log(np.dot(dlogfby2,integ[1:]+integ[:-1]))
    
    #decomposed fexps averages
    logPn20_s=np.zeros((len(svec),len(logfvec))) #recompute here since have to integrate over s again, this time with svec in integrand (n.b. svec is svec_shift here)
    for s_it in range(len(svec)):
        logPn20_s[s_it,:]=logPn2_f[f2s_step*s_it:f2s_step*s_it+len(logfvec),0]   
    integ = np.exp(logPn1_f[:,0]+logrhofvec+2*logfvec+np.log(np.sum(np.exp(svec_shift[:,np.newaxis]+logPn20_s+logPsvec[:,np.newaxis]),axis=0)))  #----------svec
    log_avgfexps_n0n0 = np.log(np.dot(dlogfby2,integ[1:]+integ[:-1]))
    
    #final expressions
    Z     = np.exp(logNclones + logPnn0 + log_avgf_n0n0    ) + np.exp(log_avgf)    
    Zdash = np.exp(logNclones + logPnn0 + log_avgfexps_n0n0) + np.exp(log_avgfexps)

    return np.log(Zdash)-np.log(Z)

def learn_diffexpr_model(init_paras,parameter_labels,null_paras,sparse_rep,acq_model_type,logfvec,logfvecwide,\
                         svec,smax,s_step,Ps_type,f2s_step,logPn1_f,logrhofvec,prtfn=print): #constraint type 1 gives only low error modes, see paper for details.
    '''
    performs constrained maximization of diffexpr model likelihood
    '''

    partialobjfunc=partial(get_diffexpr_likelihood,null_paras=null_paras,sparse_rep=sparse_rep,acq_model_type=acq_model_type,logfvec=logfvec,logfvecwide=logfvecwide,\
                                                    svec=svec,smax=smax,s_step=s_step,Ps_type=Ps_type,f2s_step=f2s_step,logPn1_f=logPn1_f,logrhofvec=logrhofvec)
    
    condict={'type':'eq','fun':diffexpr_constr_fn,'args': (null_paras,sparse_rep,acq_model_type,logfvec,logfvecwide,svec,smax,s_step,Ps_type,f2s_step,logPn1_f,logrhofvec)}
    
    nullfunctol=1e-10
    nullmaxiter=300
    header=['Iter']+parameter_labels
    prtfn(''.join(['{'+str(it)+':9s} ' for it in range(len(init_paras)+1)]).format(*header))
    global curr_iter
    curr_iter = 1
    callbackp=partial(callback,prtfn=prtfn,nparas=len(init_paras))
    method_name='SLSQP' #handles constraints
    outstruct = minimize(partialobjfunc, init_paras, method=method_name, callback=callbackp, constraints=condict,options={'ftol':nullfunctol ,'disp': True,'maxiter':nullmaxiter})
    constr_value=diffexpr_constr_fn(outstruct.x,null_paras,sparse_rep,acq_model_type,logfvec,logfvecwide,svec,smax,s_step,Ps_type,f2s_step,logPn1_f,logrhofvec)
    
    return outstruct,constr_value

def MSError_fcn(para,data,diag_pair):
    M=np.diag(diag_pair) #precision matrix
    M[0,1]=para
    M[1,0]=para
    res=0
    for val in data.itertuples():
        res+=np.power(val.y-(-M.dot(val.x).dot(val.x)/2),2)
    return res/len(data)

def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]

def get_fisherinfo_diffexpr_model(opt_paras,null_paras,sparse_rep,acq_model_type,logfvec,logfvecwide,\
                         svec,smax,s_step,Ps_type,f2s_step,logPn1_f,logrhofvec,prtfn=print): #constraint type 1 gives only low error modes, see paper for details.
    '''
    get linearized constraint equation:
        -at p_opt, estimate derivatives dC/dp=(C(eps/2)-C(-eps/2)/eps in each of p=p1,p2,p3 directions, -eps/2, +eps/2, iterative procedure to find eps: half eps until (change in dC)/dC falls below threshold
        -parametrize 2D grid: get grid data from uvec=eps_u*(-2,-1,0,1,2)*du,vvec=eps_v*(-2,-1,0,1,2)*dv, check that gives roughly quadratic form of likelihood by roughly same likelihood at low and high para values 
        -map grid to p-space using normal eqn: dC(p_opt).(p-p_opt)=0: (p1,p2,p3)=p_opt + (u/(1-dC/dp1),v/(1-dC/dp2),(0-dC/dp1*u/(1-dC/dp1)-dC/dp1*v/(1-dC/dp2))*dp3/dC)
        -eval likelihood at grid
        -obtain quadrative fit of likelihoods in (u,v) as 2-form
        -invert and diagonalize to get u and v eigenvectors and two eigenvalues of covariance matrix
        -map eigenvalue*eigenvector in (u,v) space to p-space using above mapping
        -ignore 3rd (shift) component
    '''
    opt_paras=np.asarray(opt_paras)
    
    _,_,sparse_rep_counts,_,_,_,_=sparse_rep
    Nclones=np.sum(sparse_rep_counts)
    
    partialdiffexpr_constr_fn=partial(diffexpr_constr_fn, null_paras=null_paras,sparse_rep=sparse_rep,acq_model_type=acq_model_type,logfvec=logfvec,logfvecwide=logfvecwide,svec=svec,smax=smax,s_step=s_step,Ps_type=Ps_type,f2s_step=f2s_step,logPn1_f=logPn1_f,logrhofvec=logrhofvec)
    Cval=partialdiffexpr_constr_fn(opt_paras)
    
    partialobjfunc=partial(get_diffexpr_likelihood,null_paras=null_paras,sparse_rep=sparse_rep,acq_model_type=acq_model_type,logfvec=logfvec,logfvecwide=logfvecwide,\
                                                    svec=svec,smax=smax,s_step=s_step,Ps_type=Ps_type,f2s_step=f2s_step,logPn1_f=logPn1_f,logrhofvec=logrhofvec)
    Lval=-np.power(10,partialobjfunc(opt_paras))
    print('L:'+str(Lval)+' C:'+str(Cval))
    
    #get constraint gradient
    eps_tol=1e-7
    gradC=[]
    for it,para in enumerate(opt_paras):
    #for it,para in enumerate(np.log10(opt_paras)):
        eps=para/10
        dCdpold=0
        #paras_tmp_plus=deepcopy(np.log10(opt_paras))
        #paras_tmp_minus=deepcopy(np.log10(opt_paras))
        paras_tmp_plus=deepcopy(opt_paras)
        paras_tmp_minus=deepcopy(opt_paras)
        paras_tmp_plus[it]+=eps/2
        paras_tmp_minus[it]-=eps/2
        #dCdp=(partialdiffexpr_constr_fn(np.power(10,paras_tmp_plus))-partialdiffexpr_constr_fn(np.power(10,paras_tmp_minus)))/eps
        dCdp=(partialdiffexpr_constr_fn(paras_tmp_plus)-partialdiffexpr_constr_fn(paras_tmp_minus))/eps

        while np.fabs((dCdp-dCdpold)/dCdp)>eps_tol:
            dCdpold=deepcopy(dCdp)
            eps/=2
            #paras_tmp_plus=deepcopy(np.log10(opt_paras))
            #paras_tmp_minus=deepcopy(np.log10(opt_paras))
            paras_tmp_plus=deepcopy(opt_paras)
            paras_tmp_minus=deepcopy(opt_paras)
            paras_tmp_plus[it]+=eps/2
            paras_tmp_minus[it]-=eps/2
            dCdp=(partialdiffexpr_constr_fn(paras_tmp_plus)-partialdiffexpr_constr_fn(paras_tmp_minus))/eps
            #dCdp=(partialdiffexpr_constr_fn(np.power(10,paras_tmp_plus))-partialdiffexpr_constr_fn(np.power(10,paras_tmp_minus)))/eps
            print('(dCdp-dCdpold)/dCdp='+str((dCdp-dCdpold)/dCdp))
        gradC.append(dCdp)
    gradC=np.asarray(gradC)
    nvec=gradC/np.linalg.norm(gradC) #normal vector
    eps=0.1
    print(gradC@gradC.T/eps**2)
    e_u=np.asarray([nvec[1],-nvec[0],0]) #orthgonal to normal and shift
    e_u=e_u/np.linalg.norm(e_u)    
    e_v=np.cross(nvec,e_u)
    #import ipdb
    #ipdb.set_trace()
    
    eps_u=1e-2
    eps_v=1e-2
    #eps_u=1e-4
    #eps_v=1e-4
    eps_vec=(eps_u,eps_v)
    num_steps=5
    uvec=eps_u*np.arange(-num_steps,num_steps+1)/num_steps
    vvec=eps_v*np.arange(-num_steps,num_steps+1)/num_steps
    grid_data_u,grid_data_v=np.meshgrid(uvec,vvec)
    dpara_data=np.asarray([pair[0]*e_u+pair[1]*e_v for pair in zip(grid_data_u.flatten(),grid_data_v.flatten())])
    L_data=np.reshape(np.asarray([-np.power(10,partialobjfunc(opt_paras+dpara)) for dpara in dpara_data] ),(len(uvec),len(vvec)))*Nclones
    #L_data=np.reshape(np.asarray([-np.power(10,partialobjfunc(np.power(10,np.log10(opt_paras)+dpara))) for dpara in dpara_data] ),(len(uvec),len(vvec)))*Nclones
    #import ipdb
    #ipdb.set_trace()
    print(L_data-np.max(L_data))
    #learn diags
    coeffs=np.zeros((2,3))
    cind=2
    for pit,paravec in enumerate([uvec,vvec]):
        coeffs[pit,:]=poly.polyfit(paravec-paravec[cind], -(L_data[pit,:]-L_data[pit,cind]), 2)
        fvals=(coeffs[pit,2]*(paravec-paravec[cind])**2 + coeffs[pit,1]*(paravec-paravec[cind]) + coeffs[pit,0])
        print(fvals)
    diag_entries=2*coeffs[:,2] #2 factor is correction from taylor (c.f. off-diag fit)
    
    #learn off diags
    para1vec=uvec
    para2vec=vvec
    data_df=pd.DataFrame()
    for pit1,para1 in enumerate(para1vec):
        for pit2,para2 in enumerate(para2vec):
            data_df=data_df.append(pd.Series({'x':np.array([para1-para1vec[cind],para2-para2vec[cind]]),'y':-(L_data[pit1,pit2]-L_data[cind,cind])}),ignore_index=True)
    partial_MSError_fcn=partial(MSError_fcn,data=data_df,diag_pair=diag_entries)
    initparas=1.
    outstruct_quad = minimize(partial_MSError_fcn, initparas, method='Nelder-Mead', tol=1e-10,options={'ftol':1e-10 ,'disp': False,'maxiter':200})

    M=np.diag(diag_entries)
    M[0,1]=outstruct_quad.x
    M[1,0]=outstruct_quad.x
    
    #diagonalize covariance
    Cov=np.linalg.inv(M)
    #e_val,e_vec=np.linalg.eig(Cov.T)
    cov = np.flipud(np.fliplr(Cov)) #fliplr gets coordinate order right
    vals, vecs = eigsorted(cov)
    print('eigenvalues:')
    print(vals)
    ell_1=np.sqrt(vals[0])*vecs[0]/np.linalg.norm(vecs[0])
    ell_2=np.sqrt(vals[1])*vecs[1]/np.linalg.norm(vecs[1])
    
    ell_1_p=ell_1[0]*e_u+ell_1[1]*e_v
    ell_2_p=ell_2[0]*e_u+ell_2[1]*e_v
    
    return ell_1_p,ell_2_p,cov,L_data

#for imposing constraint when running grid search
def get_shift(logPsvec,null_paras,sparse_rep,acq_model_type,shift,logfvec,logfvecwide,svec,f2s_step,logPn1_f,logrhofvec,tol=1e-3):
    
    indn1,indn2,sparse_rep_counts,_,unicounts,_,Nreads=sparse_rep.values()
    Nsamp=np.sum(sparse_rep_counts)
    dlogfby2=np.diff(logfvec)/2. #1/2 comes from trapezoid integration below

    alpha = null_paras[0] #power law exponent
    fmin=np.power(10,null_paras[-1])
    if acq_model_type<2:
        m_total=float(np.power(10, null_paras[3])) 
        r_c=Nreads/m_total
    if acq_model_type<3:
        beta_mv= null_paras[1]
        alpha_mv=null_paras[2]

    diffval=np.Inf
    addshift=0
    it=0
    #breakflag=False
    while diffval>tol:
        it+=1
        shift+=addshift
        print(str(it)+' '+str(shift)+' '+str(diffval))
        fvecwide_shift=np.exp(logfvecwide-shift) #implements shift in Pn2_fs
        svec_shift=svec-shift
        
        logPn2_f=get_logPn_f(unicounts,Nreads,np.log(fvecwide_shift),acq_model_type,null_paras)
        
        #logPn2_s=np.zeros((len(svec),len(logfvec),len(unicounts))) #svec is svec_shift
        #for s_it in range(len(svec)):
            #logPn2_s[s_it,:,:]=logPn2_f[f2s_step*s_it:f2s_step*s_it+len(logfvec),:]   #note here this is fvec long
        #iterative sum avoids having to compute P(n|f,s) above and commented out lines below
        log_Pn2_f=np.zeros((len(logfvec),len(unicounts)))
        for s_it in range(len(svec)):
            log_Pn2_f+=np.exp(logPn2_f[f2s_step*s_it:f2s_step*s_it+len(logfvec),:]+logPsvec[s_it,np.newaxis,np.newaxis])
        log_Pn2_f=np.log(log_Pn2_f)
        #log_Pn2_f=np.log(np.sum(np.exp(logPn2_s+logPsvec[:,np.newaxis,np.newaxis]),axis=0))
        integ=np.exp(log_Pn2_f[:,indn2]+logPn1_f[:,indn1]+logrhofvec[:,np.newaxis]+logfvec[:,np.newaxis])
        log_Pn1n2=np.log(np.sum(dlogfby2[:,np.newaxis]*(integ[1:,:] + integ[:-1,:]),axis=0))

        #log_Pn2_f=np.log(np.sum(np.exp(logPn2_s+logPsvec[:,np.newaxis,np.newaxis]),axis=0))
        integ=np.exp(np.log(integ)+logfvec[:,np.newaxis]) 
        #np.exp(log_Pn2_f[:,indn2]+logPn1_f[:,indn1]+logrhofvec[:,np.newaxis]+logfvec[:,np.newaxis]+logfvec[:,np.newaxis])
        tmp=deepcopy(log_Pn1n2)
        tmp[tmp==-np.Inf]=np.Inf #since subtracted in next line
        avgf_n1n2=    np.exp(np.log(np.sum(dlogfby2[:,np.newaxis]*(integ[1:,:] + integ[:-1,:]),axis=0))-tmp)
        log_avgf=np.log(np.dot(sparse_rep_counts,avgf_n1n2))

        log_expsavg_Pn2_f=np.zeros((len(logfvec),len(unicounts)))
        for s_it in range(len(svec)):
            log_expsavg_Pn2_f+=np.exp(svec_shift[s_it,np.newaxis,np.newaxis]+logPn2_f[f2s_step*s_it:f2s_step*s_it+len(logfvec),:]+logPsvec[s_it,np.newaxis,np.newaxis]) #cuts down on memory constraints
        log_expsavg_Pn2_f=np.log(log_expsavg_Pn2_f)
        #log_expsavg_Pn2_f=np.log(np.sum(np.exp(svec[:,np.newaxis,np.newaxis]+logPn2_s+logPsvec[:,np.newaxis,np.newaxis]),axis=0))
        integ=np.exp(log_expsavg_Pn2_f[:,indn2]+logPn1_f[:,indn1]+logrhofvec[:,np.newaxis]+2*logfvec[:,np.newaxis])
        avgfexps_n1n2=np.exp(np.log(np.sum(dlogfby2[:,np.newaxis]*(integ[1:,:] + integ[:-1,:]),axis=0))-tmp)
        log_avgfexps=np.log(np.dot(sparse_rep_counts,avgfexps_n1n2))

        logPn20_s=np.zeros((len(svec),len(logfvec))) #svec is svec_shift
        for s_it in range(len(svec)):
            logPn20_s[s_it,:]=logPn2_f[f2s_step*s_it:f2s_step*s_it+len(logfvec),0]   #note here this is fvec long on shifted s
        log_Pnn0_fs=logPn1_f[np.newaxis,:,0]+logPn20_s
        log_Pfsnn0=log_Pnn0_fs+logrhofvec[np.newaxis,:]+logPsvec[:,np.newaxis]
        log_Pfsnng0=np.log(1-np.exp(log_Pnn0_fs))+logrhofvec[np.newaxis,:]+logPsvec[:,np.newaxis]
        log_Pfnn0=np.log(np.sum(np.exp(log_Pfsnn0),axis=0))
        integ=np.exp(log_Pfnn0+logfvec)
        logPnn0=np.log(np.dot(dlogfby2,integ[1:]+integ[:-1]))

        log_Pnng0=np.log(1-np.exp(logPnn0))
        log_Pfs_nng0=log_Pfsnng0-log_Pnng0

        #decomposed f averages
        integ = np.exp(logPn1_f[:,0]+logrhofvec+2*logfvec+np.log(np.sum(np.exp(logPn20_s+logPsvec[:,np.newaxis]),axis=0)))
        log_avgf_n0n0 = np.log(np.dot(dlogfby2,integ[1:]+integ[:-1]))

        #decomposed fexps averages
        integ = np.exp(logPn1_f[:,0]+logrhofvec+2*logfvec+np.log(np.sum(np.exp(svec_shift[:,np.newaxis]+logPn20_s+logPsvec[:,np.newaxis]),axis=0)))  #----------svec
        log_avgfexps_n0n0 = np.log(np.dot(dlogfby2,integ[1:]+integ[:-1]))

        logNclones=np.log(Nsamp)-log_Pnng0
        Z     = np.exp(logNclones + logPnn0 + log_avgf_n0n0    ) + np.exp(log_avgf)    
        Zdash = np.exp(logNclones + logPnn0 + log_avgfexps_n0n0) + np.exp(log_avgfexps)
        
        if Z>1.5 or Zdash>1.5:
            print('Prior too big!'+str(Z)+' '+str(Zdash))
            it=-1
            break

        addshiftold=deepcopy(addshift)
        addshift=np.log(Zdash)-np.log(Z)
        diffval=np.fabs(addshift-addshiftold)
    return shift,Z,Zdash,it










    

#def constr_fn_diffexpr(paras,null_paras,svec,smax,s_step,indn1_d,indn2_d,fvec,fvecwide,rhofvec,\
                            #unicountvals_1_d,unicountvals_2_d,countpaircounts_d,\
                            #NreadsI, NreadsII, len(logfvec),f2s_step,\
                            #m_low,m_high,mvec,Nsamp,logPn1_f,acq_model_type):
    ###Ps = get_Ps_pm(np.power(10,paras[0]),np.power(10,paras[1]),paras[2],paras[3],smax,s_step)
    ##Ps = get_Ps_pm(np.power(10,paras[0]),0,0,paras[1],smax,s_step)

    ##logPsvec=np.log(Ps) 
    ##shift=paras[-1]

    ##alpha_rho = null_paras[0]
    ##if acq_model_type<2: #acq_model_type: 0=NB->Pois, 1=Pois->NB, 2=NB, 3=Pois 
        ##m_total=float(np.power(10, null_paras[3]))
        ##r_c1=NreadsI/m_total 
        ##r_c2=NreadsII/m_total      
        ##r_cvec=(r_c1,r_c2)
        ##fmin=np.power(10,null_paras[4])
    ##else:
        ##fmin=np.power(10,null_paras[3])

    ##if acq_model_type<3:
        ##beta_mv= null_paras[1]
        ##alpha_mv=null_paras[2]

    ##logfvec=np.log(fvec)
    ##dlogf=np.diff(logfvec)/2.
    ##logrhofvec=np.log(rhofvec)
    ##fvecwide_shift=np.exp(np.log(fvecwide)-shift) #implements shift in Pn2_fs
    ##svec_shift=svec-shift
    ##unicounts=unicountvals_2_d
    ##Pn2_f=np.zeros((len(fvecwide_shift),len(unicounts)))
    ##if acq_model_type==0:
        ##mean_m=m_total*fvecwide_shift
        ##var_m=mean_m+beta_mv*np.power(mean_m,alpha_mv)
        ##Poisvec=PoisPar(mvec*r_cvec[1],unicounts)
        ##for f_it in range(len(fvecwide_shift)):
            ##NBvec=NegBinPar(mean_m[f_it],var_m[f_it],mvec)
            ##for n_it,n in enumerate(unicounts):
                ##Pn2_f[f_it,n_it]=np.dot(NBvec[m_low[n_it]:m_high[n_it]+1],Poisvec[m_low[n_it]:m_high[n_it]+1,n_it]) 
    ##elif acq_model_type==1:
        ##Poisvec=PoisPar(m_total*fvecwide_shift,mvec)
        ##mean_n=r_cvec[1]*mvec
        ##NBmtr=NegBinParMtr(mean_n,mean_n+beta_mv*np.power(mean_m,alpha_mv),unicounts)
        ##for f_it in range(len(fvecwide_shift)):
            ##Poisvectmp=Poisvec[f_it,:]
            ##for n_it,n in enumerate(unicounts):
                ##Pn2_f[f_it,n_it]=np.dot(Poisvectmp[m_low[n_it]:m_high[n_it]+1],NBmtr[m_low[n_it]:m_high[n_it]+1,n_it]) 
    ##elif acq_model_type==2:
        ##mean_n=Nreadsvec[1]*fvecwide_shift
        ##var_n=mean_n+beta_mv*np.power(mean_n,alpha_mv)
        ##Pn2_f=NegBinParMtr(mean_n,var_n,unicounts)
    ##else:# acq_model_type==3:
        ##mean_n=Nreadsvec[1]*fvecwide_shift
        ##Pn2_f=PoisPar(mean_n,unicounts)
    ##logPn2_f=Pn2_f
    ##logPn2_f=np.log(logPn2_f)
    ##logPn2_s=np.zeros((len(svec),len(logfvec),len(unicounts))) #svec is svec_shift
    ##for s_it in range(len(svec)):
        ##logPn2_s[s_it,:,:]=logPn2_f[f2s_step*s_it:f2s_step*s_it+len(logfvec),:]   #note here this is fvec long


    ##log_avgf
    ##Pn2_f_ints=np.zeros((len(fvec),len(unicountvals_2_d)))
    ##for s_it in range(len(svec)):
        ##Pn2_f_ints+=np.exp(logPn2_f[f2s_step*s_it:f2s_step*s_it+len(logfvec),:]+logPsvec[s_it,np.newaxis,np.newaxis])
    ##logPn2_f_ints=Pn2_f_ints
    ##logPn2_f_ints=np.log(logPn2_f_ints)
    
    ##log_Pn2_f=np.log(np.sum(np.exp(logPn2_s+logPsvec[:,np.newaxis,np.newaxis]),axis=0))
    #integ=np.exp(logPn2_f_ints[:,indn2_d]+logPn1_f[:,indn1_d]+logrhofvec[:,np.newaxis]+logfvec[:,np.newaxis])
    #log_Pn1n2=np.log(np.sum(dlogf[:,np.newaxis]*(integ[1:,:] + integ[:-1,:]),axis=0))
    ##tmp=np.exp(log_Pn1n2-np.log(1-np.exp(logPnn0))) #renormalize
    ##log_Pn2_f=np.log(np.sum(np.exp(logPn2_s+logPsvec[:,np.newaxis,np.newaxis]),axis=0))
    #integ=np.exp(np.log(integ)+logfvec[:,np.newaxis]) 
    ##np.exp(log_Pn2_f[:,indn2]+logPn1_f[:,indn1]+logrhofvec[:,np.newaxis]+logfvec[:,np.newaxis]+logfvec[:,np.newaxis])
    #tmp=deepcopy(log_Pn1n2)
    #tmp[tmp==-np.Inf]=np.Inf #since subtracted in next line
    #avgf_n1n2=    np.exp(np.log(np.sum(dlogf[:,np.newaxis]*(integ[1:,:] + integ[:-1,:]),axis=0))-tmp)
    #log_avgf=np.log(np.dot(countpaircounts_d,avgf_n1n2))


    ##log_avgfexps
    #log_expsavg_Pn2_f=np.zeros((len(fvec),len(unicountvals_2_d)))
    #for s_it in range(len(svec)):
        #log_expsavg_Pn2_f+=np.exp(svec_shift[s_it,np.newaxis,np.newaxis]+logPn2_f[f2s_step*s_it:f2s_step*s_it+len(logfvec),:]+logPsvec[s_it,np.newaxis,np.newaxis]) #cuts down on memory constraints
    #log_expsavg_Pn2_f=np.log(log_expsavg_Pn2_f)
    ##log_expsavg_Pn2_f=np.log(np.sum(np.exp(svec[:,np.newaxis,np.newaxis]+logPn2_s+logPsvec[:,np.newaxis,np.newaxis]),axis=0))
    #integ=np.exp(log_expsavg_Pn2_f[:,indn2_d]+logPn1_f[:,indn1_d]+logrhofvec[:,np.newaxis]+2*logfvec[:,np.newaxis])
    #avgfexps_n1n2=np.exp(np.log(np.sum(dlogf[:,np.newaxis]*(integ[1:,:] + integ[:-1,:]),axis=0))-tmp)
    #log_avgfexps=np.log(np.dot(countpaircounts_d,avgfexps_n1n2))

    ##00
    #logPn20_s=np.zeros((len(svec),len(fvec))) #svec is svec_shift
    #for s_it in range(len(svec)):
        #logPn20_s[s_it,:]=logPn2_f[f2s_step*s_it:f2s_step*s_it+len(fvec),0]   #note here this is fvec long on shifted s
    #log_Pnn0_fs=logPn1_f[np.newaxis,:,0]+logPn20_s
    #log_Pfsnn0=log_Pnn0_fs+logrhofvec[np.newaxis,:]+logPsvec[:,np.newaxis]
    #log_Pfnn0=np.log(np.sum(np.exp(log_Pfsnn0),axis=0))
    #integ=np.exp(log_Pfnn0+logfvec)
    #logPnn0=np.log(np.sum(dlogf*(integ[1:]+integ[:-1])))
    #log_Pnng0=np.log(1-np.exp(logPnn0))
    #logNclones=np.log(Nsamp)-log_Pnng0
    ##decomposed f averages
    #integ = np.exp(logPn1_f[:,0]+logrhofvec+2*logfvec+np.log(np.sum(np.exp(logPn20_s+logPsvec[:,np.newaxis]),axis=0)))
    #log_avgf_n0n0 = np.log(np.dot(dlogf,integ[1:]+integ[:-1]))
    ##decomposed fexps averages
    #integ = np.exp(logPn1_f[:,0]+logrhofvec+2*logfvec+np.log(np.sum(np.exp(svec_shift[:,np.newaxis]+logPn20_s+logPsvec[:,np.newaxis]),axis=0)))  #----------svec
    #log_avgfexps_n0n0 = np.log(np.dot(dlogf,integ[1:]+integ[:-1]))
    
    ##final expressions
    #Z     = np.exp(logNclones + logPnn0 + log_avgf_n0n0    ) + np.exp(log_avgf)    
    #Zdash = np.exp(logNclones + logPnn0 + log_avgfexps_n0n0) + np.exp(log_avgfexps)

    #return np.log(Zdash)-np.log(Z)

#def get_likelihood(paras,null_paras,svec,smax,s_step,indn1_d,indn2_d,fvec,fvecwide,rhofvec,\
                                 #unicountvals_1_d,unicountvals_2_d,countpaircounts_d,\
                                 #NreadsI, NreadsII, len(logfvec),f2s_step,\
                                 #m_low,m_high,mvec,Nsamp,logPn1_f,acq_model_type):
    #logfvec=np.log(fvec)
    #dlogf=np.diff(logfvec)/2.
    #logrhofvec=np.log(rhofvec)
    #alpha_rho = null_paras[0]
    #if acq_model_type<2: #acq_model_type: 0=NB->Pois, 1=Pois->NB, 2=NB, 3=Pois 
        #m_total=float(np.power(10, null_paras[3]))
        #r_c1=NreadsI/m_total 
        #r_c2=NreadsII/m_total     
        #r_cvec=(r_c1,r_c2)
        #fmin=np.power(10,null_paras[4])
    #else:
        #fmin=np.power(10,null_paras[3])

    #if acq_model_type<3:
        #beta_mv= null_paras[1]
        #alpha_mv=null_paras[2]
    
    #Ps = get_Ps_pm(np.power(10,paras[0]),0,0,paras[1],smax,s_step)

    #logPsvec=np.log(Ps)
    #shift=paras[-1]
    

    #fvecwide_shift=np.exp(np.log(fvecwide)-shift) #implements shift in Pn2_fs
    #svec_shift=svec-shift
    #unicounts=unicountvals_2_d
    #Pn2_f=np.zeros((len(fvecwide_shift),len(unicounts)))
    #if acq_model_type==0:
        #mean_m=m_total*fvecwide_shift
        #var_m=mean_m+beta_mv*np.power(mean_m,alpha_mv)
        #Poisvec=PoisPar(mvec*r_cvec[1],unicounts)
        #for f_it in range(len(fvecwide_shift)):
            #NBvec=NegBinPar(mean_m[f_it],var_m[f_it],mvec)
            #for n_it,n in enumerate(unicounts):
                #Pn2_f[f_it,n_it]=np.dot(NBvec[m_low[n_it]:m_high[n_it]+1],Poisvec[m_low[n_it]:m_high[n_it]+1,n_it]) 
    #elif acq_model_type==1:
        #Poisvec=PoisPar(m_total*fvecwide_shift,mvec)
        #mean_n=r_cvec[1]*mvec
        #NBmtr=NegBinParMtr(mean_n,mean_n+beta_mv*np.power(mean_m,alpha_mv),unicounts)
        #for f_it in range(len(fvecwide_shift)):
            #Poisvectmp=Poisvec[f_it,:]
            #for n_it,n in enumerate(unicounts):
                #Pn2_f[f_it,n_it]=np.dot(Poisvectmp[m_low[n_it]:m_high[n_it]+1],NBmtr[m_low[n_it]:m_high[n_it]+1,n_it]) 
    #elif acq_model_type==2:
        #mean_n=Nreadsvec[1]*fvecwide_shift
        #var_n=mean_n+beta_mv*np.power(mean_n,alpha_mv)
        #Pn2_f=NegBinParMtr(mean_n,var_n,unicounts)
    #else:# acq_model_type==3:
        #mean_n=Nreadsvec[1]*fvecwide_shift
        #Pn2_f=PoisPar(mean_n,unicounts)
    #logPn2_f=Pn2_f
    #logPn2_f=np.log(logPn2_f)
    ##logPn2_s=np.zeros((len(svec),len(logfvec),len(unicounts))) #svec is svec_shift
    ##for s_it in range(len(svec)):
        ##logPn2_s[s_it,:,:]=logPn2_f[f2s_step*s_it:f2s_step*s_it+len(logfvec),:]   #note here this is fvec long

    #log_Pn2_f=np.zeros((len(fvec),len(unicountvals_2_d)))
    #for s_it in range(len(svec)):
        #log_Pn2_f+=np.exp(logPn2_f[f2s_step*s_it:f2s_step*s_it+len(logfvec),:]+logPsvec[s_it,np.newaxis,np.newaxis])
    #log_Pn2_f=np.log(log_Pn2_f)
    ##log_Pn2_f=np.log(np.sum(np.exp(logPn2_s+logPsvec[:,np.newaxis,np.newaxis]),axis=0))
    #integ=np.exp(log_Pn2_f[:,indn2_d]+logPn1_f[:,indn1_d]+logrhofvec[:,np.newaxis]+logfvec[:,np.newaxis])
    #log_Pn1n2=np.log(np.sum(dlogf[:,np.newaxis]*(integ[1:,:] + integ[:-1,:]),axis=0))
    
    #integ=np.exp(log_Pn2_f[:,0]+logPn1_f[:,0]+logrhofvec+logfvec)
    #logPnn0=np.log(np.sum(dlogf*(integ[1:] + integ[:-1]),axis=0))
    
    
    #tmp=np.exp(log_Pn1n2-np.log(1-np.exp(logPnn0))) #renormalize
    #return np.dot(countpaircounts_d/float(Nsamp),np.where(tmp>0,np.log(tmp),0))