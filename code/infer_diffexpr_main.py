#local external packages
import numpy as np
import time
from copy import deepcopy

#add package to path
import sys,os
root_path = os.path.abspath(os.path.join('..'))
if root_path not in sys.path:
    sys.path.append(root_path)
    
#load local functions
from lib.proc import get_sparserep,import_data
from lib.learning import constr_fn,callback,learn_null_model
from lib.utils.readouts import setup_outpath_and_logs

#inputs to mkl library used by numpy to parallelize np.dot 
import ctypes
mkl_rt = ctypes.CDLL('libmkl_rt.so')
num_threads=4 #number of cores available on local machine
mkl_set_num_threads = mkl_rt.MKL_Set_Num_Threads(num_threads)
print(str(num_threads)+' cores available')

def main(null_pair_1,null_pair_2,test_pair_1,test_pair_2,run_index,input_data_path,output_data_path):
  
    constr_type=1
    parvernull = 'v1_null_ct_'+str(constr_type) #prefix label for null model data path
    parvertest = 'v1_null' #prefix label for diff expr model data path

######################################Preprocessing###########################################3

    # filter out counts (set to 0 and np.inf, respectively, for no filtering)
    mincount = 0
    maxcount = np.Inf #requires about 20GB RAM. Set to 200 to run using less RAM.
    
    # script paras
    smax = 25.0     #maximum absolute logfold change value
    s_step =0.1     #logfold change step size

#######################################0
    
    # Start Computations
    starttime = time.time()
    
    null_pair_1=null_pair_1.split('.')[0][:-1]
    null_pair_2=null_pair_2.split('.')[0][:-1]
    test_pair_1=test_pair_1.split('.')[0][:-1]
    test_pair_2=test_pair_2.split('.')[0][:-1]
    logfilename='_'.join((null_pair_1,null_pair_2,test_pair_1,test_pair_2))
    
    acq_model_type=0 #which P(n|f) model to use (0:NB->Pois,1:Pois->NB,2:NBonly,3:Pois only)

    #loop: it=0 :learn null paras on specified null pair, then it=1: load test pair data
    it_label=('null_pair','diffexpr_pair')
    for it,dataset_pair in enumerate(((null_pair_1,null_pair_2),(test_pair_1,test_pair_2))):
        
        donor1,day1,rep1=dataset_pair[0].split('_')
        donor2,day2,rep2=dataset_pair[1].split('_')
            
        assert donor1==donor2, 'trying to compare data from different donors!'
        donorstr=donor1
        
        #input path    
        datasetstr=dataset_pair[0]+'_'+dataset_pair[1] 
        
        loadnull=False #set to True to skip 1st iteration that learns null model, once learned null model parameters, optparas.py, already exist.
        if (not loadnull and it==0) or it==1:
          
            if it==0:
                #set output path
                datasetstr_null=deepcopy(datasetstr)
                parver=parvernull+'_acq_model_type'+str(acq_model_type)
            else:
                parver=parvertest
                parver=datasetstr_null+"_"+parver

            runstr = it_label[it]+'_'+parver +'_min' + str(mincount) + '_max' + str(maxcount)
            outpath = output_data_path + dataset_pair[0] + '_' + dataset_pair[1] + '/' + runstr + '/'
            if not os.path.exists(outpath):
                time.sleep((run_index+1)/10.)#avoid race condition
                os.makedirs(outpath)
                 
            #write shelloutput to file
            prtfn= setup_outpath_and_logs(outpath,logfilename)
            if it==0:
                prtfn("importing and running null pair: "+datasetstr+'\n')
            else:
                if loadnull:
                    prtfn('loading learned null paras for '+str(null_pair_1) + ' ' + str(null_pair_2)+' : '+str(paras))
                prtfn("importing and running test pair: "+datasetstr+'\n')

            #read in data with heterogeneous labelling 
            if day1=='730': #2yr datasets have different column labels
                colnames1 = [u'cloneFraction',u'cloneCount',u'nSeqCDR3',u'aaSeqCDR3'] #dataset specific labels
            else:
                colnames1 = [u'Clone fraction',u'Clone count',u'N. Seq. CDR3',u'AA. Seq. CDR3'] 
            if day2=='730': #2yr datasets have different column labels
                colnames2 = [u'cloneFraction',u'cloneCount',u'nSeqCDR3',u'aaSeqCDR3'] #dataset specific labels
            else:
                colnames2 = [u'Clone fraction',u'Clone count',u'N. Seq. CDR3',u'AA. Seq. CDR3'] 
                
            # import and structure data into a dataframe:
            Nclones_samp,subset=import_data(input_data_path,dataset_pair[0]+'_.txt',dataset_pair[1]+'_.txt',mincount,maxcount,colnames1,colnames2)
            
            #transform to sparse representation adn store    
            sparse_rep=get_sparserep(subset.loc[:,['Clone_count_1','Clone_count_2']])       

            if it==0:
                #initial values for null model learning (copied near optimal ones here to speed up):
                if os.path.exists(outpath+'optparas.npy'):
                    paras=np.load(outpath+'optparas.npy')
                else:
                    prtfn('learn null:')
                    donorstrvec=['P1','P2','Q1','Q2','S1','S2']
                    #donorstrvec=('Yzh','KB',  'Luci', 'Kar', 'Azh','GS')  #delete this line of actual donor names before publishing
                    if acq_model_type==0:
                        #defaultnullparasvec=np.asarray( [
                        #[-2.21501853,  2.40694626,  1.16697953,  6.75321541, -9.41467361],
                        #[-2.20317149,  2.19642467,  1.10515637,  6.63847919, -9.57396901],
                        #[ -2.0016999,    1.69206104,   1.14715666,   6.82722752, -11.67739647],
                        #[ -2.08578182,   1.81610271,   1.10418693,   6.74893278, -10.1010326 ],
                        #[-2.15,        1.4,         1.15,        6.6685081,  -9.62994141],
                        #[-2.19686053,  2.54462487,  1.10896311,  6.63110241, -9.31923867]
                        #])
                        
                        #C=1
                        defaultnullparasvec=np.asarray( [
                        [-2.19865022,  2.40811359,  1.1885886 ,  6.73120963, -9.43713528],
                        [-2.18684369,  2.19921088,  1.12710212,  6.61652184, -9.59133643],
                        [ -1.99747218,   1.69221099,   1.15192853,   6.82786948,-11.6720086 ],
                        [ -2.08578182,   1.81610271,   1.10418693,   6.74893278, -10.1010326 ],
                        [-2.15,        1.4,         1.15,        6.6685081,  -9.62994141],
                        [-2.11369677,   2.47065402,   1.08416164,   6.62501443, -10.04874892]
                        ])
                    elif acq_model_type>0:
                        prtfn('not computed yet!')
                        
                    dind=[i for i, s in enumerate(donorstrvec) if donorstr in s][0]
                    init_paras = defaultnullparasvec[dind,:]
                    prtfn('init paras:'+str(init_paras))
                    st=time.time()
                    outstruct,constr_value=learn_null_model(sparse_rep,acq_model_type,init_paras,constr_type=constr_type,prtfn=prtfn)
                    prtfn('constr value:')
                    prtfn(constr_value)
                    prtfn('learning took '+str(time.time()-st))                    
                    if not outstruct.success:
                        prtfn('null learning failed!')
                    
                    optparas=outstruct.x
                    np.save(outpath + 'optparas', optparas)
                    np.save(outpath + 'success', outstruct.success)
                    np.save(outpath + 'outstruct', outstruct)

                    paras=optparas  #null paras to use from here on
                    
                prtfn("elapsed " + str(np.round(time.time() - st))+'\n')
                
        else:          
            datasetstr_null=datasetstr
            runstr = it_label[it]+'_'+parvernull+'_acq_model_type'+str(acq_model_type) +'_min' + str(mincount) + '_max' + str(maxcount)
            outpath = output_data_path + dataset_pair[0] + '_' + dataset_pair[1] + '/' + runstr + '/'
            paras=  np.load(outpath+'optparas.npy')

    ###############################diffexpr learning
    diffexpr=False
    if diffexpr:
        
        #get Pn1n2_s
        logrhofvec,logfvec = get_rhof(paras[0],np.power(10,paras[-1]))
        #biuld discrete domain of s, centered on s=0
        s_step_old=s_step
        logf_step=logfvec[1] - logfvec[0] #use natural log here since f2 increments in increments in exp().  
        f2s_step=int(round(s_step/logf_step)) #rounded number of f-steps in one s-step
        s_step=float(f2s_step)*logf_step
        smax=s_step*(smax/s_step_old)
        svec=s_step*np.arange(0,int(round(smax/s_step)+1))   
        svec=np.append(-svec[1:][::-1],svec)
 
        #compute conditional P(n1,n2|s) and P(n1=0,n2=0|s)
        prtfn('calc Pn1n2_s: ')
        st = time.time()
        if os.path.exists(outpath+'Pn1n2_s_d.npy'):
            Pn1n2_s=np.load(outpath+'Pn1n2_s_d.npy')
            Pn0n0_s=np.load(outpath+'Pn0n0.npy')
            logPn1_f=np.log(np.load(outpath + 'Pn1_f.npy'))
        else:
            Pn1n2_s, unicountvals_1_d, unicountvals_2_d, Pn1_f, fvec, Pn2_s, Pn0n0_s,svec = get_Pn1n2_s(paras, svec,sparse_rep,s_step=s_step)
            np.save(outpath + 'Pn1n2_s_d', Pn1n2_s)
            np.save(outpath + 'Pn0n0',Pn0n0_s)
            np.save(outpath + 'Pn1_f',Pn1_f)
            logPn1_f=np.log(Pn1_f)
            prtfn("calc Pn1n2_s elapsed " + str(np.round(time.time() - st))+'\n')
        
        #flags for 3 remaining code blocks:
        learn_surface=False
        polish_estimate=True
        output_table=False
            
        if learn_surface:
            prtfn('calc surface: \n')
            st = time.time()
            
            #define grid search parameters  
            npoints=20
            nsbarpoints=npoints
            sbarvec=np.linspace(0.01,5,nsbarpoints)
            nalppoints=21
            alpvec=np.logspace(-3,np.log10(0.99),nalppoints)

            LSurface =np.zeros((len(sbarvec),len(alpvec)))
            for sit,sbar in enumerate(sbarvec):
                for ait,alp in enumerate(alpvec):
                    Ps=get_Ps(alp,sbar,smax,s_step)
                    Pn0n0=np.dot(Pn0n0_s,Ps)
                    Pn1n2_ps=np.sum(Pn1n2_s*Ps[:,np.newaxis,np.newaxis],0)
                    Pn1n2_ps/=1-Pn0n0  #renormalize
                    LSurface[sit,ait]=np.dot(countpaircounts_d/float(Nsamp),np.where(Pn1n2_ps[indn1_d,indn2_d]>0,np.log(Pn1n2_ps[indn1_d,indn2_d]),0))

            maxinds=np.unravel_index(np.argmax(LSurface),np.shape(LSurface))
            optsbar=sbarvec[maxinds[0]]
            optalp=alpvec[maxinds[1]]
            optPs=get_Ps(optalp,optsbar,smax,s_step)

            np.save(outpath + 'optsbar', optsbar)
            np.save(outpath + 'optalp', optalp)
            np.save(outpath + 'LSurface', LSurface)
            np.save(outpath + 'sbarvec', sbarvec)
            np.save(outpath + 'alpvec', alpvec)
            np.save(outpath + 'optPs', optPs)
            prtfn("optalp="+str(optalp)+" ("+str(alpvec[0])+","+str(alpvec[-1])+"),optsbar="+str(optsbar)+", ("+str(sbarvec[0])+","+str(sbarvec[-1])+") \n")
            prtfn("surface learning elapsed " + str(np.round(time.time() - st))+'\n')
            
        if polish_estimate:
            
            optsbar=np.load(outpath + 'optsbar.npy')
            optalp=np.load(outpath + 'optalp.npy')
            prtfn('polish parameter estimate from '+str(optalp)+' '+str(optsbar))
            
            init_shift=0
            initparas=(optalp,optsbar,init_shift)   #(alpha,sbar,shift)
            NreadsI=NreadsI_d 
            NreadsII=NreadsII_d
            
            smaxind=(len(svec)-1)/2
            logfmin=logfvec[0 ]-f2s_step*smaxind*logf_step
            logfmax=logfvec[-1]+f2s_step*smaxind*logf_step
            logfvecwide=np.linspace(logfmin,logfmax,len(logfvec)+2*smaxind*f2s_step)

            #compute range of m values (number of cells) over which to compute given the n values (reads) in the data  
            m_total=float(np.power(10, paras[3]))
            r_c1=NreadsI/m_total 
            r_c2=NreadsII/m_total      
            r_cvec=(r_c1,r_c2)
            nsigma=5.
            nmin=300.
            #for each n, get actual range of m to compute around n-dependent mean m
            m_low =np.zeros((len(unicountvals_2_d),),dtype=int)
            m_high=np.zeros((len(unicountvals_2_d),),dtype=int)
            for nit,n in enumerate(unicountvals_2_d):
                mean_m=n/r_cvec[it]
                dev=nsigma*np.sqrt(mean_m)
                m_low[nit] =int(mean_m-  dev) if (mean_m>dev**2) else 0                         
                m_high[nit]=int(mean_m+5*dev) if (      n>nmin) else int(10*nmin/r_cvec[it])
            m_cellmax=np.max(m_high)
            #across n, collect all in-range m
            mvec_bool=np.zeros((m_cellmax+1,),dtype=bool) #cheap bool
            nvec=range(len(unicountvals_2_d))
            for nit in nvec:
                mvec_bool[m_low[nit]:m_high[nit]+1]=True  #mask vector
            mvec=np.arange(m_cellmax+1)[mvec_bool]                
            #transform to in-range index
            for nit in nvec:
                m_low[nit]=np.where(m_low[nit]==mvec)[0][0]
                m_high[nit]=np.where(m_high[nit]==mvec)[0][0]                

            partialobjfunc=partial(get_likelihood,null_paras=paras,svec=svec,smax=smax,s_step=s_step,indn1_d=indn1_d ,indn2_d=indn2_d,fvec=np.exp(logfvec),fvecwide=np.exp(logfvecwide),rhofvec=np.exp(logrhofvec),\
                                            unicountvals_1_d=unicountvals_1_d, unicountvals_2_d=unicountvals_2_d, countpaircounts_d=countpaircounts_d,\
                                            NreadsI=NreadsI_d, NreadsII=NreadsII_d, nfbins=nfbins,f2s_step=f2s_step,\
                                            m_low=m_low,m_high=m_high, mvec=mvec,Nsamp=Nsamp,logPn1_f=logPn1_f,acq_model_type=acq_model_type)
            
            condict={'type':'eq','fun':constr_fn_diffexpr,'args': (paras,svec,smax,s_step,indn1_d,indn2_d,np.exp(logfvec),np.exp(logfvecwide),np.exp(logrhofvec),\
                                                                unicountvals_1_d,unicountvals_2_d,countpaircounts_d,\
                                                                NreadsI, NreadsII,nfbins, f2s_step,\
                                                                m_low,m_high,mvec,Nsamp,logPn1_f,acq_model_type)\
                }
            callbackFdiffexpr_part=partial(callbackFdiffexpr,prtfn=prtfn)
            outstruct = minimize(partialobjfunc, initparas, method='SLSQP', callback=callbackFdiffexpr_part, constraints=condict,tol=1e-6,options={'ftol':1e-8 ,'disp': True,'maxiter':300})
            np.save(outpath + 'outstruct_diffexpr', outstruct)
            
        if output_table:
            st=time.time()
            prtfn('write table: ')
            optsbar=np.load(outpath + 'sbaropt.npy')
            optalp=np.load(outpath + 'alpopt.npy')
            svec=np.load(outpath + 'svec.npy')
            Pn1n2_s=np.load(outpath + 'Pn1n2_s_d.npy')
            Psopt=np.load(outpath + 'Psopt.npy')
            
            pval_expanded=True #which end of the rank list to pull out. else: most contracted
            pval_threshold=0.05  #output all clones with pval below this threshold
            smed_threshold=3.46 #ln(2^5)
            save_table(outpath+datasetstr+"table",pval_expanded,smed_threshold,pval_threshold,svec, Psopt, Pn1n2_s, Pn0n0_s,subset,unicountvals_1_d,unicountvals_2_d,indn1_d,indn2_d)
            prtfn(" elapsed " + str(np.round(time.time() - st))+'\n')

    # end computations
    prtfn('program elapsed:' + str( time.time() - starttime))
    
if __name__ == "__main__": 
    try:#optional args
        inputnull_1=sys.argv[1]
        inputnull_2=sys.argv[2]
        inputtest_1=sys.argv[3]
        inputtest_2=sys.argv[4]
    except IndexError:
        inputnull_1='S2_0_F1_.txt'
        inputnull_2='S2_0_F2_.txt' 
        inputtest_1='S2_0_F1_.txt'
        inputtest_2='S2_15_F2_.txt'
    
    #default command: "python infer_diffexpr_main nulldataset1.txt nulldataset2.txt diffexprdataset1.txt diffexprdataset2.txt"
    
    try:#optional args
        run_index=int(sys.argv[5])      #run specific index 
        output_data_path=sys.argv[6]    #custom path
        input_data_path=sys.argv[7]     #custom path
    except IndexError:
        run_index=0
        output_data_path='../../data/Yellow_fever/prepostvaccine/'
        input_data_path='../../output/'
                
    main(inputnull_1,inputnull_2,inputtest_1,inputtest_2,run_index,output_data_path,input_data_path)