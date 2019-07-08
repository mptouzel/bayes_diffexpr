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
from lib.learning import callback,learn_null_model,get_shift,learn_diffexpr_model
from lib.utils.readouts import setup_outpath_and_logs
from lib.model import get_svec,get_logPn_f,get_logPs_pm,get_rhof

#inputs to mkl library used by numpy to parallelize np.dot 
import ctypes
mkl_rt = ctypes.CDLL('libmkl_rt.so')
num_threads=2 #number of cores available on local machine
mkl_set_num_threads = mkl_rt.MKL_Set_Num_Threads(num_threads)
print(str(num_threads)+' cores available')

#with kernprof.py in directory, place @profile above any functions to be profiled , then run:
# kernprof -l tempytron_main.py
#then generate readable output by running:
# python -m line_profiler tempytron_main.py.lprof > profiling_stats.txt 

def main(null_pair_1,null_pair_2,test_pair_1,test_pair_2,run_index,input_data_path,output_data_path):
  
    constr_type=1
    acq_model_type=2 #which P(n|f) model to use (0:NB->Pois,1:Pois->NB,2:NBonly,3:Pois only)
    
    #rhs only, sbar scan for each alpha
    npoints=2
            
    Ps_type='rhs_only'
    #alpvec=np.logspace(-6.,0., 2*npoints)
    #sbarvec_p=np.logspace(-1,1.5,npoints)
    alpvec=np.logspace(-3.,0., 2*npoints)
    sbarvec_p=np.logspace(-0.5,0.5,npoints)

    #Ps_type='sym_exp'
    #alpvec=np.logspace(-8.,0., 2*npoints)
    #sbarvec_p=np.logspace(-1.,1.5,npoints)
    
    #Ps_type='cent_gauss'
    #alpvec=np.logspace(-6.,0., npoints)
    #sbarvec_p=np.logspace(-1,1,npoints)/5
    
    #Ps_type='offcent_gauss'
    #alpvec=np.logspace(-6.,0., npoints)
    #sbarvec_p=np.logspace(-2,1,npoints)
    #sbarvec_m=np.linspace(0.1,15,10)
        
    parvernull = 'v1_ct_'+str(constr_type)+'_mt_'+str(acq_model_type) #prefix label for null model data path
    parvertest = 'v2_ct_'+str(constr_type)+'_mt_'+str(acq_model_type)+'_st_'+Ps_type #prefix label for diff expr model data path

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
    
    if acq_model_type<2:
        nparas=5
    elif acq_model_type==2:
        nparas=4
    else:
        nparas=3
        
    null_pair_1=null_pair_1.split('.')[0][:-1]
    null_pair_2=null_pair_2.split('.')[0][:-1]
    test_pair_1=test_pair_1.split('.')[0][:-1]
    test_pair_2=test_pair_2.split('.')[0][:-1]
    logfilename='_'.join((null_pair_1,null_pair_2,test_pair_1,test_pair_2))
    
    #loop: it=0 :learn null paras on specified null pair, then it=1: load test pair data
    it_label=('null_pair','diffexpr_pair')
    for it,dataset_pair in enumerate(((null_pair_1,null_pair_2),(test_pair_1,test_pair_2))):
        
        donor1,day1,rep1=dataset_pair[0].split('_')
        donor2,day2,rep2=dataset_pair[1].split('_')
            
        assert donor1==donor2, 'trying to compare data from different donors!'
        donorstr=donor1
        
        #input path    
        datasetstr=dataset_pair[0]+'_'+dataset_pair[1] 
        
        loadnull=True #set to True to skip 1st iteration that learns null model, once learned null model parameters, optparas.py, already exist.
        if (not loadnull and it==0) or it==1:
          
            if it==0:
                #set output path
                datasetstr_null=deepcopy(datasetstr)
                parver=parvernull
            else:
                parver=parvertest
                parver=datasetstr_null+"_"+parver

            runstr = it_label[it]+'_'+parver +'_min' + str(mincount) + '_max' + str(maxcount)
            outpath = output_data_path + dataset_pair[0] + '_' + dataset_pair[1] + '/' + runstr + '/'
            
            time.sleep((run_index+1)/10.)#avoid race condition
            if not os.path.exists(outpath):
                os.makedirs(outpath)
                 
            #write shelloutput to file
            prtfn= setup_outpath_and_logs(outpath,logfilename)
            if it==0:
                prtfn("importing null pair: "+datasetstr+'\n')
            else:
                if loadnull:
                    prtfn('loading learned null paras for '+str(null_pair_1) + ' ' + str(null_pair_2)+' : '+str(null_paras))
                prtfn("importing test pair: "+datasetstr+'\n')

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
            data_strvec=('uni_idx_1','uni_idx_2','uni_counts','uni_vals_1','uni_vals_2','Nreads_1','Nreads_2')
            np.save(outpath+'sparserep.npy',dict(zip(data_strvec,sparse_rep)))
            
            if it==0:
                #initial values for null model learning (copied near optimal ones here to speed up): #TODO decide whether or not to keep this block
                #if os.path.exists(outpath+'optparas.npy'):
                    #null_paras=np.load(outpath+'optparas.npy')
                    #assert len(null_paras)==nparas, "loaded paras for wrong acq model!"
                #else:
                prtfn('learn null:')
                donorstrvec=['P1','P2','Q1','Q2','S1','S2']
                #donorstrvec=('Yzh','KB',  'Luci', 'Kar', 'Azh','GS')  #delete this line of actual donor names before publishing
                if acq_model_type==0:
                    #constr_type=0
                    #defaultnullparasvec=np.asarray( [
                    #[-2.21501853,  2.40694626,  1.16697953,  6.75321541, -9.41467361],
                    #[-2.20317149,  2.19642467,  1.10515637,  6.63847919, -9.57396901],
                    #[ -2.0016999,    1.69206104,   1.14715666,   6.82722752, -11.67739647],
                    #[ -2.08578182,   1.81610271,   1.10418693,   6.74893278, -10.1010326 ],
                    #[-2.15,        1.4,         1.15,        6.6685081,  -9.62994141],
                    #[-2.19686053,  2.54462487,  1.10896311,  6.63110241, -9.31923867]
                    #])
                    
                    #constr_type=1
                    defaultnullparasvec=np.asarray( [
                    [-2.19865022,  2.40811359,  1.1885886 ,  6.73120963, -9.43713528],
                    [-2.18684369,  2.19921088,  1.12710212,  6.61652184, -9.59133643],
                    [ -1.99747218,   1.69221099,   1.15192853,   6.82786948,-11.6720086 ],
                    [ -2.08578182,   1.81610271,   1.10418693,   6.74893278, -10.1010326 ],
                    [-2.15,        1.4,         1.15,        6.6685081,  -9.62994141],
                    [-2.11369677,   2.47065402,   1.08416164,   6.62501443, -10.04874892]
                    ])
                elif acq_model_type==2:
                    defaultnullparasvec=np.asarray( [
                    [-2.19865022,  2.40811359,  1.1885886 ,   -9.43713528],
                    [-2.18684369,  2.19921088,  1.12710212,   -9.59133643],
                    [ -1.99747218,   1.69221099,   1.15192853,  -11.6720086 ],
                    [ -2.08578182,   1.81610271,   1.10418693,    -10.1010326 ],
                    [-2.15,        1.4,         1.15,         -9.62994141],
                    [-2.11369677,   2.47065402,   1.08416164,    -10.04874892]
                    ])
                else:
                    prtfn('not computed yet!')
                    
                dind=[i for i, s in enumerate(donorstrvec) if donorstr in s][0]
                init_null_paras = defaultnullparasvec[dind,:]
                prtfn('init null model paras:'+str(init_null_paras))
                st=time.time()
                outstruct,constr_value=learn_null_model(sparse_rep,acq_model_type,init_null_paras,constr_type=constr_type,prtfn=prtfn)
                prtfn('constr value:')
                prtfn(constr_value)
                prtfn('learning took '+str(time.time()-st))                    
                if not outstruct.success:
                    prtfn('null learning failed!')
                
                optparas=outstruct.x
                np.save(outpath + 'optparas', optparas)
                np.save(outpath + 'success', outstruct.success)
                np.save(outpath + 'outstruct', outstruct)

                null_paras=optparas  #null paras to use from here on
                                
        else:          
            datasetstr_null=datasetstr
            runstr = it_label[it]+'_'+parvernull+'_min' + str(mincount) + '_max' + str(maxcount)
            outpath = output_data_path + dataset_pair[0] + '_' + dataset_pair[1] + '/' + runstr + '/'
            null_paras=  np.load(outpath+'optparas.npy')
            assert len(null_paras)==nparas, "loaded paras for wrong acq model!"

    ###############################diffexpr learning
    diffexpr=True
    if diffexpr:
        logrhofvec,logfvec = get_rhof(null_paras[0],np.power(10,null_paras[-1]))
        dlogfby2=np.diff(logfvec)/2. #1/2 comes from trapezoid integration below

        svec,logfvecwide,f2s_step,smax,s_step=get_svec(null_paras,s_step,smax)
        np.save(outpath+'svec.npy',svec)
        indn1,indn2,sparse_rep_counts,unicountvals_1,unicountvals_2,NreadsI,NreadsII=sparse_rep
        Nsamp=np.sum(sparse_rep_counts)
        logPn1_f=get_logPn_f(unicountvals_1,NreadsI,logfvec,acq_model_type,null_paras)
        
        #flags for 3 remaining code blocks:
        learn_surface=False
        polish_estimate=True
        output_table=False
            
        if learn_surface:
            
            #who is who
            #ait=run_index
            #alp=alpvec[run_index]
            dims=(len(alpvec),len(sbarvec_p))

            #itervec=sbarvec_p
            #dims=(len(itervec1),)
               
            itervec1=deepcopy(alpvec)    #dimension 1 
            #dims=(len(itervec1),)

            #itervec2=sbarvec_m           #dimension 2
            #dims=(len(itervec1),len(itervec2))

            #other Ps paras
            #sbar_m=0
            #bet=1.
            
            shiftMtr =np.zeros(dims)
            Zstore =np.zeros(dims)
            Pnng0_Store=np.zeros(dims)
            Zdashstore =np.zeros(dims)
            nit_list =np.zeros(dims)
            LSurface=np.zeros(dims)
           
            for spit,sbar_p in enumerate(sbarvec_p):
                #sbar_p=sbarvec_p[run_index]  #job dimension
                #spit=run_index
            
                shift=0
                first_shift=0
                sst=time.time()
                #for it in range(np.prod(dims)):
                for it,alp in enumerate(alpvec): 
                    inds=np.unravel_index(it,dims)

                    st=time.time()    
                    
                    if Ps_type=='sym_exp' or Ps_type=='rhs_only' or Ps_type=='cent_gauss':
                        Ps_paras=[alpvec[inds[0]],sbar_p]
                    elif Ps_type=='offcent_gauss':
                        Ps_paras=[alpvec[inds[0]],sbar_p,sbarvec_m[inds[1]]]
                    inds=(it,spit)
                    logPsvec = get_logPs_pm(Ps_paras,smax,s_step,Ps_type)
                                                                                                                                                                                                                                                        
                    shift,Z,Zdash,shift_it=get_shift(logPsvec,null_paras,sparse_rep,acq_model_type,shift,logfvec,logfvecwide,svec,f2s_step,logPn1_f,logrhofvec)
                    
                    if shift_it==-1:
                        LSurface[inds]=np.nan#np.dot(sparse_rep_counts/float(Nsamp),log_Pn1n2-np.log(1-Pn0n0))
                        Pnng0_Store[inds]=0
                        shiftMtr[inds]=shift
                        nit_list[inds]=shift_it
                        Zstore[inds]=Z
                        Zdashstore[inds]=Zdash 
                        continue
                    diffexpr_paras=Ps_paras+[shift]
                    L,Pn0n0= get_diffexpr_likelihood(diffexpr_paras,null_paras,sparse_rep,acq_model_type,logfvec,logfvecwide,svec,smax,s_step,Ps_type,f2s_step,logPn1_f,logrhofvec,True)
                    LSurface[inds]=L
                    Pnng0_Store[inds]=Pn0n0
                    shiftMtr[inds]=shift
                    nit_list[inds]=shift_it
                    Zstore[inds]=Z
                    Zdashstore[inds]=Zdash 
            
                    prtfn('its:('+str(run_index)+', '+str(it)+') shift:'+str(shift)+' Z:'+str(Z)+' Zdash:'+str(Zdash)+':'+str(time.time()-st))
                
            np.save(outpath+'Lsurface'+str(run_index), LSurface)
            np.save(outpath+'Pnng0_Store'+str(run_index), Pnng0_Store)
            np.save(outpath+'nit_list'+str(run_index), nit_list)
            np.save(outpath+'shift'+str(run_index), shiftMtr)
            np.save(outpath+'Zstore'+str(run_index), Zstore)
            np.save(outpath+'Zdashstore'+str(run_index), Zdashstore)
            np.save(outpath+'time_elapsed'+str(run_index),time.time()-sst)
            np.save(outpath+'sbarvec_p',sbarvec_p)
            np.save(outpath+'alpvec',alpvec)
            
            #2D
            #np.save(outpath+'offset',sbarvec_m)
            
        if polish_estimate:
            prtfn('polish parameter estimate')
            st=time.time()
            LSurface=np.load(outpath+'Lsurface'+str(run_index)+'.npy')
            maxind=np.unravel_index(np.argmax(LSurface),(len(sbarvec_p),len(alpvec)))
            
            init_shift=0
            if Ps_type=='sym_exp' or Ps_type=='rhs_only' or Ps_type=='cent_gauss':
                init_paras=[alpvec[maxind[0]],sbarvec_p[maxind[1]]]
                parameter_labels=['alpha', 'sbar']
            elif Ps_type=='offcent_gauss':
                init_paras=[alpvec[maxind[0]],sbar_p,sbarvec_m[maxind[1]]]
                parameter_labels=['alpha', 'sbar_p','second_pos']
            init_paras+=[init_shift]
            parameter_labels+=['shift']

            outstruct,constr_value=learn_diffexpr_model(init_paras,parameter_labels,null_paras,sparse_rep,acq_model_type,logfvec,logfvecwide,\
                         svec,smax,s_step,Ps_type,f2s_step,logPn1_f,logrhofvec,prtfn=prtfn)
            
            prtfn('constr value:')
            prtfn(constr_value)
            prtfn('learning took '+str(time.time()-st))                    
            if not outstruct.success:
                prtfn('diffexpr learning failed!')
            
            opt_diffexpr_paras=outstruct.x
            np.save(outpath + 'opt_diffexpr_paras', opt_diffexpr_paras)
            np.save(outpath + 'diffexpr_success', outstruct.success)
            np.save(outpath + 'diffexpr_outstruct', outstruct)
            
        if output_table: #TODO update!
            st=time.time()
            prtfn('write table: ')
            opt_diffexpr_paras=np.load(outpath + 'opt_diffexpr_paras.npy')
            logPsvec = get_logPs_pm(opt_diffexpr_paras[:-1],smax,s_step,Ps_type)

            pval_expanded=True #which end of the rank list to pull out. else: most contracted
            pval_threshold=0.05  #output all clones with pval below this threshold
            smed_threshold=3.46 #ln(2^5)
            save_table(outpath+datasetstr+"table",pval_expanded,smed_threshold,pval_threshold,svec,logPsvec,subset,sparse_rep)
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
    except IndexError:
        run_index=0
    try:
        output_data_path=sys.argv[6]    #custom path
        input_data_path=sys.argv[7]     #custom path
    except IndexError:
        output_data_path='../../data/Yellow_fever/prepostvaccine/'
        input_data_path='../../output/'
                
    main(inputnull_1,inputnull_2,inputtest_1,inputtest_2,run_index,output_data_path,input_data_path)