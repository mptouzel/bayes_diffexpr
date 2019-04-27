import numpy as np
import time
import sys
from infer_diffexpr_lib import import_data, get_Pn1n2_s, get_sparserep, save_table, get_rhof, get_Ps,callbackFnull,callbackFdiffexpr,constr_fn,NegBinParMtr,get_Ps_pm,PoisPar,NegBinPar,get_likelihood,constr_fn_diffexpr
from functools import partial
from scipy.optimize import minimize
import os
import pandas as pd
import ctypes
from copy import deepcopy

#inputs to mkl library used by numpy to parallelize np.dot 
mkl_rt = ctypes.CDLL('libmkl_rt.so')
num_threads=4 #number of cores available on local machine
mkl_set_num_threads = mkl_rt.MKL_Set_Num_Threads(num_threads)
print(str(num_threads)+' cores available')

def main(null_pair_1,null_pair_2,test_pair_1,test_pair_2,run_index,input_data_path,output_data_path):
  
    parvernull = 'v1_2yr_code_check' #prefix for null model data path
    parvertest = 'v1_2yr_code_check' #prefix for diff expr model data path


######################################Preprocessing###########################################3

    # filter out counts (no filtering for these values, 0 and np.inf, respectively)
    mincount = 0
    maxcount = np.Inf #requires about 20GB RAM. Set to 200 to run on less RAM.
    
    # script paras
    nfbins=800      #number of frequency bins
    smax = 25.0     #maximum absolute logfold change value
    s_step =0.1     #logfold change step size

#######################################0
    
    # Start Computations
    starttime = time.time()
    
    null_pair_1=null_pair_1.split('.')[0][:-1]
    null_pair_2=null_pair_2.split('.')[0][:-1]
    test_pair_1=test_pair_1.split('.')[0][:-1]
    test_pair_2=test_pair_2.split('.')[0][:-1]
    
    case=0 #which P(n|f) model to use (0:NB->Pois,1:Pois->NB,2:NBonly,3:Pois only)
    freq_dtype='float64'

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
                parver=parvernull+'_case'+str(case)
                print("importing null pair: "+datasetstr)
            else:
                parver=parvertest
                parver=datasetstr_null+"_"+parver
                print("importing test pair: "+datasetstr)

            runstr = it_label[it]+'_'+parver +'_min' + str(mincount) + '_max' + str(maxcount)
            outpath = output_data_path + dataset_pair[0] + '_' + dataset_pair[1] + '/' + runstr + '/'
            if not os.path.exists(outpath):
                time.sleep((run_index+1)/10.)#avoid race condition
                os.makedirs(outpath)
                
                
            #write shelloutput to file
            outtxtname='_'.join((null_pair_1,null_pair_2,test_pair_1,test_pair_2,'log.txt'))
            outputtxtfile=open(outpath+outtxtname, 'w')
            outputtxtfile.write('outputting logs to ' + outpath+'\n')
            if it==0:
                outputtxtfile.write("running null pair: "+datasetstr+'\n')
            else:
                outputtxtfile.write("running test pair: "+datasetstr+'\n')

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
            indn1_d,indn2_d,countpaircounts_d,unicountvals_1_d,unicountvals_2_d,NreadsI_d,NreadsII_d=get_sparserep(subset.loc[:,['Clone_count_1','Clone_count_2']])       
            Nsamp=np.sum(countpaircounts_d)
            np.save(outpath+"NreadsI_d.npy",NreadsI_d)
            np.save(outpath+"NreadsII_d.npy",NreadsII_d)
            np.save(outpath+"indn1_d.npy",indn1_d)
            np.save(outpath+"indn2_d.npy",indn2_d)
            np.save(outpath+"countpaircounts_d.npy",countpaircounts_d)
            np.save(outpath + 'unicountvals_1_d.npy', unicountvals_1_d)
            np.save(outpath + 'unicountvals_2_d.npy', unicountvals_2_d)
            
            #set sample size adjustment factor
            repfac=NreadsII_d/NreadsI_d
            outputtxtfile.write('N_II/N_I='+str(NreadsII_d/NreadsI_d))
            np.save(outpath+"repfac_prop.npy",repfac)

            if it==0:
                outputtxtfile.write('learn null:')
                st = time.time()
                partialobjfunc=partial(get_Pn1n2_s,svec=-1,unicountvals_1=unicountvals_1_d, unicountvals_2=unicountvals_2_d, NreadsI=NreadsI_d, NreadsII=NreadsII_d, nfbins=nfbins,repfac=repfac,indn1=indn1_d ,indn2=indn2_d,countpaircounts_d=countpaircounts_d,case=case,freq_dtype=freq_dtype)
                
                #initial values for null model learning (copied near optimal ones here to speed up):
                if os.path.exists(outpath+'optparas.npy'):
                    #initparas=np.load(path + donorstr+'_0_F1_'+donorstr+'_0_F2/' + runstr + '/'+'optparas.npy')
                    paras=np.load(outpath+'optparas.npy')
                else:
                    donorstrvec=('S1','P2',  'Q2', 'Q1', 'S2','P1')
                    #donorstrvec=('Yzh','KB',  'Luci', 'Kar', 'Azh','GS')  #delete actual donor names before publishing
                    if case==0:
                        defaultnullparasvec=np.asarray( [
                        [-2.15 ,                1.4,     1.15,      7.0,-9.636235],  
                        [-2.194, np.power(10,0.334),   1.0517,      7.0,-9.636235],  
                        [-2.15 ,                1.7,      1.1,      7.0,-9.636235],  
                        [-2.15 ,                0.5,      1.3,      7.0,-9.636235],  
                        [-2.191965,        2.614699, 1.166377, 6.559887,-9.636235],  
                        [-2.18 ,                2.4,      1.3,      7.0,-9.636235]  
                        ])
                    elif case==1:
                        defaultnullparasvec=np.asarray( [
                        [-2.05 , 1.4,    1.15,-9.336235],  
                        [-2.094,         0.334,  1.0517,-9.336235],  
                        [-2.05 , 1.7,     1.1,-9.336235],  
                        [-2.05 , 0.5,     1.3,-9.336235],  
                        [-2.191965 , 2.614699 , 1.166377, -9.336235],  
                        [-2.18 , 2.4,     1.3,-9.336235]  
                        ])
                    dind=[i for i, s in enumerate(donorstrvec) if donorstr in s][0]
                    initparas = defaultnullparasvec[dind,:]
                    
                    #constrained optimization. constraint: N<f>_{\rho(f)}=1, N~Nsamp/(1-P(n1=0,n2=0))
                    condict={'type':'eq','fun':constr_fn,'args': (NreadsI_d,NreadsII_d,unicountvals_1_d,unicountvals_2_d,indn1_d,indn2_d,countpaircounts_d,case,freq_dtype)}
                    print(initparas)
                    outstruct = minimize(partialobjfunc, initparas, method='SLSQP', callback=callbackFnull, constraints=condict,tol=1e-6,options={'ftol':1e-8 ,'disp': True,'maxiter':300})
                    
                    #print(initparas)
                    #outstruct = minimize(partialobjfunc, initparas, method='SLSQP', callback=callbackF, tol=1e-6,options={'ftol':1e-8 ,'disp': True,'maxiter':300})
                    
                    
                    for key,value in outstruct.items():
                        outputtxtfile.write(key+':'+str(value)+'\n')
                    if not outstruct.success:
                        print('null learning failed!')
                    
                    optparas=outstruct.x
                    np.save(outpath + 'optparas', optparas)
                    np.save(outpath + 'success', outstruct.success)
                    np.save(outpath + 'outstruct', outstruct)

                    paras=optparas
                    
                    np.save(outpath + 'paras', paras) #null paras to use from here on
                outputtxtfile.write("elapsed " + str(np.round(time.time() - st))+'\n')
                outputtxtfile.close()
        else:
            datasetstr_null=datasetstr
            runstr = it_label[it]+'_'+parvernull+'_case'+str(case) +'_min' + str(mincount) + '_max' + str(maxcount)
            outpath = output_data_path + dataset_pair[0] + '_' + dataset_pair[1] + '/' + runstr + '/'
            paras=  np.load(outpath+'optparas.npy')
            print('loading learned null paras for '+str(dataset_pair[0]) + ' ' + str(dataset_pair[1])+' : '+str(paras))
            outputtxtfile.write('loading learned null paras for '+str(dataset_pair[0]) + ' ' + str(dataset_pair[1])+' : '+str(paras))
            
    ################################diffexpr learning
    diffexpr=True
    if diffexpr:
        
        #get Pn1n2_s
        logrhofvec,logfvec = get_rhof(paras[0],nfbins,np.power(10,paras[-1]),freq_dtype)
        #biuld discrete domain of s
        s_step_old=s_step
        logf_step=logfvec[1] - logfvec[0] #use natural log here since f2 increments in increments in exp().  
        f2s_step=int(round(s_step/logf_step)) #rounded number of f-steps in one s-step
        s_step=float(f2s_step)*logf_step
        smax=s_step*(smax/s_step_old)
        svec=s_step*np.arange(0,int(round(smax/s_step)+1))   
        svec=np.append(-svec[1:][::-1],svec)
 
        #compute conditional P(n1,n2|s) and P(n1=0,n2=0|s)
        print('calc Pn1n2_s: ')
        st = time.time()
        if os.path.exists(outpath+'Pn1n2_s_d.npy'):
            Pn1n2_s=np.load(outpath+'Pn1n2_s_d.npy')
            Pn0n0_s=np.load(outpath+'Pn0n0.npy')
            logPn1_f=np.log(np.load(outpath + 'Pn1_f.npy'))
        else:
            Pn1n2_s, unicountvals_1_d, unicountvals_2_d, Pn1_f, fvec, Pn2_s, Pn0n0_s,svec = get_Pn1n2_s(paras, svec, unicountvals_1_d, unicountvals_2_d,  NreadsI_d, NreadsII_d, nfbins,repfac,indn1=indn1_d,indn2=indn2_d,freq_dtype=freq_dtype,s_step=s_step)
            np.save(outpath + 'Pn1n2_s_d', Pn1n2_s)
            np.save(outpath + 'Pn0n0',Pn0n0_s)
            np.save(outpath + 'Pn1_f',Pn1_f)
            logPn1_f=np.log(Pn1_f)
            outputtxtfile.write("calc Pn1n2_s elapsed " + str(np.round(time.time() - st))+'\n')
        
        #flags for 3 remaining code blocks:
        learn_surface=True
        polish_estimate=False
        output_table=False
            
        if learn_surface:
            print('calc surface: \n')
            st = time.time()
            
            #define grid search parameters  
            npoints=20
            nsbarpoints=npoints
            sbarvec=np.linspace(0.01,5,nsbarpoints)
            nalppoints=npoints
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
            print("optalp="+str(optalp)+" ("+str(alpvec[0])+","+str(alpvec[-1])+"),optsbar="+str(optsbar)+", ("+str(sbarvec[0])+","+str(sbarvec[-1])+") \n")
            outputtxtfile.write("optalp="+str(optalp)+" ("+str(alpvec[0])+","+str(alpvec[-1])+"),optsbar="+str(optsbar)+", ("+str(sbarvec[0])+","+str(sbarvec[-1])+") \n")
            outputtxtfile.write("surface elapsed " + str(np.round(time.time() - st))+'\n')
            
        if polish_estimate:
            
            optsbar=np.load(outpath + 'optsbar.npy')
            optalp=np.load(outpath + 'optalp.npy')
            print('polish parameter estimate from '+str(optalp)+' '+str(optsbar))
            
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
                                            m_low=m_low,m_high=m_high, mvec=mvec,Nsamp=Nsamp,logPn1_f=logPn1_f,case=case)
            
            condict={'type':'eq','fun':constr_fn_diffexpr,'args': (paras,svec,smax,s_step,indn1_d,indn2_d,np.exp(logfvec),np.exp(logfvecwide),np.exp(logrhofvec),\
                                                                unicountvals_1_d,unicountvals_2_d,countpaircounts_d,\
                                                                NreadsI, NreadsII,nfbins, f2s_step,\
                                                                m_low,m_high,mvec,Nsamp,logPn1_f,case)\
                }

            outstruct = minimize(partialobjfunc, initparas, method='SLSQP', callback=callbackFdiffexpr, constraints=condict,tol=1e-6,options={'ftol':1e-8 ,'disp': True,'maxiter':300})
            np.save(outpath + 'outstruct_diffexpr', outstruct)
            
        if output_table:
            st=time.time()
            outputtxtfile.write('write table: ')
            optsbar=np.load(outpath + 'sbaropt.npy')
            optalp=np.load(outpath + 'alpopt.npy')
            svec=np.load(outpath + 'svec.npy')
            Pn1n2_s=np.load(outpath + 'Pn1n2_s_d.npy')
            Psopt=np.load(outpath + 'Psopt.npy')
            
            pval_expanded=True #which end of the rank list to pull out. else: most contracted
            pval_threshold=0.05  #output all clones with pval below this threshold
            smed_threshold=3.46 #ln(2^5)
            save_table(outpath+datasetstr+"table",pval_expanded,smed_threshold,pval_threshold,svec, Psopt, Pn1n2_s, Pn0n0_s,subset,unicountvals_1_d,unicountvals_2_d,indn1_d,indn2_d)
            print(" elapsed " + str(np.round(time.time() - st))+'\n')

    # end computations
    endtime = time.time()
    print('program elapsed:' + str(endtime - starttime))
    
if __name__ == "__main__": 
    inputnull_1=sys.argv[1]
    inputnull_2=sys.argv[2]
    inputtest_1=sys.argv[3]
    inputtest_2=sys.argv[4]
    
    #default command: "python infer_diffexpr nulldataset1.txt nulldataset2.txt diffexprdataset1.txt diffexprdataset2.txt"
    
    #optional args
    try:
        run_index=int(sys.argv[5])      #run specific index 
        output_data_path=sys.argv[6]    #custom path
        input_data_path=sys.argv[7]     #custom path
    except IndexError:
        run_index=0
        output_data_path='../Yellow_fever/S1_S2_day45_and_2yr_data/'
        input_data_path='../output/'
                
    main(inputnull_1,inputnull_2,inputtest_1,inputtest_2,run_index,output_data_path,input_data_path)