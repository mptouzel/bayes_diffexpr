#!/home/max/anaconda2/bin

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
from lib.model import get_Pn1n2_s
from functools import partial
#inputs to mkl library used by numpy to parallelize np.dot 
import ctypes
mkl_rt = ctypes.CDLL('libmkl_rt.so')
num_threads=4 #number of cores available on local machine
mkl_set_num_threads = mkl_rt.MKL_Set_Num_Threads(num_threads)
print(str(num_threads)+' cores available')

if __name__ == '__main__':
    
    rootpath='../../../'
    acq_model_type=2
    constr_type=1

    donorstrvec=('S1','P2', 'S2','P1','Q2', 'Q1')
    dayvec=np.array(['pre0','0','7','15','45'])
    runname='test1'
    for donor in donorstrvec:
        for day in dayvec:
                
            outpath=rootpath+'output/'+donor+'_'+day+'_F1_'+donor+'_'+day+'_F2/null_pair_v1_ct_'+str(constr_type)+'_mt_'+str(acq_model_type)+'_min0_maxinf/'
            parasopt=np.load(outpath+'optparas.npy')
            sparse_rep=np.load(outpath+'sparserep.npy').item()
            partialobjfunc=partial(get_Pn1n2_s,svec=-1, sparse_rep=sparse_rep.values(),acq_model_type=acq_model_type)
            
            #cycle over all 10 pairs
            if acq_model_type<2:
                pair_inds=((0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4))
            else:
                pair_inds=((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))

            npoints=5
            lowfac=0.95
            logLikelihood_NBPois=np.zeros((len(pair_inds),npoints,npoints))
            for pairit,pair_ind in enumerate(pair_inds):
                para1vec=np.linspace(parasopt[pair_ind[0]]*lowfac,parasopt[pair_ind[0]]+parasopt[pair_ind[0]]*(1-lowfac),npoints)
                para2vec=np.linspace(parasopt[pair_ind[1]]*lowfac,parasopt[pair_ind[1]]+parasopt[pair_ind[1]]*(1-lowfac),npoints)
                for pit1,para1 in enumerate(para1vec):
                    for pit2,para2 in enumerate(para2vec):
                        print(str(pit1)+str(pit2))
                        paras=deepcopy(parasopt)
                        paras[pair_ind[0]]=para1
                        paras[pair_ind[1]]=para2
                        logLikelihood_NBPois[pairit,pit1,pit2]=partialobjfunc(paras)
            np.save(outpath+'local_likelihood_'+donor+'_'+day+'.npy',logLikelihood_NBPois)

    
