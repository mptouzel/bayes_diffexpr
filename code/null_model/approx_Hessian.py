#!/home/max/anaconda2/bin
import numpy as np
import math
from copy import deepcopy
import pandas as pd
import scipy.stats
import pylab as pl
from pylab import rcParams
from functools import partial
import numexpr as ne
import ctypes
import sys
from infer_diffexpr_lib import get_Pn1n2_s
mkl_rt = ctypes.CDLL('libmkl_rt.so')
num_threads=3
mkl_set_num_threads = mkl_rt.MKL_Set_Num_Threads(num_threads)

if __name__ == '__main__':
    
    cluster=True
    if cluster:
        rootpath='/users/puelma/scripts/diffexpr/'
    else:
        rootpath='/home/max/Dropbox/scripts/Projects/immuno/diffexpr/'

    casestrvec=(r'$NB\rightarrow Pois$')
    donorstrvec=('S1','P2', 'S2','P1','Q2', 'Q1')
    dayvec=np.array(['pre0','0','7','15','45'])
    runname='test1'
    donor=sys.argv[1]
    day=sys.argv[2]
        
    outpath=rootpath+'outdata_all/'+donor+'_'+day+'_F1_'+donor+'_'+day+'_F2/min0_maxinf_v1__case0_noconst_case0/'
    setname=outpath+'outstruct.npy'
    parasopt=np.load(setname).flatten()[0].x
    NreadsI_d=np.load(outpath+"NreadsI_d.npy")
    NreadsII_d=np.load(outpath+"NreadsII_d.npy")
    indn1_d=np.load(outpath+"indn1_d.npy")
    indn2_d=np.load(outpath+"indn2_d.npy")
    countpaircounts_d=np.load(outpath+"countpaircounts_d.npy")
    unicountvals_1_d=np.load(outpath + 'unicountvals_1_d.npy')
    unicountvals_2_d=np.load(outpath + 'unicountvals_2_d.npy')
    nfbins=800
    repfac=NreadsII_d/NreadsI_d
    case=0
    partialobjfunc=partial(get_Pn1n2_s,svec=-1,unicountvals_1=unicountvals_1_d, unicountvals_2=unicountvals_2_d, NreadsI=NreadsI_d, NreadsII=NreadsII_d, nfbins=nfbins,repfac=repfac,indn1=indn1_d ,indn2=indn2_d,countpaircounts_d=countpaircounts_d,case=case)

    #cycle over all 10 pairs
    pair_inds=((0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4))
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
    np.save('nullerrorbars_hessian_noconst_'+donor+'_'+day+'.npy',logLikelihood_NBPois)

    
