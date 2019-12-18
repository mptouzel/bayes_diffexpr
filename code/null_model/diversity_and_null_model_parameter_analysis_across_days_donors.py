# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# %matplotlib inline
import sys,os
root_path = os.path.abspath(os.path.join('..'))
if root_path not in sys.path:
    sys.path.append(root_path)
# %run -i '../lib/utils/ipynb_setup.py'
# import lib.utils.plotting
# from lib.utils.plotting import plot_n1_vs_n2,add_ticks
# from lib.utils.prob_utils import get_distsample
# from lib.proc import get_sparserep,import_data
# from lib.model import get_Pn1n2_s, get_rhof, NegBinParMtr,get_logPn_f,get_model_sample_obs
# from lib.learning import constr_fn,callback,learn_null_model
# import lib.learning
# %load_ext autoreload
# %autoreload 2


# from scipy.interpolate import interp1d
# from scipy import stats
# from scipy.stats import poisson
# from scipy.stats import nbinom
# from scipy.stats import rv_discrete

# +
import matplotlib.pyplot as pl
import seaborn as sns

for_paper=True
if not for_paper:
    pl.rc("figure", facecolor="gray",figsize = (8,8))
    pl.rc('lines',markeredgewidth = 2)
    pl.rc('font',size = 24)
else:
    pl.rc("figure", facecolor="none",figsize = (3.5,3.5))
    pl.rc('lines',markeredgewidth = 1)
    pl.rc('font',size = 10)
    sns.set_style("whitegrid", {'axes.grid' : True})

params= {'text.latex.preamble' : [r'\usepackage{amsmath}']}
pl.rcParams.update(params)
pl.rc('text', usetex=True)
# -

# Make dataframe

# casestrvec=(r'$NB\rightarrow Pois$',r'$Pois \rightarrow NB$','$NB$','$Pois$')
casestrvec=(r'$NB\rightarrow Pois$','$NB$','$Pois$')
casevec=[0,2,3]
donorvec=['P1','P2','Q1','Q2','S1','S2']
# donorvec=("Azh","KB","Yzh","GS","Kar","Luci")
dayvec=range(5)
nparasvec=(4,3,1)
outstructs=np.empty((len(casevec),len(donorvec),len(dayvec)),dtype=dict)
out_df=pd.DataFrame()
# for cit, case in enumerate(casevec):
case=2
for dit,donor in enumerate(donorvec):
    day=0
#     for ddit,day in enumerate(dayvec):
    data_name=donor+'_'+str(day)+"_F1_"+donor+'_'+str(day)+'_F2'
#             runstr='../../../output/'+dataname+'/null_pair_v1_null_ct_1_acq_model_type'+str(case)+'_min0_maxinf/'
    runstr='../../../output/'+data_name+'/null_pair_v1_ct_1_mt_'+str(case)+'_min0_maxinf/'
    setname=runstr+'outstruct.npy'
    outstructs[cit,dit,ddit]=np.load(setname).flatten()[0]
    tmpdict=np.load(setname).flatten()[0]
    tmpdict['day']=day
    tmpdict['donor']=donor
    tmpdict['mt']=case
    out_df=out_df.append(tmpdict,ignore_index=True)

# +
fig,ax=pl.subplots(3,4,figsize=(16,12))
lowrange=(-2.12,-0.9,0.7,6.5)
highrange=(-1.6,0.7,1.3,7.2)

casestrvec=(r'$NB\rightarrow Pois$')#,'$NB$','$Pois$')
casevec=[0]#,2,3]
donorvec=('S2')#S1','P2',  'Q2', 'Q1', 'S2','P1')
# dayvec=np.array([15])#range(5)
# nparasvec=(4,3,1)
# outstructs=np.empty((len(casevec),len(donorvec),len(dayvec)),dtype=dict)
out_df=pd.DataFrame()
for cit, case in enumerate(casevec):
    for dit,donor in enumerate(donorvec):
        runstr='outdata_all/'+donor+'_'+str(day)+"_0_"+str(day)+'_1_case_'#+str(case)+'_'
        setname=runstr+'outstruct_mmax1e6.npy'
        outstructs[cit,dit,ddit]=np.load(setname).flatten()[0]
        tmpdict=np.load(setname).flatten()[0]
        tmpdict['day']=day
        tmpdict['donor']=donor
        tmpdict['case']=case
        out_df=out_df.append(tmpdict,ignore_index=True)
            
nparasvec=(4,3,1)
for cit,case in enumerate(casevec):
    nparas=nparasvec[cit]
    for pit in range(nparas):
        ax[cit,pit].hist([struct.x[pit] for struct in outstructs[cit].flatten()])
        ax[cit,pit].set_xlim(lowrange[pit],highrange[pit])
parastrvec=(r'$\alpha$',r'$\log_{10}a$',r'$\gamma$','$\log_{10}M$')
for pit in range(4):
    ax[-1,pit].set_xlabel(parastrvec[pit],size=24)
casestrvec=(r'$NB\rightarrow Pois$','$NB$','$Pois$')
for cit in range(3):
    ax[cit,0].set_ylabel(casestrvec[cit],size=24)
    
casestrvec=(r'$NB\rightarrow Pois$','$NB$','$Pois$')
casevec=[0,2,3]
donorvec=("Azh","Yzh")
nparasvec=(4,3,1)
outstructs=np.empty((len(casevec),len(donorvec)),dtype=dict)
for cit, case in enumerate(casevec):
    for dit,donor in enumerate(donorvec):
        day=5
        runstr='nullparasv7/learnnulltestv7_'+donor+'_'+str(day)+"_0_"+str(day)+'_1_case_'+str(case)+'_'
        setname=runstr+'outstruct_mmax1e6.npy'
        outstructs[cit,dit]=np.load(setname).flatten()[0]
        
nparas=4
for cit,case in enumerate(casevec):
    nparas=nparasvec[cit]
    for pit in range(nparas):
        data=[struct.x[pit] for struct in outstructs[cit].flatten()]
        print(data)
        ax[cit,pit].hist(data,color='r')
fig.suptitle(r'$\rho(f)\sim f^\alpha$, $\sigma^2_m=Mf+a(Mf)^\gamma$')
fig.savefig('sameday_nullmodel_learned_paras.pdf',format='pdf',dpi=1000, bbox_inches='tight')

# +
fig,ax=pl.subplots(3,4,figsize=(16,12))
lowrange=(-2.12,-0.9,0.7,6.5)
highrange=(-1.6,0.7,1.3,7.2)

casestrvec=(r'$NB\rightarrow Pois$','$NB$','$Pois$')
casevec=[0,2,3]
donorstrvec=('S1','P2',  'Q2', 'Q1', 'S2','P1')
dayvec=np.array([15])#range(5)
nparasvec=(4,3,1)
outstructs=np.empty((len(casevec),len(donorvec),len(dayvec)),dtype=dict)
out_df=pd.DataFrame()
for cit, case in enumerate(casevec):
    for dit,donor in enumerate(donorvec):
        for ddit,day in enumerate(dayvec):
            runstr='outdata_all/'+donor+'_'+str(day)+"_0_"+str(day)+'_1_case_'#+str(case)+'_'
            setname=runstr+'outstruct_mmax1e6.npy'
            outstructs[cit,dit,ddit]=np.load(setname).flatten()[0]
            tmpdict=np.load(setname).flatten()[0]
            tmpdict['day']=day
            tmpdict['donor']=donor
            tmpdict['case']=case
            out_df=out_df.append(tmpdict,ignore_index=True)
            
nparasvec=(4,3,1)
for cit,case in enumerate(casevec):
    nparas=nparasvec[cit]
    for pit in range(nparas):
        ax[cit,pit].hist([struct.x[pit] for struct in outstructs[cit].flatten()])
        ax[cit,pit].set_xlim(lowrange[pit],highrange[pit])
parastrvec=(r'$\alpha$',r'$\log_{10}a$',r'$\gamma$','$\log_{10}M$')
for pit in range(4):
    ax[-1,pit].set_xlabel(parastrvec[pit],size=24)
casestrvec=(r'$NB\rightarrow Pois$','$NB$','$Pois$')
for cit in range(3):
    ax[cit,0].set_ylabel(casestrvec[cit],size=24)
    
casestrvec=(r'$NB\rightarrow Pois$','$NB$','$Pois$')
casevec=[0,2,3]
donorvec=("Azh","Yzh")
nparasvec=(4,3,1)
outstructs=np.empty((len(casevec),len(donorvec)),dtype=dict)
for cit, case in enumerate(casevec):
    for dit,donor in enumerate(donorvec):
        day=5
        runstr='nullparasv7/learnnulltestv7_'+donor+'_'+str(day)+"_0_"+str(day)+'_1_case_'+str(case)+'_'
        setname=runstr+'outstruct_mmax1e6.npy'
        outstructs[cit,dit]=np.load(setname).flatten()[0]
        
nparas=4
for cit,case in enumerate(casevec):
    nparas=nparasvec[cit]
    for pit in range(nparas):
        data=[struct.x[pit] for struct in outstructs[cit].flatten()]
        print(data)
        ax[cit,pit].hist(data,color='r')
fig.suptitle(r'$\rho(f)\sim f^\alpha$, $\sigma^2_m=Mf+a(Mf)^\gamma$')
fig.savefig('sameday_nullmodel_learned_paras.pdf',format='pdf',dpi=1000, bbox_inches='tight')
# -

nparas=1
fig,ax=pl.subplots(1,nparas,figsize=(4*nparas,4))
ax.hist([struct.x[0] for struct in outstructs[2].flatten()])

nparas=3
fig,ax=pl.subplots(1,nparas,figsize=(4*nparas,4))
for pit in range(nparas):
    ax[pit].hist([struct.x[pit] for struct in outstructs[1].flatten()])

nparas=4
fig,ax=pl.subplots(1,nparas,figsize=(16,4))
for pit in range(nparas):
    ax[pit].hist([struct.x[pit] for struct in outstructs[0].flatten()])

# casestrvec=(r'$NB\rightarrow Pois$',r'$Pois \rightarrow NB$','$NB$','$Pois$')
casestrvec=(r'$NB\rightarrow Pois$','$NB$','$Pois$')
casevec=[0,2,3]
donorvec=("Azh","Yzh")
nparasvec=(4,3,1)
outstructs=np.empty((len(casevec),len(donorvec)),dtype=dict)
for cit, case in enumerate(casevec):
    for dit,donor in enumerate(donorvec):
        day=5
        runstr='nullparasv7/learnnulltestv7_'+donor+'_'+str(day)+"_0_"+str(day)+'_1_case_'+str(case)+'_'
        setname=runstr+'outstruct_mmax1e6.npy'
        outstructs[cit,dit]=np.load(setname).flatten()[0]

import numpy.polynomial.polynomial as poly
casestrvec=(r'$NB\rightarrow Pois$')
casevec=[0]
donorstrvec=('S1','P2',  'Q2', 'Q1', 'S2','P1')
# donorstrvec=('S1','P2',  'Q2', 'S2','P1','S2')
dayvec=np.array(['pre0','0','7','15','45'])
outstructs=np.empty((len(dayvec),len(donorstrvec)),dtype=object)
errorbars=np.empty((len(dayvec),len(donorstrvec)),dtype=object)
N_total=np.zeros((len(dayvec),len(donorstrvec)))
for ddit, day in enumerate(dayvec):
    for dit,donor in enumerate(donorstrvec):
        try:
#             runstr='outdata_all/'+donor+'_'+day+'_F1_'+donor+'_'+day+'_F2/min0_maxinf_v1__case0/'
            runstr='outdata_all/'+donor+'_'+day+'_F1_'+donor+'_'+day+'_F2/min0_maxinf_v1_case0_noconst_case0/'
            setname=runstr+'outstruct.npy'
            parastmp=np.load(setname).flatten()[0].x
            outstructs[ddit,dit]=np.load(setname).flatten()[0].x
#             nfbins=800
#             rhofvec,fvec = get_rhof(outstructs[ddit,dit][0],nfbins,np.power(10,outstructs[ddit,dit][4]))
#             integ=np.exp(np.log(rhofvec)+2*np.log(fvec))
#             N_total[ddit,dit]=1./np.dot(np.diff(np.log(fvec))/2,integ[1:]+integ[:-1])
#             lowfac=0.95
#             npoints=5
#             logLikelihood_NBPois=np.load('nullerrorbars_'+donor+'_'+day+'.npy')
#             coeffs=np.zeros((len(parastmp),3))
#             errorbarstmp=np.zeros(len(parastmp))
#             for pit,para in enumerate(parastmp): 
#                 vec=np.linspace(para*lowfac,para+para*(1-lowfac),npoints)
#                 coeffs[pit,:]=poly.polyfit(vec-vec[2], logLikelihood_NBPois[pit,:], 2)
#                 errorbarstmp[pit]=np.fabs(1/np.sqrt(2*coeffs[pit,0]))
#             errorbars[ddit,dit]=errorbarstmp
        except IOError:
            outstructs[ddit,dit]=np.zeros(5)
            print(day+' '+donor)

for ddit, day in enumerate(dayvec):
    print(day)
    for dit,donor in enumerate(donorstrvec):
        print(donor)
        st=time.time()
        try:
            outpath='outdata_all/'+donor+'_'+day+'_F1_'+donor+'_'+day+'_F2/min0_maxinf_v1__case0/'
            setname=outpath+'outstruct.npy'
            parastmp=np.load(setname).flatten()[0].x
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
            npoints=5
            logLikelihood_NBPois=np.zeros((npoints,npoints))
            lowfac=0.95
            for pit, para in enumerate(parastmp):
                paratmpp=deepcopy(parastmp)
                for ppit,ppara in enumerate(np.linspace(para*lowfac,para+para*(1-lowfac),npoints)):
                    paratmpp[pit]=ppara
                    logLikelihood_NBPois[pit,ppit]=partialobjfunc(paratmpp)
                    
            np.save('nullerrorbars_'+donor+'_'+day+'.npy',logLikelihood_NBPois)
        except IOError:
            outstructs[ddit,dit]=np.zeros(5)
        et=time.time()
        print('elapsed:'+str(et-st))

# fit quadratic approximation to likelihood:

npoints=5

# +
# def paras2M(paras):
#     M=np.zeros((npoints,npoints))
#     inds=np.cumsum(range(npoints+1))
#     for row_ind in range(1,npoints):
#         M[row_ind,:row_ind]=paras[inds[row_ind-1]:inds[row_ind]]
#     i_upper = np.triu_indices(5)
#     M[i_upper] = M.T[i_upper]
#     M+=np.diag(paras[inds[-2]:inds[-1]])
#     return M

# +
# def MSError_fcn(paras,data):
#     M=paras2M(paras)
#     res=0
#     for val in data.itertuples():
#         res+=np.power(val.y+M.dot(val.x).dot(val.x)/2,2)
#     return res/len(data)
# -

# First learn diagonal elements from coordinate axis variation:

# +
donorstrvec=('S1','P2',  'Q2', 'Q1', 'S2','P1')
dayvec=np.array(['pre0','0','7','15','45'])
lowfac=0.95
npoints=5
import numpy.polynomial.polynomial as poly
diag_entries=np.empty((len(dayvec),len(donorstrvec)),dtype=object)
fig1, axarr = pl.subplots(1, 5)
fig2,ax= pl.subplots(1, 1)
for ddit, day in enumerate(dayvec):
    for dit,donor in enumerate(donorstrvec):
        try:
            data_df=pd.DataFrame(columns=['x','y'])
            data_name=donor+'_'+str(day)+"_F1_"+donor+'_'+str(day)+'_F2'
   
            outpath='output/'+donor+'_'+day+'_F1_'+donor+'_'+day+'_F2/null_pair_v1_ct_1_mt_'+str(case)+'_min0_maxinf/'
            sparse_rep=np.load(outpath+"sparse_rep.npy").items()            
            parasopt=np.load(outpath+'optparas.npy')
            logLikelihood_NBPois=np.load('local_likelihood_'+donor+'_'+day+'.npy')*len(sparse_rep['uni_counts'])

            logLikelihood_NBPois_diag=np.zeros((5,5))
            for pit in range(5):
                if pit==0:
                    logLikelihood_NBPois_diag[pit,:]=logLikelihood_NBPois[pit][:,2]
                else:
                    logLikelihood_NBPois_diag[pit,:]=logLikelihood_NBPois[pit-1][2,:]
    #         ax.scatter(paras[1],paras[2]) 
    #         print(np.load(setname).flatten()[0])
            #format data
    #         for pit, para in enumerate(paras):
    #             paratmpp=deepcopy(paras)
    #             for ppit,ppara in enumerate(np.linspace(para*lowfac,para+para*(1-lowfac),npoints)):
    #                 paratmpp[pit]=ppara
    #                 data_df=data_df.append(pd.Series({'x':paratmpp-paras,'y':(logLikelihood_NBPois[pit,ppit]-logLikelihood_NBPois[pit,2])}),ignore_index=True)
    #         partial_MSError_fcn=partial(MSError_fcn,data=data_df)
            #derive initial parameters as variance of cardinal directions

            coeffs=np.zeros((5,3))
    #         errorbarstmp=np.zeros(len(parastmp))
            for pit,para in enumerate(parasopt): 
                vec=np.linspace(para*lowfac,para+para*(1-lowfac),npoints)
                coeffs[pit,:]=poly.polyfit(vec-vec[2], logLikelihood_NBPois_diag[pit,:]-logLikelihood_NBPois_diag[pit,2], 2)
            if np.argmin(logLikelihood_NBPois[1,:])!=2:
                print(str(day)+' '+str(donor)+' '+str(np.argmin(logLikelihood_NBPois[1,:])))

    #         outpath='outdata_all/'+donor+'_'+day+'_F1_'+donor+'_'+day+'_F2/min0_maxinf_v1__case0/'
    #         setname=outpath+'outstruct.npy'
            ax.scatter(paras[1],paras[2])
    #         print(np.load(setname).flatten()[0])
            #format data
    #         for pit, para in enumerate(paras):
    #             paratmpp=deepcopy(paras)
    #             for ppit,ppara in enumerate(np.linspace(para*lowfac,para+para*(1-lowfac),npoints)):
    #                 paratmpp[pit]=ppara
    #                 data_df=data_df.append(pd.Series({'x':paratmpp-paras,'y':(logLikelihood_NBPois[pit,ppit]-logLikelihood_NBPois[pit,2])}),ignore_index=True)
    #         partial_MSError_fcn=partial(MSError_fcn,data=data_df)
            #derive initial parameters as variance of cardinal directions
            coeffs=np.zeros((len(paras),3))
    #         errorbarstmp=np.zeros(len(parastmp))
            for pit,para in enumerate(parasopt): 
                vec=np.linspace(para*lowfac,para+para*(1-lowfac),npoints)
                coeffs[pit,:]=poly.polyfit(vec-vec[2], logLikelihood_NBPois_diag[pit,:]-logLikelihood_NBPois_diag[pit,2], 2)
            diag_entries[ddit,dit]=coeffs[:,2]
    #             errorbarstmp[pit]=np.fabs(1/np.sqrt(2*coeffs[pit,0]))
    #         initparas=-1*np.ones(15)
    #         outstruct_quad = minimize(partial_MSError_fcn, initparas, method='SLSQP', tol=1e-10,options={'ftol':1e-10 ,'disp': False,'maxiter':200})
    #         print(outstruct_quad.x)
    #         M=paras2M(outstruct_quad.x)
    #         Cov=np.linalg.inv(M)
    #         errorbars[ddit,dit]=np.sqrt(np.diag(Cov))

            #test
    #         print(paras)
            for pit, para in enumerate(parasopt):
                paratmpp=deepcopy(paras)
                vec=np.linspace(para*lowfac,para+para*(1-lowfac),5)
                paradense=np.linspace(para*lowfac,para+para*(1-lowfac),100)
                fit=poly.polyval(paradense-vec[2],coeffs[pit,:])
                axtmp=axarr[pit].plot(paradense-para,coeffs[pit,2]*(paradense-vec[2])**2)
                axarr[pit].plot(vec-para,(logLikelihood_NBPois_diag[pit,:]-logLikelihood_NBPois_diag[pit,2]),'x',color=axtmp[-1].get_color())
    #             axarr[pit].plot(vec-para,(logLikelihood_NBPois[pit,:]-logLikelihood_NBPois[pit,2]),'x',color=axtmp[-1].get_color())
    #             print(-M[pit,pit]/2)
        except IOError:
            print('hi')
            
pd.DataFrame(diag_entries)
# -

donorstrvec=('S1','P2',  'Q2', 'Q1', 'S2','P1')
dayvec=np.array(['pre0','0','7','15','45'])
lowfac=0.95
npoints=5
import numpy.polynomial.polynomial as poly
diag_entries=np.empty((len(dayvec),len(donorstrvec)),dtype=object)
fig1, axarr = pl.subplots(1, 5)
fig2,ax= pl.subplots(1, 1)
for ddit, day in enumerate(dayvec):
    for dit,donor in enumerate(donorstrvec):
        data_df=pd.DataFrame(columns=['x','y'])
        outpath='outdata_all/'+donor+'_'+day+'_F1_'+donor+'_'+day+'_F2/min0_maxinf_v1__case0/'
        countpaircounts_d=np.load(outpath+"countpaircounts_d.npy")
        logLikelihood_NBPois=np.load('nullerror_diag/nullerrorbars_'+donor+'_'+day+'.npy')*len(countpaircounts_d)
        if np.argmin(logLikelihood_NBPois[1,:])!=2:
            print(str(day)+' '+str(donor)+' '+str(np.argmin(logLikelihood_NBPois[1,:])))
           
        outpath='outdata_all/'+donor+'_'+day+'_F1_'+donor+'_'+day+'_F2/min0_maxinf_v1__case0/'
        setname=outpath+'outstruct.npy'
        paras=np.load(setname).flatten()[0].x
        ax.scatter(paras[1],paras[2])
#         print(np.load(setname).flatten()[0])
        #format data
#         for pit, para in enumerate(paras):
#             paratmpp=deepcopy(paras)
#             for ppit,ppara in enumerate(np.linspace(para*lowfac,para+para*(1-lowfac),npoints)):
#                 paratmpp[pit]=ppara
#                 data_df=data_df.append(pd.Series({'x':paratmpp-paras,'y':(logLikelihood_NBPois[pit,ppit]-logLikelihood_NBPois[pit,2])}),ignore_index=True)
#         partial_MSError_fcn=partial(MSError_fcn,data=data_df)
        #derive initial parameters as variance of cardinal directions
        coeffs=np.zeros((len(paras),3))
#         errorbarstmp=np.zeros(len(parastmp))
        for pit,para in enumerate(paras): 
            vec=np.linspace(para*lowfac,para+para*(1-lowfac),npoints)
            coeffs[pit,:]=poly.polyfit(vec-vec[2], logLikelihood_NBPois[pit,:]-logLikelihood_NBPois[pit,2], 2)
        diag_entries[ddit,dit]=coeffs[:,2]
#             errorbarstmp[pit]=np.fabs(1/np.sqrt(2*coeffs[pit,0]))
#         initparas=-1*np.ones(15)
#         outstruct_quad = minimize(partial_MSError_fcn, initparas, method='SLSQP', tol=1e-10,options={'ftol':1e-10 ,'disp': False,'maxiter':200})
#         print(outstruct_quad.x)
#         M=paras2M(outstruct_quad.x)
#         Cov=np.linalg.inv(M)
#         errorbars[ddit,dit]=np.sqrt(np.diag(Cov))
        
        #test
#         print(paras)
        for pit, para in enumerate(paras):
            paratmpp=deepcopy(paras)
            vec=np.linspace(para*lowfac,para+para*(1-lowfac),5)
            paradense=np.linspace(para*lowfac,para+para*(1-lowfac),100)
            fit=poly.polyval(paradense-vec[2],coeffs[pit,:])
            axtmp=axarr[pit].plot(paradense-para,coeffs[pit,2]*(paradense-vec[2])**2)
            axarr[pit].plot(vec-para,(logLikelihood_NBPois[pit,:]-logLikelihood_NBPois[pit,2]),'x',color=axtmp[-1].get_color())
#             axarr[pit].plot(vec-para,(logLikelihood_NBPois[pit,:]-logLikelihood_NBPois[pit,2]),'x',color=axtmp[-1].get_color())
#             print(-M[pit,pit]/2)
pd.DataFrame(diag_entries)

# check same data computed for hessian: (but computed for same paras so ...?

diag_entries=np.empty((len(dayvec),len(donorstrvec)),dtype=object)
fig1, axarr = pl.subplots(1, 5)
fig2,ax= pl.subplots(1, 1)
for ddit, day in enumerate(dayvec):
    for dit,donor in enumerate(donorstrvec):
        countpaircounts_d=np.load(outpath+"countpaircounts_d.npy")
        logLikelihood_NBPois=np.load('nullerror_2D/nullerrorbars_hessian_'+donor+'_'+day+'.npy')[4][:,2]*len(countpaircounts_d)
#         logLikelihood_NBPois=np.load('nullerror_diag/nullerrorbars_'+donor+'_'+day+'.npy')*len(countpaircounts_d)
        if np.argmin(logLikelihood_NBPois[1,:])!=2:
            print(str(day)+' '+str(donor)+' '+str(np.argmin(logLikelihood_NBPois[1,:])))
           
        outpath='outdata_all/'+donor+'_'+day+'_F1_'+donor+'_'+day+'_F2/min0_maxinf_v1__case0/'
        setname=outpath+'outstruct.npy'
        paras=np.load(setname).flatten()[0].x
        ax.scatter(paras[1],paras[2]) 
#         print(np.load(setname).flatten()[0])
        #format data
#         for pit, para in enumerate(paras):
#             paratmpp=deepcopy(paras)
#             for ppit,ppara in enumerate(np.linspace(para*lowfac,para+para*(1-lowfac),npoints)):
#                 paratmpp[pit]=ppara
#                 data_df=data_df.append(pd.Series({'x':paratmpp-paras,'y':(logLikelihood_NBPois[pit,ppit]-logLikelihood_NBPois[pit,2])}),ignore_index=True)
#         partial_MSError_fcn=partial(MSError_fcn,data=data_df)
        #derive initial parameters as variance of cardinal directions
        coeffs=np.zeros((len(parastmp),3))
#         errorbarstmp=np.zeros(len(parastmp))
        for pit,para in enumerate(paras): 
            vec=np.linspace(para*lowfac,para+para*(1-lowfac),npoints)
            coeffs[pit,:]=poly.polyfit(vec-vec[2], logLikelihood_NBPois[pit,:]-logLikelihood_NBPois[pit,2], 2)
        diag_entries[ddit,dit]=coeffs[:,2]
#             errorbarstmp[pit]=np.fabs(1/np.sqrt(2*coeffs[pit,0]))
#         initparas=-1*np.ones(15)
#         outstruct_quad = minimize(partial_MSError_fcn, initparas, method='SLSQP', tol=1e-10,options={'ftol':1e-10 ,'disp': False,'maxiter':200})
#         print(outstruct_quad.x)
#         M=paras2M(outstruct_quad.x)
#         Cov=np.linalg.inv(M)
#         errorbars[ddit,dit]=np.sqrt(np.diag(Cov))
        
        #test
#         print(paras)
        for pit, para in enumerate(paras):
            paratmpp=deepcopy(paras)
            vec=np.linspace(para*lowfac,para+para*(1-lowfac),5)
            paradense=np.linspace(para*lowfac,para+para*(1-lowfac),100)
            fit=poly.polyval(paradense-vec[2],coeffs[pit,:])
            axtmp=axarr[pit].plot(paradense-para,coeffs[pit,2]*(paradense-vec[2])**2)
            axarr[pit].plot(vec-para,(logLikelihood_NBPois[pit,:]-logLikelihood_NBPois[pit,2]),'x',color=axtmp[-1].get_color())
#             axarr[pit].plot(vec-para,(logLikelihood_NBPois[pit,:]-logLikelihood_NBPois[pit,2]),'x',color=axtmp[-1].get_color())
#             print(-M[pit,pit]/2)
pd.DataFrame(diag_entries)


# Then learn off diagonal elements, from 2D subspace variation 

def MSError_fcn(para,data,diag_pair):
    M=np.diag(diag_pair)
    M[0,1]=para
    M[1,0]=para
    res=0
    for val in data.itertuples():
        res+=np.power(val.y+M.dot(val.x).dot(val.x)/2,2)
    return res/len(data)


data_df;

# +
errorbars=np.empty((len(dayvec),len(donorstrvec)),dtype=object)
fig1, axarr = pl.subplots(1, 5)
lowfac=0.99
for ddit, day in enumerate(dayvec):
    for dit,donor in enumerate(donorstrvec):
        data_df=pd.DataFrame(columns=['x','y'])
        outpath='outdata_all/'+donor+'_'+day+'_F1_'+donor+'_'+day+'_F2/min0_maxinf_v1_case0_noconst_case0/'
        countpaircounts_d=np.load(outpath+"countpaircounts_d.npy")
#         logLikelihood_NBPois=np.load('nullerror_2D/nullerrorbars_hessian_'+donor+'_'+day+'.npy')*len(countpaircounts_d)
        logLikelihood_NBPois=np.load('nullerrorbars_hessian_noconst_f99_'+donor+'_'+day+'.npy')*len(countpaircounts_d)
#         outpath='outdata_all/'+donor+'_'+day+'_F1_'+donor+'_'+day+'_F2/min0_maxinf_v1__case0/'

        setname=outpath+'outstruct.npy'
        parasopt=np.load(setname).flatten()[0].x
        
        logLikelihood_NBPois_diag=np.zeros((5,5))
        for pit in range(5):
            if pit==0:
                logLikelihood_NBPois_diag[pit,:]=logLikelihood_NBPois[pit][:,2]
            else:
                logLikelihood_NBPois_diag[pit,:]=logLikelihood_NBPois[pit-1][2,:]
        
#         logLikelihood_NBPois=np.load('nullerror_diag/nullerrorbars_'+donor+'_'+day+'.npy')*len(countpaircounts_d)
        if np.argmin(logLikelihood_NBPois[1,:])!=2:
            print(str(day)+' '+str(donor)+' '+str(np.argmin(logLikelihood_NBPois[1,:])))
           
#         ax.scatter(paras[1],paras[2]) 
#         print(np.load(setname).flatten()[0])
        #format data
#         for pit, para in enumerate(paras):
#             paratmpp=deepcopy(paras)
#             for ppit,ppara in enumerate(np.linspace(para*lowfac,para+para*(1-lowfac),npoints)):
#                 paratmpp[pit]=ppara
#                 data_df=data_df.append(pd.Series({'x':paratmpp-paras,'y':(logLikelihood_NBPois[pit,ppit]-logLikelihood_NBPois[pit,2])}),ignore_index=True)
#         partial_MSError_fcn=partial(MSError_fcn,data=data_df)
        #derive initial parameters as variance of cardinal directions
        coeffs=np.zeros((5,3))
#         errorbarstmp=np.zeros(len(parastmp))
        for pit,para in enumerate(parasopt): 
            vec=np.linspace(para*lowfac,para+para*(1-lowfac),npoints)
            coeffs[pit,:]=poly.polyfit(vec-vec[2], logLikelihood_NBPois_diag[pit,:]-logLikelihood_NBPois_diag[pit,2], 2)
        diag_entries[ddit,dit]=coeffs[:,2]
        
        
        
        pair_inds=((0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4))
        npoints=5
        M=np.diag(diag_entries[ddit,dit])
        for pairit,pair_ind in enumerate(pair_inds):
            likelihoodtmp=logLikelihood_NBPois[pairit,:,:]
            diag_pair=np.array([diag_entries[ddit,dit][pair_ind[0]],diag_entries[ddit,dit][pair_ind[1]]])
            
            para1vec=np.linspace(parasopt[pair_ind[0]]*lowfac,parasopt[pair_ind[0]]+parasopt[pair_ind[0]]*(1-lowfac),npoints)
            para2vec=np.linspace(parasopt[pair_ind[1]]*lowfac,parasopt[pair_ind[1]]+parasopt[pair_ind[1]]*(1-lowfac),npoints)
            for pit1,para1 in enumerate(para1vec):
                for pit2,para2 in enumerate(para2vec):
                    data_df=data_df.append(pd.Series({'x':np.array([para1-para1vec[2],para2-para2vec[2]]),'y':(likelihoodtmp[pit1,pit2]-likelihoodtmp[2,2])}),ignore_index=True)
            partial_MSError_fcn=partial(MSError_fcn,data=data_df,diag_pair=diag_pair)
            initparas=1.
            outstruct_quad = minimize(partial_MSError_fcn, initparas, method='Nelder-Mead', tol=1e-10,options={'ftol':1e-10 ,'disp': False,'maxiter':200})
#             outstruct_quad = minimize(partial_MSError_fcn, initparas, method='SLSQP', tol=1e-10,options={'ftol':1e-10 ,'disp': False,'maxiter':200})

            Mtmp=np.diag(diag_pair)
            Mtmp[0,1]=outstruct_quad.x
            Mtmp[1,0]=outstruct_quad.x
#             for pit1,para1 in enumerate(para1vec):
            pit1=2
            para1=para1vec[2]
            axtmp=axarr[pair_ind[1]].plot(para2vec-para2vec[2],likelihoodtmp[pit1,:]-likelihoodtmp[2,2],'x')
#             axtmp=axarr[pair_ind[1]].plot(para2vec,likelihoodtmp[pit1,:]-likelihoodtmp[2,2],'x')
            fval=np.zeros(50)
            para2vec_dense=np.linspace(parasopt[pair_ind[1]]*lowfac,parasopt[pair_ind[1]]+parasopt[pair_ind[1]]*(1-lowfac),50)
            for pit2,para2 in enumerate(para2vec_dense):
                x=np.array([0,para2-para2vec[2]])
                fval[pit2]=Mtmp.dot(x).dot(x)/2
#             axarr[pair_ind[1]].plot(para2vec_dense,fval,color=axtmp[-1].get_color())
            axarr[pair_ind[1]].plot(para2vec_dense-para2vec[2],fval,color=axtmp[-1].get_color())

            pit2=2
            para2=para2vec[2]
            axtmp=axarr[pair_ind[0]].plot(para1vec-para1vec[2],likelihoodtmp[:,pit2]-likelihoodtmp[2,2],'x')
#             axtmp=axarr[pair_ind[0]].plot(para1vec,likelihoodtmp[:,pit2]-likelihoodtmp[2,2],'x')
            fval=np.zeros(50)
            para1vec_dense=np.linspace(parasopt[pair_ind[0]]*lowfac,parasopt[pair_ind[0]]+parasopt[pair_ind[0]]*(1-lowfac),50)
            for pit1,para1 in enumerate(para1vec_dense):
                x=np.array([para1-para1vec[2],0])
                fval[pit1]=Mtmp.dot(x).dot(x)/2
            axarr[pair_ind[0]].plot(para1vec_dense-para1vec[2],fval,color=axtmp[-1].get_color())
#             axarr[pair_ind[0]].plot(para1vec_dense,fval,color=axtmp[-1].get_color())

            M[pair_ind[0],pair_ind[1]]=outstruct_quad.x
            M[pair_ind[1],pair_ind[0]]=outstruct_quad.x
#         print(pd.DataFrame(M))
        eig=np.linalg.eig(M)
        print(eig[0])
        Cov=np.linalg.inv(M)
#         print(pd.DataFrame(Cov))
        errorbars[ddit,dit]=np.sqrt(np.diag(Cov))
        
#         #test
# #         print(paras)
#         for pit, para in enumerate(paras):
#             paratmpp=deepcopy(paras)
#             paradense=np.linspace(para*lowfac,para+para*(1-lowfac),50)
#             fval=np.zeros(paradense.shape)
#             for ppit,ppara in enumerate(paradense):
# #                 paratmpp[pit]=ppara
# #                 paratmpp-=paras
# #                 print(paratmpp)
#                 fval[ppit]=-M[pit,pit]/2*(ppara-para)**2
#             axarr[pit].plot(paradense-para,fval)
#             axarr[pit].plot(np.linspace(para*lowfac,para+para*(1-lowfac),5)-para,(logLikelihood_NBPois[pit,:]-logLikelihood_NBPois[pit,2]),'x')
# #             print(-M[pit,pit]/2)
# errorbars
# -

# Take inverse to get covariance:

pd.DataFrame(M)

pd.DataFrame(np.linalg.inv(M))

# test by plotting

# +
lowrange=(-2.4 ,1,0.8,6.5,-12.5)
highrange=(-1.9,5,1.4,7.1,-7.5)
offset=[-0.25,-0.15,-0.05,0.05,0.15,0.25]
parastrvec=(r'$\alpha_{\rho}$',r'$a$',r'$\gamma$',r'$\log M$','$\logf_{min}$')
fig,ax=pl.subplots(5,1,figsize=(3,8*5/6))
for pit in range(len(outstructs[:,0])):
    ax[pit].set_ylim(lowrange[pit],highrange[pit])
    ax[pit].set_ylabel(parastrvec[pit])
    ax[pit].locator_params(nbins=4)
    ax[pit].set_xticklabels([])
    ax[pit].yaxis.grid()
    ax[pit].set_xlim(0+offset[0],4+offset[-1])

for pit in range(len(outstructs[:,0])):
    for dit,donor in enumerate(donorstrvec):
#         try:
        ax[pit].errorbar(np.arange(len(dayvec))+offset[dit],[struct[pit] for struct in outstructs[:,dit]],yerr=[ebar[pit] for ebar in errorbars[:,dit]],fmt='.',label=donor,zorder=10,clip_on=False,mew=1)
#         except TypeError:
#             print(donor)
ax[-1].set_xticks(range(len(dayvec)))
ax[-1].set_xlabel('day')
ax[-1].set_xticklabels(dayvec)
ax[2].legend(frameon=False,bbox_to_anchor=(1.3, 1.0))
fig.savefig('sameday_nullmodel_learned_paras_time_series.pdf',format='pdf',dpi=1000, bbox_inches='tight')
# -

import numpy.polynomial.polynomial as poly
casestrvec=(r'$NB\rightarrow Pois$')
casevec=[0]
donorstrvec=('S1','P2',  'Q2', 'Q1', 'S2','P1')
# donorstrvec=('S1','P2',  'Q2', 'S2','P1','S2')
repvec=(('F1','F1'),('F1','F2'),('F2','F1'),('F2','F2'))
dayvec=np.array(['pre0','0','7','15','45'])
dayref='0'
day='pre0'
outstructs=np.empty((len(repvec),len(donorstrvec)),dtype=object)
# for ddit, day in enumerate(dayvec):
for rit,rep_pair in enumerate(repvec):    
    for dit,donor in enumerate(donorstrvec):
        try:
#             runstr='outdata_all/'+donor+'_'+day+'_F1_'+donor+'_'+day+'_F2/min0_maxinf_v1__case0/'
            runstr='outdata_all/'+donor+'_'+dayref+'_'+rep_pair[0]+'_'+donor+'_'+day+'_'+rep_pair[1]+'/min0_maxinf_v1_case0_case0/'
            setname=runstr+'outstruct.npy'
            parastmp=np.load(setname).flatten()[0].x
            outstructs[rit,dit]=np.load(setname).flatten()[0].x
#             nfbins=800
#             rhofvec,fvec = get_rhof(outstructs[ddit,dit][0],nfbins,np.power(10,outstructs[ddit,dit][4]))
#             integ=np.exp(np.log(rhofvec)+2*np.log(fvec))
#             N_total[ddit,dit]=1./np.dot(np.diff(np.log(fvec))/2,integ[1:]+integ[:-1])
#             lowfac=0.95
#             npoints=5
#             logLikelihood_NBPois=np.load('nullerrorbars_'+donor+'_'+day+'.npy')
#             coeffs=np.zeros((len(parastmp),3))
#             errorbarstmp=np.zeros(len(parastmp))
#             for pit,para in enumerate(parastmp): 
#                 vec=np.linspace(para*lowfac,para+para*(1-lowfac),npoints)
#                 coeffs[pit,:]=poly.polyfit(vec-vec[2], logLikelihood_NBPois[pit,:], 2)
#                 errorbarstmp[pit]=np.fabs(1/np.sqrt(2*coeffs[pit,0]))
#             errorbars[ddit,dit]=errorbarstmp
        except IOError:
            outstructs[rit,dit]=np.zeros(5)
            print(day+' '+donor)

# 0-Pre0 values

# +
lowrange=(-2.4 ,0.5,1.0,6.4,-12.5)
highrange=(-2.0,5,1.9,6.9,-7.5)
offset=[-0.25,-0.15,-0.05,0.05,0.15,0.25]
parastrvec=(r'$\alpha_{\rho}$',r'$a$',r'$\gamma$',r'$\log M$','$\logf_{min}$')
fig,ax=pl.subplots(5,1,figsize=(3,8*5/6))
for pit in range(len(outstructs[0,0])):
    ax[pit].set_ylim(lowrange[pit],highrange[pit])
    ax[pit].set_ylabel(parastrvec[pit])
    ax[pit].locator_params(nbins=4)
    ax[pit].set_xticklabels([])
    ax[pit].yaxis.grid()
    ax[pit].set_xlim(0+offset[0],4+offset[-1])

for pit in range(len(outstructs[0,0])):
    for dit,donor in enumerate(donorstrvec):
#         try:
        ax[pit].plot(np.arange(len(repvec))+offset[dit],[struct[pit] for struct in outstructs[:,dit]],'.',label=donor,zorder=10,clip_on=False,mew=1)
#         except TypeError:
#             print(donor)
ax[-1].set_xticks(range(len(repvec)))
ax[-1].set_xlabel('day')
ax[-1].set_xticklabels([''.join(rep) for rep in repvec])
ax[2].legend(frameon=False,bbox_to_anchor=(1.3, 1.0))
# fig.savefig('sameday_nullmodel_learned_paras_time_series.pdf',format='pdf',dpi=1000, bbox_inches='tight')
# -

partial_MSError_fcn=partial(MSError_fcn,data=data_df)
initparas=np.ones(15)
outstruct_quad = minimize(partial_MSError_fcn, initparas, method='SLSQP', tol=1e-10,options={'ftol':1e-6 ,'disp': True,'maxiter':100})
M=paras2M(outstruct_quad.x)
Cov=np.linalg.inv(M)
var=np.diag(Cov)
var

fig1, axarr = pl.subplots(1, 5)
for ddit, day in enumerate(dayvec):
    for dit,donor in enumerate(donorstrvec):
        logLikelihood_NBPois=np.load('nullerrorbars_'+donor+'_'+day+'.npy')
        outpath='outdata_all/'+donor+'_'+day+'_F1_'+donor+'_'+day+'_F2/min0_maxinf_v1__case0/'
        setname=outpath+'outstruct.npy'
        parastmp=np.load(setname).flatten()[0].x
        coeffs=np.zeros((len(parastmp),3))
        for pit,para in enumerate(parastmp): 
            vec=np.linspace(para*lowfac,para+para*(1-lowfac),npoints)
            coeffs[pit,:]=poly.polyfit(vec-vec[3], likelihood[pit,:], 2)
            vecdense=np.linspace(vec[0],vec[-1],100)
            fit=poly.polyval(vecdense-vec[2],coeffs[pit,:])
            axarr[pit].plot(vecdense-vec[2],fit-logLikelihood_NBPois[pit,2])
            axarr[pit].plot(vec-vec[2], logLikelihood_NBPois[pit,:]-logLikelihood_NBPois[pit,2],'x')
#             print(1/np.sqrt((2*np.fabs(coeffs[pit,0]))))
            print(coeffs[pit,0])

fig1, axarr = pl.subplots(1, 5)
strvec=(r'\alpha_{\rho}',r'\beta_m',r'\alpha_m',r'\log_{10}f_{min}')
linstylist=('-o','-v','-^','-s')
rcParams['figure.figsize']=(20,5)
rcParams['font.size']= 24
pl.rcParams.update({'font.size': 24})
for tit, likelihood in enumerate(likelihoods):
    parastmp=paras_store2[tit].x
    coeffs=np.zeros((len(paras),3))
    for pit,para in enumerate(parastmp):
        ptmp=np.asarray([outstruct.x[pit] for outstruct in paras_store2])
        mcolor='r' if (np.log10(np.fabs((para-paras[pit])/paras[pit]))>-2.5) else 'b'
        vec=np.linspace(para*lowfac,para+para*(1-lowfac),npoints)
        coeffs[pit,:]=poly.polyfit(vec-vec[3], likelihood[pit,:], 2)
        axarr[pit].plot(vec,likelihood[pit,:],linstylist[pit][1],label='data',Markersize=10)
        axarr[pit].set_ylabel(r'$\log \mathcal{L}$')
        axarr[pit].set_xlabel(r'$'+strvec[pit]+' $')
        axarr[pit].locator_params(nbins=4)
        vecdense=np.linspace(vec[0],vec[-1],100)
        axarr[pit].plot(vecdense,poly.polyval(vecdense-vec[3],coeffs[pit,:]))

fig1, axarr = pl.subplots(1, 5)
for ddit, day in enumerate(dayvec):
#     print(day)
    for dit,donor in enumerate(donorstrvec):
        logLikelihood_NBPois=np.load('nullerror_diag/nullerrorbars_'+donor+'_'+day+'.npy')
        outpath='outdata_all/'+donor+'_'+day+'_F1_'+donor+'_'+day+'_F2/min0_maxinf_v1__case0/'
        setname=outpath+'outstruct.npy'
        parastmp=np.load(setname).flatten()[0].x
        coeffs=np.zeros((len(parastmp),3))
        for pit,para in enumerate(parastmp): 
#             if pit>2:
#                 vec=np.power(10,np.linspace(para*lowfac,para+para*(1-lowfac),npoints))
#             else:
            vec=np.linspace(para*lowfac,para+para*(1-lowfac),npoints)
            coeffs[pit,:]=poly.polyfit(vec-vec[2], logLikelihood_NBPois[pit,:], 2)
            vecdense=np.linspace(vec[0],vec[-1],100)
            fit=poly.polyval(vecdense-vec[2],coeffs[pit,:])
            axarr[pit].plot(vecdense-vec[2],fit-logLikelihood_NBPois[pit,2])
            axarr[pit].plot(vec-vec[2],logLikelihood_NBPois[pit,:]-logLikelihood_NBPois[pit,2],'x')
#             print(1/np.sqrt((2*np.fabs(coeffs[pit,0]))))
            print(coeffs[pit,0])

# +
fig,ax=pl.subplots(1,1,figsize=(5,5))

for dit,donor in enumerate(donorstrvec):
    ax.plot([struct[1] for struct in outstructs[:,dit]],[struct[2] for struct in outstructs[:,dit]],'.',label=donor)

# +
# lowrange=(-2.3,1,1.0,np.power(10,6.5),np.power(10,-11))
# lowrange=(-2.3,1,1.0,1e6,1e-12)
# highrange=(-2.0,5,1.3,1e7,1e-8)
# lowrange=(-3.  ,1,0.5,6,-13)
# highrange=(-1.0,5,2.0,8,-8)
lowrange=(-2.4 ,1,0.8,6.5,-12.5)
highrange=(-2.0,5,1.4,7,-7.5)
# lowrange=(-3.0,1,0.5,1e6,1e-12)
# highrange=(-1.4,5,2.0,1e7,1e-8)

parastrvec=(r'$\alpha_{\rho}$',r'$a$',r'$\gamma$',r'$\log M$','$\logf_{min}$')

fig,ax=pl.subplots(5,1,figsize=(3,8*5/6))
for pit in range(len(outstructs[:,0])):
    ax[pit].set_ylim(lowrange[pit],highrange[pit])
    ax[pit].set_ylabel(parastrvec[pit])
#     if pit>3:
#         ax[pit].set_yscale('log')
#     else:
    ax[pit].locator_params(nbins=4)
    ax[pit].set_xticklabels([])
    ax[pit].yaxis.grid()
    ax[pit].set_xlim(0+offset[0],4+offset[-1])
offset=[-0.25,-0.15,-0.05,0.05,0.15,0.25]
for pit in range(len(outstructs[:,0])):
    for dit,donor in enumerate(donorstrvec):
#         if pit>2:
#             ax[pit].plot(range(len(dayvec)),[np.power(10,struct[pit]) for struct in outstructs[:,dit]],'.',label=donor,zorder=10,clip_on=False)
#             ax[pit].errorbar(range(len(dayvec)),[np.power(10,struct[pit]) for struct in outstructs[:,dit]],yerr=[ebar[pit] for ebar in errorbars[:,dit]],fmt='.',label=donor,zorder=10,clip_on=False)

            #             if pit==3:
#                 ax[pit].set_yticks((lowrange[pit],highrange[pit])) 

#             ax[pit].set_ylim(1e-11,1e-5)
#         else:
#             ax[pit].plot(range(len(dayvec)),[struct[pit] for struct in outstructs[:,dit]],'.',label=donor,zorder=10,clip_on=False)
        ax[pit].errorbar(np.arange(len(dayvec))+offset[dit],[struct[pit] for struct in outstructs[:,dit]],yerr=[ebar[pit] for ebar in errorbars[:,dit]],fmt='.',label=donor,zorder=10,clip_on=False,mew=1)
#         print(outstructs[:,dit])

            # for dit,donor in enumerate(donorstrvec):
#     ax[5].plot(range(len(dayvec)),np.log10(N_total[:,dit]),'.',label=donor,clip_on=False)
# ax[5].set_ylim(7,11)
# ax[5].set_ylabel(r'$\log_{10}N=\log_{10}\frac{1}{\langle f\rangle}$')
# ax[5].locator_params(nbins=4)
# ax[5].yaxis.grid()
ax[-1].set_xticks(range(len(dayvec)))
ax[-1].set_xlabel('day')
ax[-1].set_xticklabels(dayvec)
ax[2].legend(frameon=False,bbox_to_anchor=(1.3, 1.0))

# fig.savefig('sameday_nullmodel_learned_paras_time_series.pdf',format='pdf',dpi=1000, bbox_inches='tight')
# -

# Diversity estimates

# +
# lowrange=(-2.3,1,1.0,np.power(10,6.5),np.power(10,-11))
# lowrange=(-2.3,1,1.0,1e6,1e-12)
# highrange=(-2.0,5,1.3,1e7,1e-8)
lowrange=(-3.  ,1,0.5,6,-13)
highrange=(-1.0,5,2.0,8,-8)
# lowrange=(-3.0,1,0.5,1e6,1e-12)
# highrange=(-1.4,5,2.0,1e7,1e-8)

parastrvec=(r'$\alpha_{\rho}$',r'$a$',r'$\gamma$',r'$\log M$','$\logf_{min}$')
donorstrvec=('S1','P2',  'Q2', 'Q1', 'S2','P1')
donorstrvec=('S1','P2',  'Q2', 'S2','P1')

# donorstrvec=('S1','P2',  'Q2', 'S2','P1','S2')
dayvec=np.array(['pre0','0','7','15','45'])
fig,ax=pl.subplots(1,1,figsize=(3,2))
# for pit in range(len(outstructs[:,0])):
#     ax[pit].set_ylim(lowrange[pit],highrange[pit])
#     ax[pit].set_ylabel(parastrvec[pit])
#     if pit>3:
#         ax[pit].set_yscale('log')
#     else:
#     ax[pit].locator_params(nbins=4)
#     ax[pit].set_xticklabels([])
ax.yaxis.grid()
#     ax[pit].set_xlim(0+offset[0],4+offset[-1])
offset=np.array([-0.25,-0.15,-0.05,0.05,0.15,0.25])
# for pit in range(len(outstructs[:,0])):
div_bet0=np.zeros((len(donorstrvec),len(dayvec)))
div_bet1=np.zeros(div_bet0.shape)
div_bet2=np.zeros(div_bet0.shape)
Ntotal=np.zeros((len(donorstrvec),len(dayvec)))
N1total=np.zeros((len(donorstrvec),len(dayvec)))
N2total=np.zeros((len(donorstrvec),len(dayvec)))
for dit,donor in enumerate(donorstrvec):
    nfbins=800
    for ddit,day in enumerate(dayvec):
        print(day)
        outpath='outdata_all/'+donor+'_'+day+'_F1_'+donor+'_'+day+'_F2/min0_maxinf_v1__case0/'
        setname=outpath+'outstruct.npy'
        try:
            paras=np.load(setname).flatten()[0].x
        except IOError:
            print(day +' '+donor)
        print(paras)
        NreadsI_d=np.load(outpath+"NreadsI_d.npy")
        NreadsII_d=np.load(outpath+"NreadsII_d.npy")
        Nclones=np.sum(np.load(outpath+"countpaircounts_d.npy"))
        uni1=np.load(outpath+'unicountvals_1_d.npy')
        uni2=np.load(outpath+'unicountvals_2_d.npy')
        indn1=np.load(outpath+'indn1_d.npy')
        indn2=np.load(outpath+'indn2_d.npy')
        countpairs=np.load(outpath+'countpaircounts_d.npy')
        N_1=np.sum((uni1[indn1]>0)*countpairs)
        N_2=np.sum((uni2[indn2]>0)*countpairs)
        N_all=np.sum(countpairs)
        print(str(N_all)+' '+str(N_1)+' '+str(N_2))
        
        
        fmax=1e0
        logfvec=np.linspace(paras[-1],np.log10(fmax),nfbins)
        fvec=np.power(10,logfvec)
        logfvec=np.log(fvec)
        rhovec=np.power(fvec,paras[0])
        integ=np.exp(np.log(rhovec)+logfvec)
        Z=np.dot(np.diff(logfvec)/2.,integ[1:]+integ[:-1])
        logrhofvec=paras[0]*logfvec-np.log(Z)
        
#         integ= np.exp(logrhofvec+ 2*logfvec)     #richness 
#         div_bet0[ddit,dit]=np.dot(np.diff(logfvec)/2,integ[1:]+integ[:-1])
        
        st=time.time()
        repfac=NreadsII_d/NreadsI_d #=1
        case=0
        alpha_rho=paras[0] 
        if case<2:
            m_total=np.power(10,paras[3])
            r_c1=NreadsI_d/m_total 
            r_c2=repfac*r_c1     
            r_cvec=(r_c1,r_c2)
            fmin=np.power(10,paras[4])
        else:
            fmin=np.power(10,paras[3])
        beta_mv= paras[1]
        alpha_mv=paras[2]

        m_max=1e3 #1e3 conditioned on n=0
        r_cvec=(r_c1,r_c2)
        Nreadsvec=(NreadsI_d,NreadsII_d)
        for it in range(2):
            Pn_f=np.empty((len(fvec),),dtype=object)
            if case==3:
                m1vec=Nreads*fvec
                for find,m1 in enumerate(m1vec):
                    Pn_f[find]=poisson(m1)
                logPn0_f=-m1vec
            elif case==2:
                m1=Nreadsvec[it]*fvec
                v1=m1+beta_mv*np.power(m1,alpha_mv)
                p=1-m1/v1
                n=m1*m1/v1/p
                for find,(n,p) in enumerate(zip(n,p)):
                    Pn_f[find]=nbinom(n,1-p)
                Pn0_f=np.asarray([Pn_find.pmf(0) for Pn1_find in Pn1_f])
                logPn0_f=np.log(Pn0_f)
            elif case==1:
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
            else: #case=0
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
            if it==0:
                Pn1_f=Pn_f
                logPn10_f=logPn0_f
            else:
                Pn2_f=Pn_f
                logPn20_f=logPn0_f

        integ=np.exp(logrhofvec+logPn20_f+logfvec)
        Pn10=np.dot(np.diff(logfvec)/2,integ[1:]+integ[:-1])
        
        integ=np.exp(logrhofvec+logPn10_f+logfvec)
        Pn20=np.dot(np.diff(logfvec)/2,integ[1:]+integ[:-1])
        
        integ=np.exp(logrhofvec+logPn10_f+logPn20_f+logfvec)
        Pn0n0=np.dot(np.diff(logfvec)/2,integ[1:]+integ[:-1])
        
        #         print(Pn0n0)
        
        N1total[dit,ddit]=N_1/(1-Pn10)
        N2total[dit,ddit]=N_2/(1-Pn20)
        
        Ntotal[dit,ddit]=Nclones/(1-Pn0n0)
        
        div_bet0[dit,ddit]=(N1total[dit,ddit]+N2total[dit,ddit])/2
        integ= -logfvec*np.exp(logrhofvec+ 2*logfvec)   #shannon    
#         integ= -logfvec*(fvec**2)*np.exp(logrhofvec)   #shannon    
        div_bet1[dit,ddit]=np.exp(div_bet0[dit,ddit]*np.dot(np.diff(logfvec)/2,integ[1:]+integ[:-1]))

        integ=np.exp(logrhofvec+3*logfvec)     #simpson
        div_bet2[dit,ddit]=1          /(div_bet0[dit,ddit]*np.dot(np.diff(logfvec)/2,integ[1:]+integ[:-1]))
        
        
        print(str(div_bet0[dit,ddit])+' '+str(div_bet1[dit,ddit])+' '+str(div_bet2[dit,ddit]))
#     for pit,div in enumerate((div_bet1,div_bet2)):
#         ax[pit].locator_params(nbins=4)


# +
#     ax[pit].set_xlim(0+offset[0],4+offset[-1])
offset=np.array([-0.25,-0.15,-0.05,0.05,0.15,0.25])
fig,ax=pl.subplots(1,1,figsize=(3,2))
ax.yaxis.grid()
ax.set_xlim(0+offset[0],4+offset[-1])
seaborn.set_style("whitegrid", {'axes.grid' : False})
for dit,donor in enumerate(donorstrvec):
    h=ax.plot(np.arange(len(dayvec))+offset[dit],div_bet2[dit,:],'o',mec=None,mew=0,ms=5,label=donor,zorder=10,clip_on=False)
    ax.plot(np.arange(len(dayvec))+offset[dit],div_bet1[dit,:],'s',mec=None,mew=0,ms=5,color=h[-1].get_color(),zorder=10,clip_on=False)
    ax.plot(np.arange(len(dayvec))+offset[dit],div_bet0[dit,:],'v',mec=None,mew=0,ms=5,color=h[-1].get_color(),zorder=10,clip_on=False)

#     for ddit,day in enumerate(dayvec):
#         ax.plot([ddit+offset[dit],ddit+offset[dit]],[div_bet1[ddit,dit],div_bet2[ddit,dit]],'-',linewidth=1,color=h[-1].get_color(),zorder=10,clip_on=False)
#         ax.plot(np.arange(len(dayvec))+offset[dit],div_bet2[:,dit],divbet1[:,dit]),'-',label=donor,zorder=10,clip_on=False)
    ax.set_xticks(range(len(dayvec)))
    ax.set_xticks(range(len(dayvec)))
    ax.set_xlabel('day')
    ax.set_yscale('log')

    ax.set_xticklabels(dayvec)
    ax.set_yticks([1e2,1e4,1e6,1e8,1e10])
#     ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
# ax[1].set_yticklabels([])

# ax[0].set_ylim(lowrange[4],highrange[pit])
ax.set_ylabel(r'$\exp\left[H_\beta\right]$')
# ax[0].set_title(r'$\beta=1$ (Shannon)')
# ax[1].set_title(r'$\beta=2$ (Simpsons)')
        
#         if pit>2:
#             ax[pit].plot(range(len(dayvec)),[np.power(10,struct[pit]) for struct in outstructs[:,dit]],'.',label=donor,zorder=10,clip_on=False)
#             ax[pit].errorbar(range(len(dayvec)),[np.power(10,struct[pit]) for struct in outstructs[:,dit]],yerr=[ebar[pit] for ebar in errorbars[:,dit]],fmt='.',label=donor,zorder=10,clip_on=False)

        #             if pit==3:
#                 ax[pit].set_yticks((lowrange[pit],highrange[pit])) 

#             ax[pit].set_ylim(1e-11,1e-5)
#         else:
#             ax[pit].plot(range(len(dayvec)),[struct[pit] fiior struct in outstructs[:,dit]],'.',label=donor,zorder=10,clip_on=False)
# pit=0
# ax[pit].errorbar(np.arange(len(dayvec))+offset[dit],[struct[pit] for struct in outstructs[:,dit]],yerr=[ebar[pit] for ebar in errorbars[:,dit]],fmt='.',label=donor,zorder=10,clip_on=False)
#         print(outstructs[:,dit])

            # for dit,donor in enumerate(donorstrvec):
#     ax[5].plot(range(len(dayvec)),np.log10(N_total[:,dit]),'.',label=donor,clip_on=False)
# ax[5].set_ylim(7,11)
# ax[5].set_ylabel(r'$\log_{10}N=\log_{10}\frac{1}{\langle f\rangle}$')
# ax[5].locator_params(nbins=4)
# ax[5].yaxis.grid()

ax.legend(frameon=False,bbox_to_anchor=(1.3, 1.0))

fig.savefig('fig4_div_estimates.pdf',format='pdf',dpi=1000, bbox_inches='tight')

# +
casestrvec=(r'$NB\rightarrow Pois$')
casevec=[0]
donorstrvec=('S1','P2',  'Q2', 'S2','P1','S2')
dayvec=np.array(['pre0','0','7','15','45'])
outstructs=[]
for ddit, day in enumerate(dayvec):
    for dit,donor in enumerate(donorstrvec):
        try:
            runstr='outdata_all/'+donor+'_'+day+'_F1_'+donor+'_'+day+'_F2/min0_maxinf_v1__case0/'
            setname=runstr+'outstruct.npy'
            outstructs.append(np.load(setname).flatten()[0].x)
        except IOError:
            outstructs.append(np.zeros(5))
            print(day+' '+donor)
            
lowrange=(-2.3,1,1.0,6.5,-11)
highrange=(-1.9,5,1.3,7.0,-8)
lowrange=(-2.5,0,0.9,6.5,-13)
highrange=(-1.8,6,1.5,7.1,-7)
lowrange=(-2.25,1,0.9,6.5,-12)
highrange=(-1.95,5,1.5,7,-7.5)

fig,ax=pl.subplots(1,5,figsize=(20,4))
for pit in range(len(outstructs[0])):
    data=[struct[pit] for struct in outstructs]
#         data=[struct[pit] for struct in outstructs[:,dit]]
    ax[pit].hist(data,bins=np.linspace(lowrange[pit],highrange[pit],15))
    ax[pit].set_xlim(lowrange[pit],highrange[pit])
    ax[pit].set_ylim(0,18)
parastrvec=(r'$\alpha$',r'$\log_{10}a$',r'$\gamma$','$\log_{10}M$','$\log_{10}f_{min}$')
for pit in range(5):
    ax[pit].set_xlabel(parastrvec[pit],size=24)

fig.savefig('sameday_nullmodel_learned_paras.pdf',format='pdf',dpi=1000, bbox_inches='tight')

# +
lowrange=(-2.3,1,1.0,6.5,-11)
highrange=(-2.0,4,1.3,7.0,-8)

for ddit, day in enumerate(dayvec):
# for dit,donor in enumerate(donorstrvec):
    fig,ax=pl.subplots(1,5,figsize=(20,4))
    for pit in range(len(outstructs[0,0])):
        data=[struct[pit] for struct in outstructs[ddit,:]]
#         data=[struct[pit] for struct in outstructs[:,dit]]
        ax[pit].hist(data,bins=np.linspace(lowrange[pit],highrange[pit],10))
        ax[pit].set_xlim(lowrange[pit],highrange[pit])
        ax[pit].set_ylim(0,13)
    parastrvec=(r'$\alpha$',r'$\log_{10}a$',r'$\gamma$','$\log_{10}M$','$\log_{10}f_{min}$')
    for pit in range(5):
        ax[pit].set_xlabel(parastrvec[pit],size=24)

fig.savefig('sameday_nullmodel_learned_paras.pdf',format='pdf',dpi=1000, bbox_inches='tight')
