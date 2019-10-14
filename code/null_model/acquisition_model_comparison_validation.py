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

# %matplotlib inline
import sys,os
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
# %run -i '../lib/utils/ipynb_setup.py'
from lib.utils.plotting import plot_n1_vs_n2,add_ticks
from lib.proc import get_sparserep,import_data
from lib.model import get_Pn1n2_s, get_rhof, NegBinParMtr,get_logPn_f,get_model_sample_obs
from lib.learning import nullmodel_constr_fn,callback,learn_null_model
import lib.learning
# %load_ext autoreload
# %autoreload 2

# +
import matplotlib.pyplot as pl
paper=True
if not paper:
    pl.rc("figure", facecolor="gray",figsize = (8,8))
    pl.rc('lines',markeredgewidth = 2)
    pl.rc('font',size = 24)
else:
    pl.rc("figure", facecolor="none",figsize = (3.5,3.5))
    pl.rc('lines',markeredgewidth = 0)
    pl.rc('font',size = 10)
    
pl.rc('text', usetex=True)

import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : True})
params= {'text.latex.preamble' : [r'\usepackage{amsmath}']}
pl.rcParams.update(params)
# -

# Data

# +
parvernull = 'v4'
rootpath = '../../../'
path = rootpath + 'output/'
daynull="0"#"pre0"
day="0"
donor='S2'
dataset_pair=(donor+'_'+daynull+'_F1', donor+'_'+day+'_F2')
colnames = [u'Clone fraction',u'Clone count',u'N. Seq. CDR3',u'AA. Seq. CDR3'] 

datarootpath=rootpath+'data/Yellow_fever/prepostvaccine/' 
mincount=0
maxcount=np.Inf
case=0
runstr = 'null_pair_v1_ct_1_mt_'+str(case)+'_min' + str(mincount) + '_max' + str(maxcount)
outpath = path + dataset_pair[0] + '_' + dataset_pair[1] + '/' + runstr + '/'
initparas=np.load(path + donor+'_0_F1_'+donor+'_0_F2/' + runstr + '/'+'optparas.npy')
Nclones_samp,subset=import_data(datarootpath,dataset_pair[0]+'_.txt',dataset_pair[1]+'_.txt',mincount,maxcount,colnames,colnames)

#transform to sparse representation
sparse_rep=get_sparserep(subset.loc[:,['Clone_count_1','Clone_count_2']])
indn1_d,indn2_d,countpaircounts_d,unicountvals_1_d,unicountvals_2_d,NreadsI_d,NreadsII_d=sparse_rep       
Nsamp=np.sum(countpaircounts_d)


# -

def get_Pn1_n2(indn1,indn2,unicountvals_1,unicountvals_2,clonecountpair_counts,NreadsI):
    Pn1n2=np.zeros((len(unicountvals_1),len(unicountvals_2)))
    clonesum=np.sum(clonecountpair_counts)
    for it,val in enumerate(clonecountpair_counts):
        Pn1n2[indn1[it],indn2[it]]=float(clonecountpair_counts[it])/clonesum#NreadsI
    Pn2_n1=np.zeros((len(unicountvals_2),len(unicountvals_1)))
    Pn1=np.zeros((len(unicountvals_1),))
    Pn2=np.zeros((len(unicountvals_2),))
    for n1it,n1 in enumerate(unicountvals_1):
        Pn1[n1it]=np.nansum(Pn1n2[n1it,:])
        if (Pn1[n1it]!=0):
            Pn2_n1[:,n1it]=Pn1n2[n1it,:]/Pn1[n1it]  
    for n2it,n2 in enumerate(unicountvals_2):
        Pn2[n2it]=np.nansum(Pn1n2[:,n2it])
    return Pn2_n1,Pn1,Pn2


Pn2_n1_d,Pn1_d,Pn2_d=get_Pn1_n2(indn1_d,indn2_d,unicountvals_1_d,unicountvals_2_d,countpaircounts_d,NreadsI_d)

# Model

# +
# casestrvec=(r'$NB\rightarrow Pois$',r'$Pois \rightarrow NB$','$NB$','$Pois$')
casestrvec=(r'$NB\rightarrow Pois$','$NB$','$Pois$')
casevec=[0,2,3]
donorvec=['S2']#['P1','P2','Q1','Q2','S1','S2']
# donorvec=("Azh","KB","Yzh","GS","Kar","Luci")
dayvec=range(5)
nparasvec=(4,3,1)
outstructs=np.empty((len(casevec),len(donorvec),len(dayvec)),dtype=dict)
out_df=pd.DataFrame()
#for cit, case in enumerate(casevec):
acq_model_type=2
constr_type=1
#for dit,donor in enumerate(donorvec):
donors='S2'         
#for ddit,day in enumerate(dayvec):
day=0

parvernull = 'v4_ct_'+str(constr_type)+'_mt_'+str(acq_model_type)
data_name=donor+'_'+str(day)+"_F1_"+donor+'_'+str(day)+'_F2'
#             runstr='../../../output/'+dataname+'/null_pair_v1_null_ct_1_acq_model_type'+str(case)+'_min0_maxinf/'
runstr='../../../output/'+data_name+'/null_pair_'+parvernull+'_min0_maxinf/'
setname=runstr+'outstruct.npy'
opt_null_paras=np.load(setname).flatten()[0].x
opt_null_paras
#             outstructs[cit,dit,ddit]=np.load(setname).flatten()[0]
#             tmpdict=np.load(setname).flatten()[0]
#             tmpdict['day']=day
#             tmpdict['donor']=donor
#             tmpdict['mt']=case
#             out_df=out_df.append(tmpdict,ignore_index=True)
# -

Pn1n2_t,unicountvals_1_t,unicountvals_2_t,logPn1_f_t,logPn2_f_t,logfvec=get_Pn1n2_s(opt_null_paras, 0, sparse_rep,acq_model_type)

##compute conditional
Pn2_n1_t=np.zeros((len(unicountvals_2_t),len(unicountvals_1_t)))
Pn1_t=np.zeros((len(unicountvals_1_t,)))
for n1it,n1 in enumerate(unicountvals_1_t):
    Pn1_t[n1it]=np.nansum(Pn1n2_t[n1it,:])
    if Pn1_t[n1it]!=0:
        Pn2_n1_t[:,n1it]=Pn1n2_t[n1it,:]/Pn1_t[n1it]

# +
colors = pl.cm.inferno(np.linspace(0.1, 1, 11))
fig1, ax1 = pl.subplots(1, 2,figsize=(3.5,2.))
datmkr='.'
simmkr='.'
xlimmax=4
for it,nII in enumerate(range(11)):
    ax1[0].plot(unicountvals_2_d, Pn2_n1_d[:,2*nII], datmkr, label='data',color=colors[it])#,mfc='w')    
    if it==0:
        ax1[0].plot(unicountvals_2_t[1:], Pn2_n1_t[1:,2*nII], '-', label='model',color=colors[it])
    else:
        ax1[0].plot(unicountvals_2_t, Pn2_n1_t[:,2*nII], '-', label='model',color=colors[it])
    ax1[0].plot(unicountvals_2_t[1:3], np.power(10,-4-it/4)*np.ones((2,)), '-',color=colors[it])#,clip_on=False)#,mfc='w')
    if np.mod(it,5)==0:
        ax1[0].text(unicountvals_2_t[3], np.power(10,-4-it/4-0.3),'$'+str(2*it)+'$')

    ax1[0].set_title(r'$P(n^{\prime}|n)$')
    ax1[0].set_xlabel(r'$n^{\prime}$')
    ax1[0].set_xlim([0,xlimmax])
    ax1[0].set_ylim([1e-7,1e0])
ax1[0].set_yscale('log')
ax1[0].text(unicountvals_2_t[1], np.power(10,-4+0.6),'$n$')

xlimmax=20
ax1[0].set_xlim([0,xlimmax])

mcol='r'

ax1[1].plot(unicountvals_1_d,Pn1_d,'k.',label='data')
ax1[1].plot(unicountvals_1_t[1:],Pn1_t[1:]/Pn1_t[1]*Pn1_d[1],mcol+'-',label='model',linewidth=1)
ax1[1].set_title(r'$P(n)$')
ax1[1].set_xlabel(r'$n$')
ax1[1].set_ylim([1e-7,1e0])
ax1[1].set_yticklabels([''])
ax1[1].set_yscale('log')
ax1[1].set_xscale('log')
fig1.tight_layout()
fig1.savefig("null_model_validation.pdf",format='pdf',dpi=500,bbox_inches='tight')
