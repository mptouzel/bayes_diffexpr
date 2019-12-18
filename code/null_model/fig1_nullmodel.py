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

# ## Fig1: plots likelihood over all data and gives example sampled (n,n') distributions over 3 models (NB->Pois, NB, Pois)

# %matplotlib inline
import sys,os
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
# %run -i '../lib/utils/ipynb_setup.py'
from lib.utils.plotting import plot_n1_vs_n2,add_ticks
from lib.proc import get_sparserep,import_data,get_distsample
from lib.model import get_Pn1n2_s, get_rhof, NegBinParMtr,get_logPn_f,get_nullmodel_sample_observedonly
from lib.learning import nullmodel_constr_fn,callback,learn_null_model
import lib.learning
# %load_ext autoreload
# %autoreload 2

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

# ## Data plot

# import example same day data and transform to sparse rep:

# +
rootpath = '../../../'
daynull="0"
day="0"
donor='S2'
dataset_pair=(donor+'_'+daynull+'_F1', donor+'_'+day+'_F2')
colnames = [u'Clone fraction',u'Clone count',u'N. Seq. CDR3',u'AA. Seq. CDR3'] 
datarootpath=rootpath+'data/Yellow_fever/prepostvaccine/' 
mincount=0
maxcount=np.Inf
Nclones_samp,subset=import_data(datarootpath,dataset_pair[0]+'_.txt',dataset_pair[1]+'_.txt',mincount,maxcount,colnames,colnames)

#transform to sparse representation
sparse_rep_d=get_sparserep(subset.loc[:,['Clone_count_1','Clone_count_2']])
indn1_d,indn2_d,countpaircounts_d,unicountvals_1_d,unicountvals_2_d,NreadsI_d,NreadsII_d=sparse_rep_d       
Nsamp=np.sum(countpaircounts_d)

display(Math('$N_\mathrm{samp}='+str(Nclones_samp)+r'$, $N=10^9$, sampling efficincy:$'+str(Nclones_samp/float(10**9))+'$')) #approximate N=1e9
display(Math('$N_\mathrm{reads}='+str(NreadsI_d)+r'$, $M_\textrm{cells}=10^{6.8}$, sequencing efficincy:$'+str(NreadsI_d/(countpaircounts_d*(unicountvals_1_d[indn1_d])))+'$')) #approximate total reads=1e6.8

num_x0_data=subset.loc[subset.Clone_count_2==0,['Clone_count_1','Clone_count_2']].shape[0]
num_0x_data=subset.loc[subset.Clone_count_1==0,['Clone_count_1','Clone_count_2']].shape[0]
num_xx_data=np.sum(countpaircounts_d)-num_x0_data-num_0x_data
print(str(np.sum(countpaircounts_d))+' '+str(num_x0_data)+' '+str(num_0x_data)+' '+str(num_xx_data))
print(str(num_x0_data/float(np.sum(countpaircounts_d)))+' '+str(num_0x_data/float(np.sum(countpaircounts_d)))+' '+str(num_xx_data/float(np.sum(countpaircounts_d))))
# -

fig=plot_n1_vs_n2(sparse_rep_d,'','Pn1n2_data_S2',True,pl,colorbar=True)

# ## Model results

# load precomputed results (see below for examples of computations)

# +
# casestrvec=(r'$NB\rightarrow Pois$',r'$Pois \rightarrow NB$','$NB$','$Pois$')
casestrvec=(r'$NB\rightarrow Pois$','$NB$','$Pois$')
casevec=[0,2,3]
donorvec=['P1','P2','Q1','Q2','S1','S2']
# donorvec=("Azh","KB","Yzh","GS","Kar","Luci")
dayvec=['pre0','0','7','15','45']
nparasvec=(4,3,1)
outstructs=np.empty((len(casevec),len(donorvec),len(dayvec)),dtype=dict)
df = pd.DataFrame(columns={'donor','day','model','likelihood'})

for cit, case in enumerate(casevec):
    for dit,donor in enumerate(donorvec):
        for ddit,day in enumerate(dayvec):
            data_name=donor+'_'+str(day)+"_F1_"+donor+'_'+str(day)+'_F2'
        #             runstr='../../../output/'+dataname+'/null_pair_v1_null_ct_1_acq_model_type'+str(case)+'_min0_maxinf/'
            runstr='../../../output/'+data_name+'/null_pair_v4_ct_1_mt_'+str(case)+'_min0_maxinf/'
            setname=runstr+'outstruct.npy'
#             try:
            df=df.append(pd.Series({'donor':donor,'day':day,'model':case,'likelihood':-np.load(setname).flatten()[0].fun}),ignore_index=True)
            outstructs[cit,dit,ddit]=np.load(setname).flatten()[0]
#                 print(data_name+ ' '+str(case))
#             except IOError:
#                 True
#                 print(data_name+ ' '+str(case))
# -

dft=df.set_index(['donor','day'])

# +
fig,ax=pl.subplots()
bins=np.logspace(-4,0,20)
counts,bins=np.histogram(-(dft[dft['model']==0].likelihood.values[1:]-dft[dft['model']==2].likelihood.values[1:])/dft[dft['model']==0].likelihood.values[1:],bins)
ax.bar(bins[:-1],counts,width=np.diff(bins),align='edge',label=r'$\ell=\ell_\mathrm{NB}$')

bins3=np.logspace(-4,0,20)
counts3,bins3=np.histogram(-(dft[dft['model']==0].likelihood.values-dft[dft['model']==3].likelihood.values)/dft[dft['model']==0].likelihood.values,bins3)
ax.bar(bins3[:-1],counts3,width=np.diff(bins3),align='edge',label=r'$\ell=\ell_\mathrm{P}$')
ax.legend(frameon=False)
# ax.bar(np.power(10,bins3[:-1]),counts3)
ax.set_xscale('log')
ax.set_xlabel(r'$(\ell-\ell_{\mathrm{NBP}})/\ell_{\mathrm{NBP}}$')
ax.set_ylabel('count')
# -

fig.savefig('noise_model_likeihoods_hist.pdf',format='pdf',dpi=1000, bbox_inches='tight')

#other unused plots:
fig,ax=pl.subplots()
colorvec=['r','g','b','y','k','m']
stylevec=['solid','solid','dashed']
for it,model in enumerate(casevec):
    for cit,donor in enumerate(donorvec):
        df[np.logical_and(df['model']==model,df['donor']==donor)].plot.line(x='day',y='likelihood',ax=ax,color=colorvec[cit],linestyle=stylevec[it])
ax.get_legend().remove()
fig.savefig('noise_model_likeihoods_over_days.pdf',format='pdf',dpi=1000, bbox_inches='tight')
fig,ax=pl.subplots()
markervec=['o','s','d','v','^','+']
for cit,donor in enumerate(donorvec):
    for dit,day in enumerate(dayvec):
        cond_base=(df['model']==0) & (df['donor']==donor) &(df['day']==day)
        cond_2=(df['model']==2) & (df['donor']==donor) &(df['day']==day)
        cond_3=(df['model']==3) & (df['donor']==donor) &(df['day']==day)
        ax.plot(df.loc[cond_base,'likelihood'].values,df.loc[cond_base,'likelihood'].values-df.loc[cond_2,'likelihood'].values,color=colorvec[dit],marker=markervec[cit])
        cond=(df['model']==2) & (df['donor']==donor) &(df['day']==day)
        ax.plot(df.loc[cond,'likelihood'].values,df.loc[cond_base,'likelihood'].values-df.loc[cond_3,'likelihood'].values,color=colorvec[dit],marker=markervec[cit])
# ax.plot(ax.get_xlim(),ax.get_xlim(),'k--')
ax.set_yscale('log')
ax.set_xlabel(r'$\mathcal{L}_{\mathrm{max}}$')
ax.set_ylabel(r'$\mathcal{L}_{\mathrm{max}}-\mathcal{L}$')
ax.set_ylim(1e-5,1e0)
fig.savefig('noise_model_likeihoods_gap_w_best.pdf',format='pdf',dpi=1000, bbox_inches='tight')

# example Learning  noise model parameters

constr_type=1

acq_model_type=2
init_null_paras=np.array([-2.1674403 ,  1.09554235,  1.01933961, -9.54175371])
prtfn=print
outstruct_2,constr_value_2=learn_null_model(sparse_rep_d,acq_model_type,init_null_paras,constr_type=constr_type,prtfn=prtfn)
prtfn('constr value:')
prtfn(constr_value_2)

acq_model_type=0
init_null_paras=np.asarray([ -2.09861242,   2.41461504,   1.07386958,   6.62476807,-10.29170942])
prtfn=print
outstruct_0,constr_value_0=learn_null_model(sparse_rep_d,acq_model_type,init_null_paras,constr_type=constr_type,prtfn=prtfn)
prtfn('constr value:')
prtfn(constr_value_0)

acq_model_type=3
init_null_paras= np.asarray([-2.546525, -6.995147 ])
prtfn=print
outstruct,constr_value=learn_null_model(sparse_rep_d,acq_model_type,init_null_paras,constr_type=constr_type,prtfn=prtfn)
prtfn('constr value:')
prtfn(constr_value)

# Sample from learned models

# +
init_paras_arr_S2 = [ outstruct_0.x, \
                   outstruct_2.x, \
                   oustruct_3.x]

for acq_model_type in np.arange(3):
    opt_paras=init_paras_arr_S2[acq_model_type]
    f_samples,pair_samples=get_nullmodel_sample_observedonly(opt_paras,acq_model_type,NreadsI_d,NreadsII_d,Nsamp)
    
    #plot
    sparse_rep_t=get_sparserep(pair_samples.loc[:,['Clone_count_1','Clone_count_2']])
    plot_n1_vs_n2(sparse_rep_t,'null_'+donor+'_n1_v_n2_mt_'+str(acq_model_type),'',True,pl)
