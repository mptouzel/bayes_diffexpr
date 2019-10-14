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

# ## Code that learns paras and produces (n,nprime) histograms for different measurement models, i.e. P(n|f)

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

# run learning and sampling for all models

#load paths to where learned paras are stored
path = rootpath + 'output/'#outdata_all/'
constr_type=1
acq_model_type=2
parvernull='null_pair_v4_ct_'+str(constr_type)+'_mt_'+str(acq_model_type)+'_'
runstr = parvernull+'min' + str(mincount) + '_max' + str(maxcount) 
outpath = path + dataset_pair[0] + '_' + dataset_pair[1] + '/' + runstr + '/'

# +
init_paras_arr_S2 = [ np.asarray([ -2.09861242,   2.41461504,   1.07386958,   6.62476807,-10.29170942]), \
                   np.asarray([ -2.13049915,   2.30240556,  -0.29026783,   6.56695534,-10.31522596]), \
                   np.asarray([-2.15940069,  2.61809851,  0.9954879 , -9.89262766]), \
                   np.asarray([-2.15199494,  -9.73615977]) \
                 ]
# init_paras_arr_S1 = [ np.asarray([ -2.07678873,   2.29472138,   1.09323841,   6.65204556,-10.27061188]), \
#                    np.asarray([ -2.07585556,   2.33165493,  -0.34198692,   6.53797226,-10.58516877]), \
#                    np.asarray([-2.15206189,  0.67881258,  1.04086898, -9.46699067]), \
#                    np.asarray([-2.15199494,  -9.73615977]) \
#                  ]

for acq_model_type in np.arange(3):
    
    #learn
#     outstruct,constr_value=learn_null_model(sparse_rep,acq_model_type,init_paras_arr[acq_model_type])
#     print('model '+str(acq_model_type)+' took '+str(time.time()-st))
#     print(constr_value)
#     print(outstruct.x)
#     np.save('outstruct_mt_'+str(acq_model_type),outstruct)
    
    #sample
#     outstruct=np.load('outstruct_'+donor+'_mt_'+str(acq_model_type)+'.npy').item()
#     opt_paras=outstruct.x
    opt_paras=init_paras_arr_S2[acq_model_type]
    f_samples,pair_samples=get_nullmodel_sample_observedonly(opt_paras,acq_model_type,NreadsI_d,NreadsII_d,Nsamp)
    
    #plot
    sparse_rep_t=get_sparserep(pair_samples.loc[:,['Clone_count_1','Clone_count_2']])
    plot_n1_vs_n2(sparse_rep_t,'null_'+donor+'_n1_v_n2_mt_'+str(acq_model_type),'',True,pl)
# -

f_samples,pair_samples=get_nullmodel_sample_observedonly(np.asarray([-2.19629167,  2.40560287,  1.10944424,  6.61597503, -9.32561711]),0,NreadsI_d,NreadsII_d,Nsamp)
initparas=np.load(path + donor+'_0_F1_'+donor+'_0_F2/' + runstr + '/'+'optparas.npy')
