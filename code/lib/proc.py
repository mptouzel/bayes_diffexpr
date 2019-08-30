import numpy as np
import pandas as pd
from functools import partial

def import_data(path,filename1,filename2,mincount,maxcount,colnames1,colnames2):
    '''
    Reads in Yellow fever data from two datasets and merges based on nt sequence.
    Outputs dataframe of pair counts for all clones.
    Considers clones with counts between mincount and maxcount
    Uses specified column names and headerline in stored fasta file.
    '''
    
    headerline=0 #line number of headerline
    newnames=['Clone_fraction','Clone_count','ntCDR3','AACDR3']    
    with open(path+filename1, 'r') as f:
        F1Frame_chunk=pd.read_csv(f,delimiter='\t',usecols=colnames1,header=headerline)[colnames1]
    with open(path+filename2, 'r') as f:
        F2Frame_chunk=pd.read_csv(f,delimiter='\t',usecols=colnames2,header=headerline)[colnames2]
    F1Frame_chunk.columns=newnames
    F2Frame_chunk.columns=newnames
    suffixes=('_1','_2')
    mergedFrame=pd.merge(F1Frame_chunk,F2Frame_chunk,on=newnames[2],suffixes=suffixes,how='outer')
    for nameit in [0,1]:
        for labelit in suffixes:
            mergedFrame.loc[:,newnames[nameit]+labelit].fillna(int(0),inplace=True)
            if nameit==1:
                mergedFrame.loc[:,newnames[nameit]+labelit].astype(int)
    def dummy(x):
        val=x[0]
        if pd.isnull(val):
            val=x[1]    
        return val
    mergedFrame.loc[:,newnames[3]+suffixes[0]]=mergedFrame.loc[:,[newnames[3]+suffixes[0],newnames[3]+suffixes[1]]].apply(dummy,axis=1) #assigns AA sequence to clones, creates duplicates
    mergedFrame.drop(newnames[3]+suffixes[1], 1,inplace=True) #removes duplicates
    mergedFrame.rename(columns = {newnames[3]+suffixes[0]:newnames[3]}, inplace = True)
    mergedFrame=mergedFrame[[newname+suffix for newname in newnames[:2] for suffix in suffixes]+[newnames[2],newnames[3]]]
    filterout=((mergedFrame.Clone_count_1<mincount) & (mergedFrame.Clone_count_2==0)) | ((mergedFrame.Clone_count_2<mincount) & (mergedFrame.Clone_count_1==0)) #has effect only if mincount>0
    number_clones=len(mergedFrame)
    return number_clones,mergedFrame.loc[((mergedFrame.Clone_count_1<=maxcount) & (mergedFrame.Clone_count_2<=maxcount)) & ~filterout]

def get_sparserep(counts):
    '''
    Tranforms {(n1,n2)} data stored in pandas dataframe to a sparse 1D representation.
    unicountvals_1(2) are the unique values of n1(2).
    sparse_rep_counts gives the counts of unique pairs.
    indn1(2) is the index of unicountvals_1(2) giving the value of n1(2) in that unique pair.
    len(indn1)=len(indn2)=len(sparse_rep_counts)
    '''
    counts['paircount']=1 #gives a weight of 1 to each observed clone
    clone_counts=counts.groupby(['Clone_count_1','Clone_count_2']).sum()
    sparse_rep_counts=np.asarray(clone_counts.values.flatten(),dtype=int)
    clonecountpair_vals=clone_counts.index.values
    indn1=np.asarray([clonecountpair_vals[it][0] for it in range(len(sparse_rep_counts))],dtype=int)
    indn2=np.asarray([clonecountpair_vals[it][1] for it in range(len(sparse_rep_counts))],dtype=int)
    NreadsI=counts.Clone_count_1.sum()
    NreadsII=counts.Clone_count_2.sum()

    unicountvals_1,indn1=np.unique(indn1,return_inverse=True)
    unicountvals_2,indn2=np.unique(indn2,return_inverse=True)
    
    return indn1,indn2,sparse_rep_counts,unicountvals_1,unicountvals_2,NreadsI,NreadsII

def get_distsample(pmf,Nsamp,dtype='uint32'):
    '''
    generates Nsamp index samples of dtype (e.g. uint16 handles up to 65535 indices) from discrete probability mass function pmf.
    Handles multi-dimensional domain. Output is sorted.
    '''
    #assert np.sum(pmf)==1, "cmf not normalized!"
    
    shape = np.shape(pmf)
    sortindex = np.argsort(pmf, axis=None)#uses flattened array
    pmf = pmf.flatten()
    pmf = pmf[sortindex]
    cmf = np.cumsum(pmf)
    choice = np.random.uniform(high = cmf[-1], size = int(float(Nsamp)))
    index = np.searchsorted(cmf, choice)
    index = sortindex[index]
    index = np.unravel_index(index, shape)
    index = np.transpose(np.vstack(index))
    sampled_inds = np.array(index[np.argsort(index[:,0])],dtype=dtype)
    return sampled_inds


def get_pvalue(row,svec,cdfPs_n1n2ps):
    '''
    when applied to dataframe, generates 'p-value'-like (null hypo) probability, 1-P(s>0|n1_i,n2_i), where i is the row (i.e. the clone)
    '''
    return cdfPs_n1n2ps[np.argmin(np.fabs(svec)),row.sparse_ind]

def get_sparse_ind(row,unicountvals_1,unicountvals_2,n1_n2_to_n1n2):
    '''
    when applied to dataframe, generates unidimensional index from count pair
    '''
    return n1_n2_to_n1n2[row['Clone_count_1']==unicountvals_1,row['Clone_count_2']==unicountvals_2]
                   
def suffstats_table(pthresh, svec,logPsvec,subset, sparse_rep,logPn1n2_s):
    #'''
    #takes learned diffexpr model, Pn1n2_s*Ps, computes posteriors over (n1,n2) pairs, and writes to file a table of data with clones as rows and columns as measures of thier posteriors 
    #pthresh is the threshold in 'p-value'-like (null hypo) probability, 1-P(s>0|n1_i,n2_i), where i is the row (i.e. the clone) n.b. lower null prob implies larger probability of expansion. Used to reduce table to managable size.
    #'''

    indn1,indn2,sparse_rep_counts,unicountvals_1,unicountvals_2,NreadsI,NreadsII=sparse_rep

    Psn1n2_ps=np.exp(logPn1n2_s+logPsvec[:,np.newaxis]) 
    
    #compute marginal likelihood (neglect renormalization , since it cancels in conditional below) 
    Pn1n2_ps=np.sum(Psn1n2_ps,0)

    Ps_n1n2ps=np.exp(logPn1n2_s+logPsvec[:,np.newaxis])/Pn1n2_ps[np.newaxis,:]
    #compute cdf to get p-value to threshold on to reduce output size
    cdfPs_n1n2ps=np.cumsum(Ps_n1n2ps,0)

    n1_n2_to_n1n2=np.zeros((len(unicountvals_1),len(unicountvals_2)),dtype=int)
    for n1it,n1 in enumerate(unicountvals_1):
        for n2it,n2 in enumerate(unicountvals_2):
            ind=np.where(np.logical_and(n1==unicountvals_1[indn1],n2==unicountvals_2[indn2]))[0]
            if not ind:
                continue
            else:
                n1_n2_to_n1n2[n1it,n2it]=int(ind)                
    get_sparseind_part=partial(get_sparse_ind,unicountvals_1=unicountvals_1,unicountvals_2=unicountvals_2,n1_n2_to_n1n2=n1_n2_to_n1n2)        
    subset['sparse_ind']=subset.apply(get_sparseind_part, axis=1)
    subset.sparse_ind=subset.sparse_ind.astype('int32')
    get_pvalue_part=partial(get_pvalue,svec=svec,cdfPs_n1n2ps=cdfPs_n1n2ps)
    cdflabel=r'$1-P(s>0)$'
    subset[cdflabel]=subset.apply(get_pvalue_part, axis=1)
    
    #decrease clone list size by pvalue
    subset=subset[subset[cdflabel]<pthresh].reset_index(drop=True)
    subset=subset.sort_values(by=cdflabel)    
    
    #posteriors over data clones
    Ps_n1n2ps_datpairs=Ps_n1n2ps[:,subset.sparse_ind.values]
    
    #compute posterior metrics
    mean_est=np.zeros((len(subset),))
    max_est= np.zeros((len(subset),))
    slowvec= np.zeros((len(subset),))
    smedvec= np.zeros((len(subset),))
    shighvec=np.zeros((len(subset),))
    pval=0.025 #double-sided comparison statistical test
    pvalvec=[pval,0.5,1-pval] #bound criteria defining slow, smed, and shigh, respectively
    for it,column in enumerate(np.transpose(Ps_n1n2ps_datpairs)):
        mean_est[it]=np.sum(svec*column)
        max_est[it]=svec[np.argmax(column)]
        forwardcmf=np.cumsum(column)
        backwardcmf=np.cumsum(column[::-1])[::-1]
        inds=np.where((forwardcmf[:-1]<pvalvec[0]) & (forwardcmf[1:]>=pvalvec[0]))[0]
        slowvec[it]=np.mean(svec[inds+np.ones((len(inds),),dtype=int)])  #use mean in case there are two values
        inds=np.where((forwardcmf>=pvalvec[1]) & (backwardcmf>=pvalvec[1]))[0]
        smedvec[it]=np.mean(svec[inds])
        inds=np.where((forwardcmf[:-1]<pvalvec[2]) & (forwardcmf[1:]>=pvalvec[2]))[0]
        shighvec[it]=np.mean(svec[inds+np.ones((len(inds),),dtype=int)])
    
    subset.drop(columns='sparse_ind')
    colnames=(r'$\bar{s}$',r'$s_{max}$',r'$s_{3,high}$',r'$s_{2,med}$',r'$s_{1,low}$')
    for it,coldata in enumerate((mean_est,max_est,shighvec,smedvec,slowvec)):
        subset.insert(0,colnames[it],coldata)
    oldcolnames=( 'AACDR3',  'ntCDR3', 'Clone_count_1', 'Clone_count_2', 'Clone_fraction_1', 'Clone_fraction_2')
    newcolnames=('CDR3_AA', 'CDR3_nt',        r'$n_1$',        r'$n_2$',           r'$f_1$',           r'$f_2$')
    subset=subset.rename(columns=dict(zip(oldcolnames, newcolnames)))
    
    return subset


    ##go from clone count pair (n1,n2) to index in unicountvals_1_d and unicountvals_2_d
    #data_pairs_ind_1=np.zeros((len(subset),),dtype=int)
    #data_pairs_ind_2=np.zeros((len(subset),),dtype=int)
    #n1n2_ind=np.zeros((len(subset),),dtype=int)
    #for it in range(len(subset)):
        #data_pairs_ind_1[it]=np.where(int(subset.iloc[it].Clone_count_1)==unicountvals_1)[0]
        #data_pairs_ind_2[it]=np.where(int(subset.iloc[it].Clone_count_2)==unicountvals_2)[0]
        #n1n2_ind[it]=np.logical_and(data_pairs_ind_1[it],unicountvals_1[indn1],data_pairs_ind_2[it],unicountvals_2[indn2])