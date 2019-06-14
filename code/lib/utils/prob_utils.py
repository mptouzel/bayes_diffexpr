import numpy as np

def get_distsample(pmf,Nsamp,dtype='uint16'):
    '''
    generates Nsamp index samples of dtype (e.g. uint16 handles up to 65535 indices) from discrete probability mass function pmf
    '''
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
    sampled_pairs = np.array(index[np.argsort(index[:,0])],dtype=dtype)
    return sampled_pairs