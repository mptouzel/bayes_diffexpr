import os
from functools import partial
from shutil import copytree,rmtree,ignore_patterns
    
def print_func(strvar,fileid):
    print(str(strvar))
    fileid.write(str(strvar)+'\n')
    
def setup_outpath_and_logs(outpath,batchname):
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    prefix=outpath+batchname
    logfileid=open(prefix+'logs.txt', 'a')
    prtfn=partial(print_func,fileid=logfileid)
    prtfn('-------------New run-------------')
    prtfn('output directory: '+outpath)
    prtfn('logging run in '+batchname+ 'logs.txt')
    
    ignr_patts=('.git','.pyc','.npy.','ipynb')
    #if os.path.exists(outpath+'code'):
        #prtfn('overwriting '+outpath+'code')
        #rmtree(outpath+'code')
    #copytree('../code',outpath+'code',ignore=ignore_patterns(*ignr_patts))
    
    return prtfn