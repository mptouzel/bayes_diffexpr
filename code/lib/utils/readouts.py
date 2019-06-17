import os
from functools import partial
from shutil import copytree,ignore_patterns
    
def print_func(strvar,fileid):
    print(str(strvar))
    fileid.write(str(strvar)+'\n')
    
def setup_outpath_and_logs(outpath,batchname):
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    prefix=outpath+batchname
    ignr_patts=('.git','.pyc')
    copytree('../code',prefix+'/code',ignore=ignore_patterns(*ignr_patts))
    logfileid=open(prefix+'logs.txt', 'a')
    prtfn=partial(print_func,fileid=logfileid)
    prtfn('-------------New run-------------')
    prtfn('output directory: '+outpath)
    prtfn('logging run in '+batchname+ 'logs.txt')
    return prtfn