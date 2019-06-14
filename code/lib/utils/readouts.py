import os

def print_func(strvar,fileid):
    print(str(strvar))
    fileid.write(str(strvar)+'\n')
    
def setup_outpath_and_logs(outpath,batchname):
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    from shutil import copy2
    prefix=outpath+batchname
   
    if not os.path.exists(prefix+'infer_diffexpr_main.py'):
        copy2('infer_diffexpr_main.py',prefix+'infer_diffexpr_main.py')
        copy2('infer_diffexpr_lib.py',prefix+'infer_diffexpr_lib.py')
    logfileid=open(prefix+'logs.txt', 'a')
    prtfn=partial(print_func,fileid=logfileid)
    prtfn('output directory: '+outpath)
    prtfn('logging run in '+batchname+ 'logs.txt')
    return prtfn