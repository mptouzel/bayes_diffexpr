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



# Read in data

# Plot raw numbers

for dit,donor in enumerate(donorstrvec):
    nfbins=800
    for ddit,day in enumerate(dayvec):
        print(day)
        outpath='outdata_all/'+donor+'_'+day+'_F1_'+donor+'_'+day+'_F2/min0_maxinf_v1__case0/'
        setname=outpath+'outstruct.npy'
        uni1=np.load(outpath+'unicountvals_1_d.npy')
        uni2=np.load(outpath+'unicountvals_2_d.npy')
        indn1=np.load(outpath+'indn1_d.npy')
        indn2=np.load(outpath+'indn2_d.npy')
        countpairs=np.load(outpath+'countpaircounts_d.npy')
        N_1=np.sum((uni1[indn1]>0)*countpairs)
        N_2=np.sum((uni2[indn2]>0)*countpairs)
        N_all=np.sum(countpairs)
        print(str(N_all)+' '+str(N_1)+' '+str(N_2))
