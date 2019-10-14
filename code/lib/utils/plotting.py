import numpy as np
#import matplotlib.pyplot as pl
#pl.rc("figure", facecolor="gray",figsize = (8,8))
#pl.rc('lines',markeredgewidth = 2)
#pl.rc('font',size = 24)
#pl.rc('text', usetex=True)
#import seaborn as sns
#sns.set_style("whitegrid", {'axes.grid' : True})

def plot_n1_vs_n2(sp,savename,savepath,save,pl,colorbar=False,figsize=(8,8),thresh=0.05,mkrsz=70,minmkrsz=5,ftsz=20):
    if save:
        figsize=(5.5/3,5.5/3)
        thresh=1e-7
        mkrsz=20
        minmkrsz=2
        ftsz=8

    indn1_d,indn2_d,countpaircounts_d,unicountvals_1_d,unicountvals_2_d,NreadsI_d,NreadsII_d=sp
    fig1,ax1=pl.subplots(1,1,figsize=figsize)
    maxcount=1e4#1e6
    ax1.set_aspect(1)
    ax1.set_xlim([1.5e-1,maxcount])
    ax1.set_ylim([1.5e-1,maxcount])
    ax1.plot( ax1.get_xlim(), ax1.get_ylim(),color=[0.9,0.9,0.9])
    
    vals = ( np.log(countpaircounts_d)-np.log(np.min(countpaircounts_d)) )/ ( np.log(np.sum(countpaircounts_d))-np.log(np.min(countpaircounts_d)))
    smallinds=(vals<thresh) #0.05
    unicountvals_1_dt=np.asarray(unicountvals_1_d, dtype=float)
    unicountvals_2_dt=np.asarray(unicountvals_2_d, dtype=float)
    zeroval=10**-0.5
    unicountvals_1_dt[unicountvals_1_dt==0]=zeroval
    unicountvals_2_dt[unicountvals_2_dt==0]=zeroval
    ax1.plot(unicountvals_1_dt[indn1_d[smallinds]],unicountvals_2_dt[indn2_d[smallinds]],'.',mew=0,mfc='k',markersize=minmkrsz)
    
    smallind1=indn1_d[~smallinds]
    smallind2=indn2_d[~smallinds]
    for vit,val in enumerate(vals[~smallinds]): 
        ax1.plot(unicountvals_1_dt[smallind1[vit]],unicountvals_2_dt[smallind2[vit]],'.',mew=0.,color=str(val*0.9),markersize=mkrsz*val)
    
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_xlabel(r'$n$',fontsize=ftsz)
    ax1.set_ylabel(r'$n^\prime$',fontsize=ftsz)
    ax1.set_xticks([10**n for n in range(5)])
    ax1.set_yticks([10**n for n in range(5)])
    fig1.suptitle(r'$p(n,n^\prime)$')    
    add_ticks(ax1,[zeroval],[r'$0$'],'x',ftsz,pl)
    add_ticks(ax1,[zeroval],[r'$0$'],'y',ftsz,pl)

    if colorbar:
        import matplotlib
        ax2 = fig1.add_axes([.95, 0.12, 0.05, 0.76])
        cmap = pl.cm.gray
        norm = matplotlib.colors.Normalize(vmin=np.log10(1/float(np.sum(countpaircounts_d))), vmax=0)
        cb1 = matplotlib.colorbar.ColorbarBase(ax2, cmap=cmap,
                                    norm=norm,
                                    orientation='vertical',
                                    ticks=[-6,-4,-2,0.])
        cb1.set_ticklabels([r'$10^{-6}$',r'$10^{-4}$',r'$10^{-2}$',r'$10^{0}$'])#,r'$10^-3$',r'$10^-2$',r'$10^-1$'])

    if save:
        fig1.savefig(savepath+savename+'.pdf',format= 'pdf',dpi=1000, bbox_inches='tight')
    
    return fig1
        
def add_ticks(ax,newLocs,newLabels,pos,ftsz,pl):
    # Draw to get ticks
    pl.draw()

    # Get existing ticks
    if pos=='x':
        locs = ax.get_xticks().tolist()
        labels=[x.get_text() for x in ax.get_xticklabels()]
    elif pos =='y':
        locs = ax.get_yticks().tolist()
        labels=[x.get_text() for x in ax.get_yticklabels()]
    else:
        print("WRONG pos. Use 'x' or 'y'")
        return

    # Build dictionary of ticks
    Dticks=dict(zip(locs,labels))

    # Add/Replace new ticks
    for Loc,Lab in zip(newLocs,newLabels):
        Dticks[Loc]=Lab

    # Get back tick lists
    locs=list(Dticks.keys())
    labels=list(Dticks.values())

    # Generate new ticks
    if pos=='x':
        ax.set_xticks(locs)
        ax.set_xticklabels(labels,fontsize=ftsz)
    elif pos =='y':
        ax.set_yticks(locs)
        ax.set_yticklabels(labels,fontsize=ftsz)
