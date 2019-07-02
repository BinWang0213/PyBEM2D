import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


import matplotlib.tri as tri
import math


def plotKirsch(func,unit=1):
    #Modified from https://matplotlib.org/examples/pylab_examples/tripcolor_demo.html

    # First create the x and y coordinates of the points.
    n_angles = 72
    n_radii = 100 #number of refinement in radial direction
    min_radius = R
    max_radisu = r_max
    radii = np.linspace(min_radius, max_radisu, n_radii)

    angles = np.linspace(0, 2*math.pi, n_angles, endpoint=False)
    angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)
    angles[:, 1::2] += math.pi/n_angles

    x = (radii*np.cos(angles)).flatten()
    y = (radii*np.sin(angles)).flatten()

    triang = tri.Triangulation(x, y)

    # Mask off unwanted triangles.
    xmid = x[triang.triangles].mean(axis=1)
    ymid = y[triang.triangles].mean(axis=1)
    mask = np.where(xmid*xmid + ymid*ymid < min_radius*min_radius, 1, 0)
    triang.set_mask(mask)

    z=func(radii,angles).flatten()/unit

    plt.gca().set_aspect('equal')
    plt.tripcolor(triang, z,shading='gouraud',cmap='jet')
    plt.colorbar()

    return z

def showTables(X,Y=[],XLables=[],YLabels=[],preview=10):
    from IPython.display import display_html,display
    def display_side_by_side(*args):
        html_str=''
        for df in args:
            html_str+=df.to_html()
        display_html(html_str.replace('table','table style="display:inline"'),raw=True)
    
    NumDataSets=len(X)       
    
    Tables=[]
    for i in range(NumDataSets):
        NumData=len(X[i])
        PreviewStep=int(NumData/preview)+1
        if(len(Y)==0):#Only one table needed
            Tables.append(pd.DataFrame({XLables[i] : X[i][0::PreviewStep]}))
        else:
            Tables.append(pd.DataFrame({XLables[i] : X[i][0::PreviewStep],
                                            YLabels[i] : Y[i][0::PreviewStep]}))
    
    if(len(Y)==0):
        Tables=[pd.concat(Tables,axis=1)]
    display_side_by_side(*[x for x in Tables])


#Publication quality figure paramters
_params = {'font.family': 'sans-serif',
           'font.serif': ['Times', 'Computer Modern Roman'],
           'font.sans-serif': ['Helvetica', 'Arial',
                               'Computer Modern Sans serif'],
           'font.size': 14,

           'axes.labelsize': 14,
           'axes.linewidth': 1,

            
           'savefig.dpi': 300,
           'savefig.format': 'eps',
           # 'savefig.bbox': 'tight',
           # this will crop white spaces around images that will make
           # width/height no longer the same as the specified one.

           'legend.fontsize': 14,
           'legend.frameon': False,
           'legend.numpoints': 1,
           'legend.handlelength': 2,
           'legend.scatterpoints': 1,
           'legend.labelspacing': 0.5,
           'legend.markerscale': 0.9,
           'legend.handletextpad': 0.5,  # pad between handle and text
           'legend.borderaxespad': 0.5,  # pad between legend and axes
           'legend.borderpad': 0.5,  # pad between legend and legend content
           'legend.columnspacing': 1,  # pad between each legend column

           'xtick.labelsize': 14,
           'ytick.labelsize': 14,
           'xtick.direction':'in',
           'ytick.direction':'in',
           
           'lines.linewidth': 1,
           'lines.markersize': 7,
           # 'lines.markeredgewidth' : 0,
           # 0 will make line-type markers, such as '+', 'x', invisible
           }

linestyles = ['-', '--', ':','-.' ,'-','-.','-']
colors = ['b', 'r','k', 'g', 'c','m','tab:pink']
TabColor=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
markers=['o', 's', 'v', '^', 'D', '<', '>', 'p', 'h']

def plotTables(X,Y,XLable='',YLabel='',DataNames=[],Title='',
               Xlim=None,Ylim=None, subplots=[111],RegionShade=None, Alpha=[],LineWidth=[],Colors=[],legendOut=False,
               MarkerSize=[],InvertY=False,logXY=False,logX=False,logY=False,img_fname=None):
    #http://www.scipy-lectures.org/intro/matplotlib/matplotlib.html

    from matplotlib import rcParams
    
    #Set paramters all at once
    for i in _params:
        rcParams[i]=_params[i]
    
    NumDataSets=len(X)

    if(NumDataSets>6):#So many datasets we need put legend out of box
        legendOut=True

    if(legendOut): fig=plt.figure(figsize=(6.3,4),dpi=80)
    else: fig=plt.figure(figsize=(5,4),dpi=80)
    if(len(subplots)>1):
        fig=plt.figure(figsize=(4*len(subplots)*0.8,5),dpi=80)


    if(len(Alpha)==0):
        Alpha=NumDataSets*[1]
    if(len(MarkerSize)==0):
        MarkerSize=NumDataSets*[7]
    if(len(LineWidth)==0):
        LineWidth=NumDataSets*[1.5]
    if(len(Colors)==0):
        Colors=colors
    
    for i in range(NumDataSets):
        y = np.array(Y[i])
        x = np.array(X[i])

        Space =max(1,int(len(x) / 10000))
        if(len(subplots)>1):
            fig.add_subplot(subplots[i])
        plt.plot(x, y, color=Colors[i],linestyle=linestyles[i],marker=markers[i],alpha=Alpha[i],mfc='none',
                        MarkerSize=MarkerSize[i], linewidth=LineWidth[i], markevery=Space,label=DataNames[i])            

        #Set XYlim
        if(Xlim!=None):
            if(any(isinstance(t, list) for t in Xlim)):
                plt.xlim(Xlim[i])
            else:
                plt.xlim(Xlim)
        if(Ylim!=None):
            if(any(isinstance(t, list) for t in Ylim)):
                plt.ylim(Ylim[i])
            else:
                plt.ylim(Ylim)
        #Invert Y
        if (InvertY==True):
            plt.gca().xaxis.tick_top()
            plt.gca().xaxis.set_label_position('top')
            plt.gca().invert_yaxis()
        
        if(logXY):
            plt.yscale('log')
            plt.xscale('log')
        if(logX):
            plt.xscale('log')
        if(logY):
            plt.yscale('log')


        if(RegionShade!=None):
            for ri in range(len(RegionShade[i])):
                plt.axhspan(*RegionShade[i][ri], facecolor=TabColor[ri], edgecolor='k',alpha=0.3)

        #plt.grid(linestyle='--')
        plt.title(Title)
        plt.xlabel(XLable)
        plt.ylabel(YLabel)
        if(legendOut):
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size': 12})
        else:
            plt.legend(loc='best',prop={'size': 12})
    
    plt.tight_layout(pad=0.7)
    if(img_fname is not None):
        plt.savefig(img_fname, bbox_inches='tight')
    plt.show()


def smooth(y, x,windows,plot=False,xlim=[],ylim=[]): #moving average
    df = pd.DataFrame({'y': y,'x':x})
    smooth=df.rolling(on='x',window=windows).mean()

    if(plot==True):
        plotTables(X=[smooth.y.values,y],Y=[smooth.x.values,x],
        Xlim=xlim,Ylim=ylim,Alpha=[1,0.5],LineWidth=[1.5,1.0],Colors=['b','tab:gray'],
        DataNames=['Smoothed','Raw'],InvertY=True,img_fname='img.png')

    return smooth.y.values,smooth.x.values


def rangeMean(y,x,ranges=[]): #Find y average for a given range in x
    if(np.array(ranges).ndim==1):
        return y[np.where((x>ranges[0]) & (x<ranges[1]))].mean()
    else:
        return [y[np.where((x>ri[0]) & (x<ri[1]))].mean() for ri in ranges]
