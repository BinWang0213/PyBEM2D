#########################################################################
#       (C) 2017 Department of Petroleum Engineering,                   # 
#       Univeristy of Louisiana at Lafayette, Lafayette, US.            #
#                                                                       #
# This code is released under the terms of the BSD license, and thus    #
# free for commercial and research use. Feel free to use the code into  #
# your own project with a PROPER REFERENCE.                             #
#                                                                       #
# PYBEM2D Code                                                          #
# Author: Bin Wang                                                      # 
# Email: binwang.0213@gmail.com                                         # 
# Reference: Wang, B., Feng, Y., Berrone, S., et al. (2017) Iterative   #
# Coupling of Boundary Element Method with Domain Decomposition.        #
# doi:                                                                  #
#########################################################################

'''BEM 2D Module
-BEM_2D.py		        #Mainfunction(Mesh,BC,Solve,Plot)
-BEM_Elements.py        #BEM Element main class
 -Constant_element.py       #Constant element module
 -Linear_element.py         #Linear element module
 -Quadratic_element.py      #Quadratic element module
 -Constant_element_Trace.py #Constant element with internal trace module
 -Quadratic_element_Trace.py#Quadratic element with internal trace module
 -ShapeFunc.py              #Shape function in kernel integration
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.mlab import griddata
import matplotlib
matplotlib.rcParams.update({'font.size': 15}) #font

#[General Geometry Lib]
from Lib.Tools.Geometry import *

###############################
#
#  Core BEM 2D Class
#
###############################

class BEM_2DPostprocess:
    """Contains information and functions related to a 2D potential problem using BEM."""
        
    def __init__(self,Mesh,BEMobj):
        """Creates a BEM objecti with some specific paramters

        
        
        Author:Bin Wang(binwang.0213@gmail.com)
        Date: July. 2017
        """

        self.Mesh = Mesh
        self.BEMobj = BEMobj

    def print_Solution(self):
        """Check Meshing and B.C for different type of elements
        
        Author:Bin Wang(binwang.0213@gmail.com)
        Date: May. 2018
        """
        #Check Meshing and B.C for different type of elements
        print("[Mesh] State")
        print("Number of boundary elements:%s" %
              (len(self.BEMobj.BEs_edge) + len(self.BEMobj.BEs_trace)))
        print("Edge Num.:%s" % (self.Mesh.Num_boundary))
        print("# Neumann-1   Dirichlet-0")
        print("(E)Pts\tP\tQ\tU\tV")

        mass_balance=0

        for i, pl in enumerate(self.BEMobj.BEs_edge):
            eleid = i + 1
            P, Q = pl.get_PQ()
            U,V =pl.get_U(),pl.get_V()
            for j in range(self.Mesh.Nedof_edge):
                nodeid = self.Mesh.getNodeId(i, j, 'Edge') + 1
                print("(%s)%s\t%.2f\t%.2f\t%.2f\t%.2f" % (eleid, nodeid, P[j], Q[j],U[j],V[j]))
                mass_balance = mass_balance + Q[j]*pl.length # Mass balance should be the unit strength
        mass_e = mass_balance
        
        print('Unit Boundary Flux', mass_e)
        print("Trace Num.:%s" % (self.Mesh.Num_trace))
        eleid = 0
        for ti in range(self.Mesh.Num_trace):
            print("--Trace ", i + 1)
            for i, pl in enumerate(self.BEMobj.BEs_trace[ti]):
                P, Q = pl.get_PQ()
                U,V =pl.get_U(),pl.get_V()
                for j in range(self.Mesh.Nedof_trace):
                    global_eleid = eleid + self.Mesh.Ne_edge + 1
                    global_index = self.Mesh.getNodeId(eleid, j, 'Trace') + 1
                    #print("(%s)%s\t%5.3f\t\t%.3f\t\t%.4s\t\t%d" % (index,2*len(self.BEMobj.BEs_edge)+index,pl.xa,pl.ya,pl.element_type,pl.bd_marker))
                    print("(%s)%s\t%.2f\t%.2f\t%.2f\t%.2f" %
                          (global_eleid, global_index, P[j], Q[j], U[j], V[j]))
                    mass_balance = mass_balance + Q[j]
                eleid = eleid + 1
        mass_t = mass_balance - mass_e
        print('Unit Trace Flux', mass_t)
        if(mass_t==0):mass_t=1
        print('Mass Balance=', mass_balance,abs(mass_e/mass_t))

    ###########Post-processing Module#################
    def get_BDFlux(self,bd_markerID):
        """Get the real flux value of a boundary
           
        Arguments
        ---------
        bd_markerID  -- boundary marker id

        Author:Bin Wang(binwang.0213@gmail.com)
        Date: July. 2017
        """
        q=[]#flux,dp/dn

        elementID=self.Mesh.bdmarker2element(bd_markerID)#find the element idx on this edge

        if (bd_markerID>self.Mesh.Num_boundary-1):#this is a trace
            ti = elementID[0][0]  # tracerID
            for ei, pl in enumerate(self.BEMobj.BEs_trace[ti]):
                q = q+ pl.get_Q()

        else:  # this is a boundary edge
            for i in range(len(elementID)):  # loop for all elements on this edge
                pl = self.BEMobj.BEs_edge[elementID[i]]
                q.append(np.array(pl.get_Q())*pl.length)

        return q,sum(q)

    def get_BDSolution(self,bd_markerID):
        """Get the solution variable(p,ux,uy) at all nodes on any given boundary (bd_markerID)
           normally used for iterative solver
           
        Arguments
        ---------
        bd_markerID  -- boundary marker id

        Author:Bin Wang(binwang.0213@gmail.com)
        Date: July. 2017
        """
        p=[]#pressure,p
        q=[]#flux,dp/dn
        u=[]#flux in x direction,dp/dx
        v=[]#flux in y direction,dp/dy

        elementID=self.Mesh.bdmarker2element(bd_markerID)#find the element idx on this edge

        if (bd_markerID>self.Mesh.Num_boundary-1):#this is a trace
            ti = elementID[0][0]#tracerID
            for ei, pl in enumerate(self.BEMobj.BEs_trace[ti]):
                temp = pl.get_PQ()
                p=p+temp[0]
                q=q+temp[1]
                u=u+pl.get_U()
                v=v+pl.get_V()
        
        else:#this is a boundary edge
            for i in range(len(elementID)):#loop for all elements on this edge
                ele=self.BEMobj.BEs_edge[elementID[i]]
                p,q=ele.get_PQ()
                u=ele.get_U()
                v=ele.get_V()

        #darcy flow, velcoity=-dp/dx
        p=np.array(p)
        q=np.array(q)
        u=np.array([-i for i in u]) #May not correct
        v=np.array([-i for i in v]) #May not correct
        return p,q,u,v
    
    ###########Visulation Module################
    def plot_Solution(self,v_range=None,p_range=None,resolution=20):
        """Plot pressure&velocity field and Preview the streamline
           v_range=(0,300),p_range=(50,100),resolution=20

        Author:Bin Wang(binwang.0213@gmail.com)
        Date: July. 2017
        """
        print("[Visulization] Plotting Solution")
        
        debug=0
        
        #Get the sampling point within a polygon using mtri triangulation
        extent = (self.Mesh.domain_min[0],self.Mesh.domain_max[0],
                  self.Mesh.domain_min[1],self.Mesh.domain_max[1])
        Pts, triang = GetPtsInPolygon(self.Mesh.Pts_e, resolution)
        N=len(Pts)
        
        #Calculate the velocity and pressure field
        p = np.empty(N, dtype=float)
        u = np.empty(N, dtype=float)
        v = np.empty(N, dtype=float)
        for i in range(N):
                puv=self.BEMobj.get_Solution(Pts[i])
                p[i],u[i],v[i]=puv[0],puv[1],puv[2]    
        Vtotal = np.sqrt(u**2 + v**2)
        #Resampling data to regular grid
        xs, ys = np.meshgrid(np.linspace(extent[0], extent[1], resolution), np.linspace(extent[2], extent[3], resolution))
        Vtotal_grid = griddata(*Pts.T, Vtotal, xs, ys, interp='linear')
        p_grid = griddata(*Pts.T, p, xs, ys, interp='linear')
        u_grid = griddata(*Pts.T, u, xs, ys, interp='linear')
        v_grid = griddata(*Pts.T, v, xs, ys, interp='linear')

        fig, axes = plt.subplots(nrows=3,figsize=(15, 15))
        

        if(v_range is None):
            v_range = [min(Vtotal), max(Vtotal)]
        if(p_range is None):
            p_range=[min(p),max(p)]

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        
        for i, ax in enumerate(axes.flat):
            ax.set(aspect='equal')

            #Background Mesh
            ax.plot(*np.asarray(list(self.Mesh.Pts_e) + [self.Mesh.Pts_e[0]]).T, 'k-',lw=1)
            for ti in range(self.Mesh.Num_trace):
                ax.plot(*np.asarray(self.Mesh.Pts_t[ti]).T, 'k-',lw=1)
        

            if i==0:
                ax.set_title(r'Velocity Field')
                level = np.linspace(v_range[0], v_range[1], 64, endpoint=True)
                #im = ax.tricontour(*Pts.T, Vtotal, linestyles='-',
                #                   colors='black', linewidths=0.5)
                #im = ax.tricontourf(*Pts.T, Vtotal, level, cmap=plt.cm.jet)
                #im = ax.tricontour(triang, Vtotal, linestyles='-',
                #                   colors='black', linewidths=0.5)
                im = ax.tricontourf(triang, Vtotal, level, cmap=plt.cm.jet)
                '''
                im=ax.imshow(Vtotal_grid,vmin=v_range[0],vmax=v_range[1],
                                 extent=extent,origin='lower',interpolation='bicubic',cmap=plt.cm.jet)
                '''
                divider = make_axes_locatable(ax)
                cax1=divider.append_axes("right", "10%", pad=0.15)
                cbar1 = plt.colorbar(im, cax=cax1,format="%.2f")

                ax.use_sticky_edges = False
                ax.margins(0.05)

            if i==1:
                ax.set_title(r'Pressure Field')
                #im=ax.pcolormesh(X,Y,Vtotal,vmax=Vtotal_max)
                level = np.linspace(p_range[0], p_range[1], 64, endpoint=True)
                #im = ax.tricontour(*Pts.T, p, level, linestyles='-',
                #                   colors='black', linewidths=0.5)
                #im = ax.tricontourf(*Pts.T, p, level, cmap=plt.cm.jet)
                #im = ax.tricontour(triang, p, level, linestyles='-',
                #                   colors='black', linewidths=0.5)
                im = ax.tricontourf(triang, p, level, cmap=plt.cm.jet)
                #im=ax.imshow(p_grid,vmin=p_range[0],vmax=p_range[1],
                #                extent=extent,origin='lower',interpolation='bicubic',cmap=plt.cm.jet)
                divider = make_axes_locatable(ax)
                cax2=divider.append_axes("right", "10%", pad=0.15)
                cbar2 = plt.colorbar(im, cax=cax2,format="%.2f")

                ax.use_sticky_edges = False
                ax.margins(0.05)
            if i==2:
                import matplotlib.colors as colors
                import warnings
                warnings.filterwarnings('ignore') #hide the warnning when "nan" involves
                ax.set_title(r'Velocity Field')
                #ax.quiver(*Pts.T, u, v, Vtotal, pivot='mid', cmap=plt.cm.jet)
                space = int(resolution/10)+1
                Pts, u, v, Vtotal = Pts[::space], u[::space], v[::space], Vtotal[::space]
                ax.quiver(*Pts.T, u/Vtotal, v/Vtotal, pivot='mid')
                ax.scatter(*Pts.T,s=10,alpha=0.8,c='r')
                divider = make_axes_locatable(ax)
                cax3=divider.append_axes("right", "10%", pad=0.15)
                cbar3 = plt.colorbar(im, cax=cax3,format="%.2f")
                
                ax.use_sticky_edges = False
                ax.margins(0.05)

        
        fig.tight_layout()
        #Give some margin for velocity field
        #plt.savefig('Field_Plot.png',dpi=300)
        plt.show()      
        
        return p,u,v
        
    def plot_SolutionBD(self, plot=1, func=None):
        """Line plot solution along the boundary
        
        Author:Bin Wang(binwang.0213@gmail.com)
        Date: July. 2017
        """

        Pts=EndPointOnPolygon(self.Mesh.Pts_e,Nseg=100*len(self.Mesh.Pts_e))
        Pts=Pts+[Pts[0]]
        ArcLength=[0]*len(Pts)
        for i in range(len(Pts)-1):
            ArcLength[i + 1] = ArcLength[i]+calcDist(Pts[i], Pts[i + 1])
        
        if(func is None):
            Y=np.array([self.BEMobj.get_Solution(p) for p in Pts])
        else:
            Y = np.array([func(p) for p in Pts])

        if(plot == 1):
            self.plot_PUV_Pts(ArcLength, Y)
        return ArcLength,Y

    def plot_Solution_overline(self,Pts0,Pts1,plot=1,func=None):
        """Line plot solution along LINE(pts0,pts1)
        
        Author:Bin Wang(binwang.0213@gmail.com)
        Date: July. 2017
        """
        Pts = EndPointOnLine(Pts0, Pts1, Nseg=100, refinement="linspace")
        if(func is None):
            Y = np.array([self.BEMobj.get_Solution(p) for p in Pts])
        else:
            Y = np.array([func(p) for p in Pts])

        X = [calcDist(Pts[0],Pts1) for Pts1 in Pts]

        if(plot==1):
            self.plot_PUV_Pts(X,Y)
        return X,Y

    ###########Additional Tools###############
    def plot_PUV_Pts(self,X,Y):

        font = {'family': 'serif',
                'color':  'black',
                'weight': 'normal',
                'size': 16,
                }

        #General line plot for PUV see plot_Solution_overline
        p = Y[:, 0]
        u = Y[:, 1]
        v = Y[:, 2]
        Space = int(len(p) / 25)

        fig, axes = plt.subplots(nrows=3, figsize=(10, 10))
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        for i, ax in enumerate(axes.flat):
            if i == 0:
                ax.set_title(r'Pressure along Boundary', fontdict=font)
                im = ax.plot(X, p, 'bo-', markevery=Space)
                ax.set_xlabel('Arc length', fontdict=font)
                ax.set_ylabel('Values', fontdict=font)
                divider = make_axes_locatable(ax)
            if i == 1:
                ax.set_title(r'Velocity(x) along boundary', fontdict=font)
                im = ax.plot(X, u, 'bo-', markevery=Space)
                ax.set_xlabel('Arc length', fontdict=font)
                ax.set_ylabel('Values', fontdict=font)
                divider = make_axes_locatable(ax)
            if i == 2:
                ax.set_title(r'Velocity(y) along boundary', fontdict=font)
                im = ax.plot(X, v, 'bo-', markevery=Space)
                ax.set_xlabel('Arc length', fontdict=font)
                ax.set_ylabel('Values', fontdict=font)
                divider = make_axes_locatable(ax)

        fig.tight_layout()
        plt.show()
    
    def Compare_LinePlots(self, DataSetsX, DataSetsY, DataNames, title='Solution Compare'):
        """Compare differernt solutions in one plot

        Author:Bin Wang(binwang.0213@gmail.com)
        Date: July. 2018
        """
        plt.figure(num=None, figsize=(5, 5), dpi=80, facecolor='w', edgecolor='k')

        NumSets = len(DataSetsX)

        font = {'family': 'serif',
                'color':  'black',
                'weight': 'normal',
                'size': 16,
                }

        linestyles = ['_', '-', '--', ':']
        colors = ['b', 'g', 'r', 'c']
        markers=['o','v','s','*']

        for i in range(NumSets):
            Y = np.array(DataSetsY[i])
            X = np.array(DataSetsX[i])
            Space =int(len(X) / 25)
            plt.plot(X, Y, color=colors[i],marker=markers[i],
                    markersize=6, linewidth=2, markevery=Space)

        plt.legend(DataNames,bbox_to_anchor=(1.05, 1),
                   loc=2, borderaxespad=0., fontsize=15)
        plt.title(title, fontdict=font)
        plt.xlabel('X', fontdict=font)
        plt.ylabel('Y', fontdict=font)
        plt.show()
