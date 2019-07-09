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

'''Domain Decomposition Module
-Coupling_Main.py		#Mainfunction(Class)
-Schemes/h              #Iterative schemes (4)
  |-P_DD.cpp				#Parallel Dirichelt-Dirichlet
  |-P_NN.cpp			    #Parallel Neumann-Neumann
  |-P_RR.cpp                #Parallel Robin-Robin
  |-S_DN.cpp                #Sequential Dirichelt-Neumann
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib
matplotlib.rcParams.update({'font.size': 18}) #font

#[General Geometry Module]
from ..Tools.Geometry import point_on_line,Interp_Nonconforming

#[DDM Schemes]
from .Schemes.P_RR import PRR
from .Schemes.P_NN import PNN
from .Schemes.P_DD import PDD
from .Schemes.S_DN import SDN


#####################################
#
#  General Iterative DDM Solver Class
#
#####################################

class DDM_Solver:
    """Class object for iterative solver"""

    def __init__(self, BEMobj=[], Connection=[], Intersection=[],plot_mesh=1):
        """Init the multidomain problem by given the BEMobj and connection table
           Assume a non-conforming mesh are given
    
        BEMobj idx example:
        ----------------
        |  0 | 1 |  2  |
        |    |   |     |
         ---------------
        Face idx example:
            2
          ------
          |    | 
        3 |    | 1
          -----
            0
        
        Format:
        BEMobj=[BEM_Case1,BEM_Case2,BEM_Case3]
        Intersects=[[Intersect1],[Intersect2]...]
        Coonection=[BEMobj1-(Connected BEMobj,Connected edge idx)..,
                    BEMojb2-[(connected idx,edge idx),(connected idx,edge idx)]

        Arguments
        ---------
        BEMobj     -- List of involved sub-BEM2D problems e.g. [BEMobj1,2,3...]
        Intersects -- List of intersect's two end coords [(pts_a,pts_b),(pts_a,pts_b)....]
        Connect    -- Connect table between sub-BEM2D problems [(Connected BEMobj,Connected edge idx)..]
                      The default order is [BEMojb index]                    (BEMobj1)....
        Method     -- Itertive coupling method (P-DD,P-NN,P-RR,S-DN)
        DFN        -- 1 coupling between fractures, 0 coupling between domain boundary
        
        Numobj     -- Number of subdomains
        NumInt     -- NUmber of intersections(Interfaces)
        error_abs  -- releative error at each iteration step
        
        Author:Bin Wang (binwang.0213@gmail.com)
        Date: July. 2017
        """
        #[Input Parameters]
        self.BEMobjs=BEMobj[:] #copy the list, [list]=[list] just passed the reference
        self.Intersects=Intersection
        self.Connect=Connection
        self.TraceOn=0
        
        self.Method="CG"
        #[Derived Parameters]
        self.NumObj=len(BEMobj)       #Num of objects
        self.NumInt=len(Intersection)       #Num of intersected edges
        
        #[Plot]
        self.error_abs=[]  #abs error at interface,np.sum(abs(Q_new-Q_old)+abs(P_current-P_connect))
        
        if(plot_mesh): self.plot_mesh()
        
    
    ####Main fucntions####
    def Solve_Iter(self,Method="CG",initial_guess=0.0,TOL=1e-5,max_iters=100,alpha=0.1,opt=0):
        """Solve a multi-domain problem using a specific method
           Reference: Section 3 in the reference paper

           Support for:
           1. Parallel Neumann-Neumann method
           2. Parallel Dirichlet-Dirichlet method
           3. Parallel Robin-Robin method
           4. Sequential Dirichlet-Neumann method
           5. Classic Parallel Robin-Robin
           
           Following acclerating technqiue supported:
           1. Dynamic Relaxation method
           2. Aitken method [future]
           3. CG method [future]
        
        Arguments
        ---------
        TOL   -- Convergence tolerance(1e-5,1e-6...)
        alpha -- Initial relaxation paramters at step 0
        opt   -- switcher between static relaxation parameter(0) and dynamic relaxation parameter(1)
        
        Author:Bin Wang(binwang.0213@gmail.com)
        Date: July. 2017
        """
        
        self.Method=Method
        
        if(self.TraceOn == 1):
            print("DDM is not support internal Trace now!")
            #return

        #Parallel Neumann-Neumann method with dynamic relaxation paramters
        if(Method=="P-NN"):
            PNN(self,alpha,TOL,max_iters,opt)
        #Parallel Dirichlet-Dirichlet method with dynamic relaxation paramters
        if(Method=='P-DD'):
            PDD(self,alpha,TOL,max_iters,opt)
        #Sequential Dirichlet-Neumann method with dynamic relaxation paramters
        if(Method=='S-DN'):
            print('[Warnning] This method is not applicable for new input format')
            #SDN(self,alpha,TOL,opt)
        #Parallel Robin-Robin method with dynamic relaxation paramters
        if(Method=='P-RR'):
            PRR(self,alpha,1.0,TOL,max_iters,opt)
        #Berrone's Conguate gradient method [not available now]
        if(Method=="CG"):
            assert 3==1, 'This method is not available'
            #self.CG_loop(TOL)

    ####Plot functions####
    def plot_mesh(self):
        """Plot the overall BEM mesh and domain geometry

        Each domain can have different color by modifing line:
        
        plt.plot(np.append([BE.xa for BE in BEMobj.BEs_edge], BEMobj.BEs_edge[0].xa), 
             np.append([BE.ya for BE in BEMobj.BEs_edge], BEMobj.BEs_edge[0].ya), 
             ,color=c,markersize=5)

        plt.scatter([BE.xc for BE in BEMobj.BEs_edge], 
                            [BE.yc for BE in BEMobj.BEs_edge], color=c, s=25)

        Author:Bin Wang(binwang.0213@gmail.com)
        Date: July. 2017
        """

        if(self.BEMobjs[0].TraceOn==1):
            self.TraceOn=1
            print('Plot Mesh only support non-DFN type!')
            return

        plt.figure(figsize=(6, 6))
        plt.axes().set_aspect('equal')
        plt.title('BEM Mesh',fontsize=15)
        plt.xlabel('x(m)')
        plt.ylabel('y(m)')

        #plt.xticks(np.arange(-3, 9, 1))
        #plt.yticks(np.arange(-2, 5,1))

        #plt.grid(b=True, which='major', color='k', linestyle=':')
        #plt.grid(b=True, which='minor', color='k', linestyle='--', alpha=0.2)
        #plt.minorticks_on()

        Total_Ele=0
        
        color=cm.jet(np.linspace(0,1,self.NumObj+1))
        for i,c in zip(range(self.NumObj),color):#For each subdomain
            BEMobj=self.BEMobjs[i]
            
            #Boundary elements
            plt.plot(np.append([BE.xa for BE in BEMobj.BEs_edge], BEMobj.BEs_edge[0].xa), 
             np.append([BE.ya for BE in BEMobj.BEs_edge], BEMobj.BEs_edge[0].ya), 
             'bo-',markersize=5)
            #Trace elements
            if (self.TraceOn):
                for i in range(BEMobj.Num_trace):
                    plt.plot([BEMobj.BEs_trace[i][0].xc, BEMobj.BEs_trace[i][0].xc],
                         [BEMobj.BEs_trace[i][0].yc, BEMobj.BEs_trace[i][0].yc],
                         'go-',  markersize=5)
                    plt.plot([BEMobj.BEs_trace[i][0].xa, BEMobj.BEs_trace[i][0].xb],
                         [BEMobj.BEs_trace[i][0].ya, BEMobj.BEs_trace[i][0].yb],
                         'go-',  markersize=2)
                    for j in range(BEMobj.NumE_t[i] - 1):
                            plt.plot([BEMobj.BEs_trace[i][j].xa, BEMobj.BEs_trace[i][j + 1].xb],
                                     [BEMobj.BEs_trace[i][j].ya, BEMobj.BEs_trace[i][j + 1].yb],
                                     'go-', markersize=2)
                            plt.plot([BEMobj.BEs_trace[i][j].xc, BEMobj.BEs_trace[i][j + 1].xc],
                                     [BEMobj.BEs_trace[i][j].yc, BEMobj.BEs_trace[i][j + 1].yc],
                                     'go-', markersize=5)
            
            if (BEMobj.BEs_edge[0].element_type=="Quad"):
                plt.scatter([BE.xc for BE in BEMobj.BEs_edge], 
                            [BE.yc for BE in BEMobj.BEs_edge], color='b', s=25)

            Total_Ele+=len(BEMobj.BEs_edge)
            
        #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.tight_layout()
        #plt.savefig('Couplded_Fig.png', format='png', dpi=300)
        
        print('-----Mesh Info-----')
        print('Total Number of Elements:',Total_Ele)
        print("Total Number of Subdomains:",self.NumObj)
        plt.show()
    
    def plot_Convergence(self):
        """Plot the convergence plots in log-log scale and standard scale

        Author:Bin Wang(binwang.0213@gmail.com)
        Date: July. 2017
        """
        Y=np.array(self.error_abs)
        X=[i+1 for i in range(len(Y))]
        
        
        fig, axes = plt.subplots(nrows=2,figsize=(7, 7))
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        for i, ax in enumerate(axes.flat):
            if i==1:
                ax.set_title(r'Convergence Plot (log-log)',fontsize=15)
                ax.set_xlabel('Iteration',fontsize=15)
                ax.set_ylabel('Error',fontsize=15)
                im=ax.plot(X, Y,'ro-',markersize=5)#log scale
                divider = make_axes_locatable(ax)
            if i==0:
                ax.set_title(r'Convergence Plot',fontsize=15)
                ax.set_xlabel('Iteration',fontsize=15)
                ax.set_ylabel('Error(log)',fontsize=15)
                im=ax.semilogy(X, Y,'bo-',markersize=5)#log scale
                divider = make_axes_locatable(ax)
        fig.tight_layout()
        plt.show()
        
        return Y

    ####Additional auxilary functions####
    def Interp_intersection(self,objID,ConnectObjID,Intersect,Vals=[],varID=0):
        #! This function will not support in new version
        """Variables interpolation at the intersection between two domains with non-conforming mesh
           Namely, project variables of connected domain to current domain
           Reference: Eq. 11 and Figure 4 in the reference paper

           This function called when calculating flux(pressure) jump at interface: 
           (1) P_current-P_connect at P-NN,S-DN,P-RR
           (2) Q_current+Q_connect at P-RR,P-DD
           (2) dlth_current-dlth_connect at Eq.8 CG
           (3) h_current-h_connect at Eq. 6 CG
           all above three cases, Interpoalting variabls are calcualted from BEM matrix
           (4) dltq_current+dltq_connect at Eq.8 CG
           Expecially, dltq comes from other variables
        
        Interp_Nonconforming[Kernel interpolation using shape function] in Lib.Tools.Geometry
        
        Arguments
        ---------
        objID        -- Index of BEMobj at current domain
        ConnectobjID -- Index of BEMobj at connect domain
        Intersect    -- Coords of intersection between two connected domains
        Val_connect    -- Variabls at intersection nodes on connect domain
                        which can be obtained from BEM solution or other variables
        Vals         -- P(Q)_connect from other variabls [Input], used for dltq
        varID        -- 0 is P_connect   1 is Q_connect
        order        -- shape function order
        
        Author:Bin Wang(binwang.0213@gmail.com)
        Date: July. 2017
        """
        #Variables interpolation between two interface with non-conforming mesh
        #assert 3==1,'Non-conforming mesh is disabled! Please update for discontinous element'

        bdID=self.BEMobjs[objID].Mesh.EndPoint2bdmarker(Intersect[0],Intersect[1])
        CurrentNodes=self.BEMobjs[objID].Mesh.mesh_nodes[bdID]
        bdID_connect=self.BEMobjs[ConnectObjID].Mesh.EndPoint2bdmarker(Intersect[0],Intersect[1])
        
        if(len(Vals)==0):#Used for general variable(dlth,h,p) by solving matrix Eqs.4-6
            Val_connect=self.BEMobjs[ConnectObjID].PostProcess.get_BDSolution(bdID_connect)[varID]
        else:#Only Used for dltq
            Val_connect=Vals
        
        ConnectNodes=self.BEMobjs[ConnectObjID].Mesh.mesh_nodes[bdID_connect]
        if(self.BEMobjs[ConnectObjID].TypeE_edge=="Quad"): 
            Val_connect=Interp_Nonconforming(CurrentNodes,ConnectNodes,Val_connect,order=2)
        if(self.BEMobjs[ConnectObjID].TypeE_edge=="Linear"): 
            Val_connect=Interp_Nonconforming(CurrentNodes,ConnectNodes,Val_connect,order=1)
        if(self.BEMobjs[ConnectObjID].TypeE_edge=="Const"):
            Val_connect=Interp_Nonconforming(CurrentNodes,ConnectNodes,Val_connect,order=0)#Constant element
        
        return Val_connect
    
    def new_var(self):
        #! This function will not support in new version
        """Create a new variables with respect to iterative solver object
        Default Structure: Var[NumSubdomain][NumInterface]

        Author:Bin Wang(binwang.0213@gmail.com)
        Date: July. 2017
        """

        var=[[] for i in range(self.NumObj)]
        for i in range(self.NumObj):#For each subdomain
                Num_shared_edge=len(self.Connect[i])
                #CurrentBEM=self.BEMobjs[i]
                for j in range(Num_shared_edge):#For each connected edge in this domain
                    var[i].append(0.0)

        return var
    
    def get_ConnectID(self,objID,Intersect):
        #! This function will not support in new version
        """Find the intersection id in a domain(objID) based on Intersection Coords(Intersect)

        Arguments
        ---------
        objID       -- Subdomain id in self.BEMobjs
        Intersect   -- Interface coordinates, e.g (0.1,0.2)

        Output
        ---------
        Intersect id in the self.Connect list

        Author:Bin Wang(binwang.0213@gmail.com)
        Date: July. 2017
        """
        
        debug=0
        Num_shared_edge=len(self.Connect[objID])
        
        Pts_c=[(Intersect[0][0]+Intersect[1][0])/2,(Intersect[0][1]+Intersect[1][1])/2]
        
        if(debug): print('Pts_c of query edge',Pts_c)
        for j in range(Num_shared_edge):
            IntersectID=self.Connect[objID][j][1]
            if(debug): print("Domain",objID+1,j+1,"-th Intersection")
            if(debug): print("End Nodes",self.Intersects[IntersectID][0],self.Intersects[IntersectID][1])
            if(point_on_line(Pts_c,self.Intersects[IntersectID][0],self.Intersects[IntersectID][1])):
                return j
        
        print("No match Intersection Found!")
    
    
    
    
