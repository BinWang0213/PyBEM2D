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
-BEM_Elements.py        #BEM Element main class, const, linear, quad element, etc
-Assembler.py           #BEM matrix assembler and solution calculation
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

#[BEM Elements]
from .Elements.BEM_Elements import BEM_element
from .Elements.Assembler import build_matrix_all,solution_allocate_all,Field_Solve_all

#[BEM Mesh]
from .BEM_2D_Mesh import BEM_2DMesh

#[BEM Postprocessing]
from .BEM_2D_Postprocessing import BEM_2DPostprocess

#[General Geometry Lib]
from ..Tools.Geometry import *

###############################
#
#  Core BEM 2D Class
#
###############################

class BEM2D:
    """Contains information and functions related to a 2D potential problem using BEM."""
        
    def __init__(self):
        """Creates a BEM objecti with some specific paramters
        
        Arguments
        ---------
        [BEM Mesh]
        Mesh

        [BEM Solver]
        TraceOn         -- Enable the internal (geometry) trace
        BEs_edge        -- BEM element collection for boundary edge
        BEs_trace       -- BEM element collection for internal edge
        
        Author:Bin Wang(binwang.0213@gmail.com)
        Date: July. 2017
        """

        self.Mesh = BEM_2DMesh(self) #Link to mesh
        self.PostProcess = BEM_2DPostprocess(self.Mesh,self) #Link to postprocess

        self.BEs_edge = []
        self.BEs_trace = []
        self.BEs_source = []
        
        self.TraceOn=0

        #Physics properties
        self.k_coeff=1.0/2/np.pi
        self.h=1.0       #2D plane thickness, specially used for Darcy flow in plane
        self.k = np.array([1.0,0.0,1.0]) #Anistropic permeability[k11,k12,k22]
        self.miu=1.0     #fluid viscosity

        #BEM element order
        self.TypeE_edge="Quad"
        self.TypeE_trace="Const"

        #Nice reference of BC https://www.comsol.com/blogs/how-to-make-boundary-conditions-conditional-in-your-simulation/
        self.NeumannBC=[]
        self.DirichletBC=[]
        self.RobinBC=[]
     
    def set_Mesh(self, Pts_e=[], Pts_t=[], Pts_s=[], h_edge=0.1, h_trace=0.1, Ne_edge=None, Ne_trace=None, Type='Quad', mode=0,geo_check=True):
        """Create BEM mesh based on either number of element or length of element
        
        Arguments
        ---------
        BEs_edge        -- BEM element collection for boundary edge
        BEs_trace       -- BEM element collection for internal edge
        h_edge    -- Length of element for boundary edge [optional]
        h_trace   -- Length of element for trace [optional]
        Ne_edge     -- Number of element on all edges
        Ne_trace    -- Number of element on all traces
        
        Author:Bin Wang(binwang.0213@gmail.com)
        Date: July. 2017
        """
        self.Mesh.set_Mesh(Pts_e,Pts_t,Pts_s,h_edge,h_trace,Ne_edge,Ne_trace,Type,mode,geo_check)

        if(self.Mesh.Ndof_trace == 0):
            self.TraceOn = 0
        else:
            self.TraceOn = 1

    def plot_Mesh(self, Annotation=1,legend=1,node_size=5,scale=1.0,img_fname=None):
        """Plot BEM Mesh

        Author:Bin Wang(binwang.0213@gmail.com)
        Date: July. 2017
        """
        self.Mesh.plot_Mesh(Annotation,legend,node_size,scale,img_fname)

    ###########Boundary Conditions Module#################
    def set_BoundaryCondition(self,DirichletBC=[],NeumannBC=[],RobinBC=[],update=0,mode=0,Robin_a=1,debug=1):
        """Set up boundary conditions for generated bem mesh using element marker
           Dirichlet or Neumann boundary can be applied for each edge or trace
        
        Input format:
            Dirichelt[(element marker,BC_value)]
            BC_value
        1. The default BC is no-flow Neumann boundary condition
        2. Support for const boundary condition assignment(mode-0)
        3. Support for non-constant(node-wise) boundary condition assignment(mode-1)
        4. mode 1 currently only support for boundary edge
           
        Arguments
        ---------
        DirichletBC   -- Dirichlet boundary condition for a specifc edge, id=0
        NeumannBC     -- Neumann boundary condition for a specific edge, id=1
        RobinBC       -- Robin boundary condition for a specific edge, id=2
        update        -- model selection, 0-set up new BCs 1-update old BCs, normally used for DDM
        mode          -- 0 constant BC assignmetn  
                         1 node-wise BC assignment, normally used for DDM
                         func spatial function-wise BC assignment, such as def func(x,y): return P=sin(pi*x)
        Robin_a       -- Robin coefficient
        
        Author:Bin Wang(binwang.0213@gmail.com)
        Date: July. 2017
        """

        if(debug): print("[Boundary Condition] BCs set up")

        #Reset all element's BC as no-flow BC
        if(update==0):
            #Store BC as original BC when new BC created
            self.DirichletBC=DirichletBC
            self.NeumannBC=NeumannBC
            self.RobinBC=RobinBC
            for i in range(len(self.BEs_edge)):
                self.BEs_edge[i].reset_Element()
            for i in range(len(self.BEs_trace)):
                for j in range(len(self.BEs_trace[i])):
                    self.BEs_trace[i][j].reset_Element()

        #Loop for three boundary conditions
        for BCid in range(3):
            BCs=[]
            if(BCid==0 and len(DirichletBC)!=0):
                if(debug): print("[Boundary Condition] Dirichlet BC @", DirichletBC)
                BCs=DirichletBC
            if(BCid==1 and len(NeumannBC)!=0):
                if(debug): print("[Boundary Condition] Neumann BC @", NeumannBC)
                BCs=NeumannBC
            if(BCid==2 and len(RobinBC)!=0):
                if(debug): print("[Boundary Condition] Robin BC @", RobinBC)
                BCs=RobinBC
            
            for i in range(len(BCs)):
                bd_markerID=BCs[i][0]
                elementID=self.Mesh.bdmarker2element(bd_markerID)
                for j in range(len(elementID)):

                    #Edge BC
                    if(bd_markerID<self.Mesh.TraceStartID): #This is a edge
                        eID=elementID[j]
                        if(mode==0):
                            self.BEs_edge[eID].set_BC(BCid,BCs[i][1],Robin_a) # Neumann-1   Dirichlet-0 Robin-2
                        elif(mode==1):
                            bd_values=self.Mesh.bd2element(self.TypeE_edge,eleid=j,node_values=BCs[i][1])
                            self.BEs_edge[eID].set_BC(BCid,bd_values,Robin_a,mode=1)
                        elif(mode=='func'):
                            Pts=self.Mesh.getEdgeEleNodeCoords(eID)
                            func = BCs[i][1]
                            bd_values = [func(xy) for xy in Pts]
                            self.BEs_edge[eID].set_BC(BCid,bd_values,Robin_a,mode=1)
                    
                    #Trace BC
                    if(bd_markerID>=self.Mesh.TraceStartID and bd_markerID<self.Mesh.SourceStartID):#this is a trace
                        tID=elementID[j][0]
                        eID=elementID[j][1]
                        #Special case of flux constrain for the first time BC setup
                        if(update==0 and BCid>=1): #Flux should be average flux strength per unit length                           
                            average_Q = BCs[i][1] / len(self.BEs_trace[tID])
                            length_trace = calcDist(self.Mesh.Pts_t[tID][0], self.Mesh.Pts_t[tID][1])
                            average_Q = average_Q * length_trace
                            #average_Q = BCs[i][1] / length_trace
                            # Neumann-1   Dirichlet-0 Robin-2
                            self.BEs_trace[tID][eID].set_BC(BCid, average_Q, Robin_a)
                        #General case with Dirichlet
                        else:
                            if(mode==0):
                                self.BEs_trace[tID][eID].set_BC(BCid,BCs[i][1],Robin_a) # Neumann-1   Dirichlet-0 Robin-2
                            elif(mode==1):                                    
                                bd_values=self.Mesh.bd2element(self.TypeE_trace,eleid=j,node_values=BCs[i][1])                                
                                self.BEs_trace[tID][eID].set_BC(BCid,bd_values,Robin_a,mode=1)
                    
                    #Source BC
                    if(bd_markerID>self.Mesh.SourceStartID-1):#This is a source 
                        eID=elementID[j]
                        self.BEs_source[eID].set_BC(BCid,BCs[i][1])
        
    def SetBDBoundaryConditionValue(self,bd_markerID,bcval):
        '''Set the boundary conditions value for a specific edge
        bcval is a scalar
        '''
        elementID=self.Mesh.bdmarker2element(bd_markerID)#find the element idx on this edge
        if (bd_markerID > self.Mesh.Num_boundary - 1):  # this is a trace
            TracerID = elementID[0][0]
            for ei, pl in enumerate(self.BEs_trace[TracerID]):
                pl.set_bdvals(bcval)
        else:  # This is boundary edge
            for i in range(len(elementID)):  # loop for all elements on this edge
                self.BEs_edge[elementID[i]].set_bdvals(bcval)

    def SetProps(self,k_tensor=[],k=1.0,miu=1.0,h=1.0):
        '''Set the potential properties

        viscosity and fracture aperature is converted into equivalent permeability

        Author:Bin Wang(binwang.0213@gmail.com)
        Date: July. 2018
        '''
        if(len(k_tensor)==0):#Isotropic k 
            k_tensor=np.array([k,0.0,k])
        else:
            assert k_tensor[1]==0, "[Error] Current version doesn't support full tensor!"
        
        #Anistropic K
        self.k = np.array(k_tensor)
        self.k=self.k/miu # mu/k/h = 1/(k*h/u)

        k_det = self.k[0] * self.k[2] - self.k[1] * self.k[1]
        if(k_det<0): assert k_det>=0,'Incorrect K tensor, k11*k22-k12^2 must >0!!'        
        
        #Fluid flow problem
        self.miu = miu # fluid viscosity in Darcy problem
        self.h = h # plane thickness in Darcy problem
        #self.k_coeff = self.miu / self.h / np.sqrt(k_det) / 2 / np.pi #* Normalized k_coeff by set miu and h as 1
        self.k_coeff = 1.0 / 1.0 / np.sqrt(k_det) / 2 / np.pi #* Normalized k_coeff by set miu and h as 1\

    ###########Matrix Assemble and Solve Module#################
    def Solve(self,DDM=0,AB=[],debug=1):
        """Build up and solve BEM matrix using collocation method
        
        #Reference: BEM Introduction Course-1991-Brebbia,P81
        
        DDM--Domain Decomposion Solution, 1=DDM 0=general
        AB--Input HG matrix for DDM

        Author:Bin Wang(binwang.0213@gmail.com)
        Date: July. 2017
        """
        debug=0
        
        if(debug): print("[Solution] Assembling Matrix...")
        
        #Ab = build_matrix_trace(self.BEs_edge,self.BEs_trace, self.Mesh, DDM, AB)  # matrix AB
        Ab = build_matrix_all(self.BEs_edge,self.BEs_trace,self.BEs_source,self.Mesh, DDM, AB)
        if(debug): print("[Solution] Solving problem...", end='')
        X = np.linalg.solve(Ab[0], Ab[1])  # linear solution X
        if(debug): print("Done")
        # assign solution back to element class
        #solution_allocate_trace(self.BEs_edge, self.BEs_trace,self.Mesh, X, debug)
        solution_allocate_all(self.BEs_edge, self.BEs_trace,self.BEs_source,self.Mesh, X, debug)
        if(DDM==0): print("[Solution] #DOFs=",len(Ab[1]))

        return Ab

    def get_Solution(self, Pts, bd_markerID=-1):
        """Get the solution variable(p,ux,uy) at any given Point(Pts)
           Reference: BEM Introduction Course-1991
           Reference: A BEM solution of steady-state ﬂow problems in discrete fracture networks with minimization of core storage
        
        Arguments
        ---------
        Pts_location -- the element index which Pts belongs to (-1 internal, 1 element, 2 edge connection node)
        Pts          -- query points (x,y)
        bd_markerID  -- [optional] boundary marker id which helps to determine the solution on edge connection point

        Author:Bin Wang(binwang.0213@gmail.com)
        Date: July. 2017
        """
        Pts_Element,Pts_location = self.Mesh.point_on_element(Pts)  # check point location
        
        return Field_Solve_all(Pts[0], Pts[1], self.BEs_edge,self.BEs_trace,self.BEs_source,
        self.Mesh, elementLoc=Pts_Element,elementID=Pts_location)
            



