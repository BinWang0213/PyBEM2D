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

import numpy as np

#[BEM Element type]
from ...Tools.Geometry import calcDist, point_on_line


###############################
#
#  General BEM 2D Element Class
#
###############################


class Source_element(object):
    """Class object for a point source element"""
    def __init__(self, Pts=(0,0),bd_marker=0):
        self.x0, self.y0 = Pts[0], Pts[1] 
        self.ndof=1

        #Solution Variables
        self.bd_Indicator=1    # Neumann-1   Dirichlet-0
        self.bd_marker=bd_marker
        self.bd_values = [0.0] #Default 0 flux Neumann BC
        self.P = [0.0] 
        self.Q = [0.0] 
        self.u = [0.0] 
        self.v = [0.0] 

    def __repr__(self):
        s = "Element Type=" + "source element\n"
        s +="Num of DOF="+str(self.ndof)+"\n"
        s += "Boundary Marker ID=" + str(self.bd_marker) + "\n"
        s += "Pts=(" + str(self.x0) + "," + str(self.y0) + ") "
        s += "BD_Vals" + str(self.bd_values) + ") \n"
        s += "P" + str(self.P[0]) + ") \n"
        s += "Q" + str(self.Q[0]) + ") \n"
        s += "U" + str(self.u[0]) + ") \n"
        s += "V" + str(self.v[0]) + ") \n"
        return s
    
    def set_BC(self,bd_Indicator,bd_value):
        #set up boundary condition for a element
        self.bd_Indicator=bd_Indicator
        self.bd_values=[bd_value]

    def get_node(self):
        #get a node for a element
        return self.x0,self.y0
    
    def get_bdvals(self):
        return self.bd_values
    
    def set_PQ(self,P,Q):
        #set solution values
        self.P=P
        self.Q=Q
    def get_P(self):
        return self.P
    def get_Q(self):
        return self.Q


class BEM_element(object):
    """Class object for a single boundary element"""
    def __init__(self, Pts_a=(0,0),Pts_c=(0,0),Pts_b=(0,0),Type="quad",bd_marker=0,tol=1e-7):
        """Creates a boundary element from A->B.
           The position fo A,B decide the element direction
        
        Boundary Element Type:

        [Continuous Element] d=1.0
        Type        Independent-Node    Sharded-Node    Continuous?      a         c        b    
        Constant        Pts_c(1)              -              No          |--------|*|--------|
        Linear          Pts_a(1)           Pts_b(1)          Yes         |*|---------------|*|
        Quadrature      Pts_a,Pts_c(2)     Pts_b(1)          Yes         |*|------|*|------|*|

        [Discontinuous Element] d=[0,1]
        Linear          Pts_a',Pts_b'(2)      -              No          |---|*|-------|*|---|
        Quadrature      Pts_a',Pts_c',Pts_b'(3)  -           No          |---|*|--|*|--|*|---|

        Arguments
        ---------
        xa, ya    -- Cartesian coordinates of the first end-point A.
        xb, yb    -- Cartesian coordinates of the second end-point B.
        xc, yc    -- Cartesian coordinates of the center point.
        length    -- length of this BE.
        nx,ny     -- unit normal vector (available for linear line element, curve element requires jacobian)
        tx,ty     -- unit tagential vector
        
        bd_Indicator   -- 1 for Neumann B.C and Robin ....0 for Dirchelet B.C
        bd_value  -- boundary value(up to 3)
        Type      -- boundary element method type
                     [Const] [Linear] [Quad]
        bd_marker -- Classify boundary as index for BC
        
        P         -- Pressure on boundary element node
        Q         -- flux(pressure derivate in normal direction), u=dp/dx v=dp/dy  Q=nx*dp/dx+ny*dp/dy
        
        Author:Bin Wang (binwang.0213@gmail.com)
        Date: July. 2017
        """
        
        #[Geometry]
        self.tol=tol #! Geometry calculation tolerance
        self.xa, self.ya = Pts_a[0], Pts_a[1]  #First node for a quadratic BE
        self.xb, self.yb = Pts_b[0], Pts_b[1]  #Third node for a quadratic BE
        self.xc,self.yc=0,0
        if (Type=="Const" or Type=="Linear"): 
            self.xc, self.yc=(self.xa+self.xb)/2, (self.ya+self.yb)/2
        if (Type=="Quad"):  
            self.xc, self.yc = Pts_c[0], Pts_c[1]  #Second node for a quadratic BE 
        self.length = calcDist(Pts_a,Pts_b)     # length of the element   
        
        self.nx=(self.yb-self.ya)/self.length   # Normal unit Vector of the element nx*i+ny*j
        self.ny=-(self.xb-self.xa)/self.length  # 
        self.tx=-self.ny #Tagential unit vector tx*i+ty*j
        self.ty=self.nx
        
        self.d=1.0 #discontionus element node offset from Pts_c to Pts_a
        if(Type=="Linear"): self.d=0.5
        elif(Type=="Quad"): self.d=2/3
        
        self.ndof=0
        
        #[Properties]
        self.element_type=Type
        self.bd_marker=bd_marker
        self.Discontinuous=False #Discontinuous element for linear and quad element
        
        #[Boundary Conditions & Values]
        self.bd_Indicator=1                            # Neumann-1   Dirichlet-0
        self.Robin_alpha=0                             # Robin is a special of Neumann, bd_indicator still use 1

        #[Solution Variables]
        if (self.element_type=="Const"):
            self.ndof=1
        
        if (self.element_type=="Linear"):
            self.ndof=2
        
        if (self.element_type=="Quad"):
            self.ndof=3

        #Solution Variables
        self.bd_values = [0.0] * self.ndof #Default 0 flux Neumann BC
        self.P = [0.0] * self.ndof
        self.Q = [0.0] * self.ndof
        self.u = [0.0] * self.ndof
        self.v = [0.0] * self.ndof
            
    def __repr__(self):
        s = "Element Type=" + self.element_type + " element\n"
        s +="Num of DOF="+str(self.ndof)+"\n"
        s += "Boundary Marker ID=" + str(self.bd_marker) + "\n"
        s += "Pts=(" + str(self.xa) + "," + str(self.ya) + ") "
        s += "(" + str(self.xc) + "," + str(self.yc) + ") "
        s += "(" + str(self.xb) + "," + str(self.yb) + ") \n"
        s += "BD_Vals" + str(self.bd_values) + ") \n"
        s += "P" + str(self.get_P()) + ") \n"
        s += "Q" + str(self.get_Q()) + ") \n"
        s += "U" + str(self.get_U()) + ") \n"
        s += "V" + str(self.get_V()) + ") \n"
        return s

    def get_bdvals(self):
        #Get the boudary condtions values into array
        return self.bd_values
    
    def set_bdvals(self,val):
        #Set the boudary condtions values into array
        self.bd_values=[val]*self.ndof

    def set_PQ(self,P,Q):
        #set the vector P,Q and gradient to internal storage
        self.P=P
        self.Q=Q

    def get_PQ(self):        
        return self.P, self.Q     

    def get_P(self):
        return self.P
    
    def get_Q(self):
        return self.Q

    def get_node(self,nodeid):
        #Get the coords of the local node
        
        if (self.element_type == "Const"):
            local=0
        if (self.element_type == "Linear"):
            if(nodeid == 0):
                local = -self.d
            if(nodeid == 1):
                local = self.d
        if(self.element_type == "Quad"):
            if(nodeid==0):
                local=-self.d
            if(nodeid==1):
                local=0
            if(nodeid==2):
                local=self.d

        N = [0.5 * (1 - local), 0.5 * (1 + local)]
        
        Pts_a=(self.xa,self.ya)
        Pts_b=(self.xb,self.yb)

        return N[0]*Pts_a[0]+N[1]*Pts_b[0],N[0]*Pts_a[1]+N[1]*Pts_b[1]

    def get_nodes(self):
        Pts=[]
        if (self.element_type == "Const"):
            Pts.append(self.get_node(0))
        if (self.element_type == "Linear"):
            Pts.append(self.get_node(0))
            Pts.append(self.get_node(1))
        if(self.element_type == "Quad"):
            Pts.append(self.get_node(0))
            Pts.append(self.get_node(1))
            Pts.append(self.get_node(2))
        return Pts        

    def eval_P(self,Pts):
        #calculate the p on a specific element using shape function interpolation
        phi = self.get_ShapeFunc(Pts)
        P,Q=self.get_PQ()

        p = 0.0
        for i in range(self.ndof):
            p += phi[i] * P[i]

        return p

    def eval_UV(self,k_tensor):
        #This function is no longer used
        #Evaluate flux gradient in x and y direction at the nodes
        #P,Q must be completed first by set_PQ
        
        #Special Case of element Flux=0
        dist = 1e-14
        unit_vector = np.array((self.tx, self.ty))

        Nodes=self.get_nodes()

        for i in range(self.ndof):
            Pts=Nodes[i]
            Pts_diff = Pts + dist * unit_vector
            if(self.nx==0.0):#Horizontal line
                Qt = (self.eval_P(Pts_diff) - self.eval_P(Pts)) / dist
                dpdx = Qt * self.tx
                dpdy = Qt * self.ty
            else:#verticle line or Slant line 
                Qn = self.Q[i]
                dpdx = Qn * self.nx
                dpdy = Qn * self.ny
            self.u[i] = dpdx * k_tensor[0] + dpdy * k_tensor[1]
            self.v[i] = dpdx * k_tensor[1] + dpdy * k_tensor[2]


    def eval_UV2(self,k_tensor):
        #This function is no longer used
        #Assuming Element is linear stright line, but variables can be quadratic
        #Not complete, to be finished in the furture
        JacInvY=2/(self.yb-self.ya)
        JacInvX = 2 / (self.xb - self.xa)

        Pts=self.get_nodes()

        BX = JacInvX * self.get_DerivShapeFunc(Pts[0])
        BY = JacInvY * self.get_DerivShapeFunc(Pts[1])

        P=np.array(self.get_P())
        dpdx=np.dot(BX,P)
        print(BX,dpdx)

    def set_UV(self,U,V):
        self.u=U
        self.v=V

    def get_U(self):
        return self.u
    
    def get_V(self):
        return self.v

    def set_BC(self,bd_Indicator,bd_value,Robin_a=1,mode=0):
        #set up boundary condition for a element
        #mode-0 constant value along a element
        #mode-1 assign (1,2,3) values for a element

        self.bd_Indicator=bd_Indicator
        if(bd_Indicator==2):#Robin boundary conditions
            self.bd_Indicator=1
            self.Robin_alpha=Robin_a

        if(mode==0):
            self.bd_values = [bd_value] * self.ndof
        if(mode==1):
            self.bd_values=bd_value

    def get_Jac(self):
        #Jacobian of the element
        return self.length/2.0

    def get_ShapeFunc(self,Pts=None,local=None):
        #Interpolate the solution on the 1D boundary element
        #order=0-constant,1-linear,2-quadratic
        #Pts-Query point Ele_Pts-Element node points
    
        phi=[] #weights of shape function
        
        if(local is None):
            local = self.get_LocalGeometricCoord(Pts)

        if(self.element_type=="Linear"):#Two nodes
            phi.append(0.5*(1-local/self.d))
            phi.append(0.5*(1+local/self.d))
        elif(self.element_type=="Quad"):#Three nodes
            phi.append(0.5*local/self.d*(local/self.d-1))
            phi.append((1 - local / self.d) * (1 + local / self.d))
            phi.append(0.5*local/self.d*(local/self.d+1))
        else:#Const ele
            phi.append(1)
        
        return np.array(phi)

    def get_DerivShapeFunc(self,Pts=None,local=None):
        
        dphi = []  # weights of shape function

        if(local is None):
            local = self.get_LocalGeometricCoord(Pts)

        if(self.element_type == "Linear"):  # Two nodes
            dphi.append(-1/2/self.d)
            dphi.append(1/2/self.d)
        if(self.element_type == "Quad"):
            dphi.append(local/self.d/self.d-1/2/self.d)
            dphi.append(-2*local/self.d/self.d)
            dphi.append(local/self.d/self.d+1/2/self.d)
        else:
            dphi.append(0)
        
        return np.array(dphi)

    def get_LocalGeometricCoord(self, Pts):
        #Get the local coordinats (-1,1) for a Pts on the element
        #Assuming Geometric a stright line
        Pts_a = (self.xa, self.ya)
        Pts_b = (self.xb, self.yb)
        return -1 + 2 * calcDist(Pts, Pts_a) / calcDist(Pts_a, Pts_b)

    def isPtsOnElement(self, xi, yi):
        #Test of a point is on the element
        return point_on_line((xi, yi), (self.xa, self.ya), (self.xb, self.yb),self.tol)

    def get_InnerPoints(self,xc,yc,rab=0.00003):
        #When the Neumann BC = 0, the pressure/velocity component can be obtained the close inner point
        #rab the inner distance 

        x_a = xc - rab / 2 * (self.ya - self.yb) / self.length
        y_a = yc - rab / 2 * (self.xb - self.xa) / self.length
        x_b = xc + rab / 2 * (self.ya - self.yb) / self.length
        y_b = yc + rab / 2 * (self.xb - self.xa) / self.length
        Ptsa=[x_a,y_a]
        Ptsb=[x_b,y_b]
        return Ptsa,Ptsb

    def isNoFlow(self):
        #Determine this is no flow boundary condition
        if(np.sum(self.Q)<1e-15):
            return True
        return False

    def reset_Element(self):
        #set the intial condition and solution in default model
        #[Boundary Conditions & Values]
        self.bd_Indicator=1                           
        self.bd_values = [0.0] * self.ndof  # Default 0 flux Neumann BC
        self.P = [0.0] * self.ndof
        self.Q = [0.0] * self.ndof
        self.u = [0.0] * self.ndof
        self.v = [0.0] * self.ndof

        
    






