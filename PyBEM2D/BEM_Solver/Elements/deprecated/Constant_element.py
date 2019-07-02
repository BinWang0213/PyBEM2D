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
import math
import time
from Lib.Tools.Geometry import cosspace,Point2Segment,GaussLib,Global2Iso,Subdivision

from .Exact_Integration import Analytical_Intergration_cython


######################## Solver Module-Kernal Integration ########################
def GHCalc(xi,yi,panelj):
    #Gauss Quadrature intergration,n=10
    
    GI=[-0.97390652851717172008,-0.86506336668898451073,     
        -0.67940956829902440623,-0.43339539412942719080,     
        -0.14887433898163121089,+0.97390652851717172008,     
        +0.86506336668898451073,+0.67940956829902440623,     
        +0.43339539412942719080,+0.14887433898163121089]    #Gauss point
    WC=[0.06667134430868813759,0.14945134915058059315,       
        0.21908636251598204400,0.26926671930999635509,       
        0.29552422471475287017,0.06667134430868813759,       
        0.14945134915058059315,0.21908636251598204400,       
        0.26926671930999635509,0.29552422471475287017]     #weight cofficient
 
    GausOrder=10
        #Integration interval transoform
    Xj_G=np.empty(GausOrder) #Gauss point transfer
    Yj_G=np.empty(GausOrder) #Gauss point transfer
    Rij=0  #Distance from node to Gauss point on boundary element
    Rd1=0  #Radius derivatives 
    Rd2=0  #Raidus derivatives
    Rd=0   #Rd=nx*Rd1+ny*Rd2
    #Boundary solution variable
    Hij=0  #H matrix value-velocity
    Gij=0  #G matrix value-pressure
    Gij_dfn=0 #G matrix for DFN
    #Internal solution variable
    DU1=0  #G derivative in x direction
    DU2=0  #G derivative in y direction
    DQ1=0  #H derivative in x direction
    DQ2=0  #H derivative in y direction
    DQ1_dfn=0 #H for DFN
    DQ2_dfn=0 #H for DFN
    
    for j in range(GausOrder):
        Xj_G[j]=(panelj.xb-panelj.xa)*GI[j]/2+(panelj.xb+panelj.xa)/2
        Yj_G[j]=(panelj.yb-panelj.ya)*GI[j]/2+(panelj.yb+panelj.ya)/2
        Rij=np.sqrt((xi-Xj_G[j])**2+(yi-Yj_G[j])**2)
        Rd1=(Xj_G[j]-xi)/Rij
        Rd2=(Yj_G[j]-yi)/Rij
        Rd=Rd1*panelj.nx+Rd2*panelj.ny
        Hij=Hij-Rd/Rij*WC[j]*panelj.length/2
        Gij=Gij+np.log(1/Rij)*WC[j]*panelj.length/2
        
        Gij_dfn=Gij_dfn+np.log(1/Rij)*WC[j]/2
        
        DU1=DU1+Rd1/Rij*WC[j]*panelj.length/2
        DU2=DU2+Rd2/Rij*WC[j]*panelj.length/2
        DQ1=DQ1-(2*Rd1*Rd2*panelj.ny+(2*Rd1**2-1)*panelj.nx)/(Rij**2)*WC[j]*panelj.length/2
        DQ2=DQ2-(2*Rd1*Rd2*panelj.nx+(2*Rd2**2-1)*panelj.ny)/(Rij**2)*WC[j]*panelj.length/2
        
        DQ1_dfn=DQ1_dfn-(2*Rd1*Rd2*panelj.ny+(2*Rd1**2-1)*panelj.nx)/(Rij**2)*WC[j]
        DQ2_dfn=DQ2_dfn-(2*Rd1*Rd2*panelj.nx+(2*Rd2**2-1)*panelj.ny)/(Rij**2)*WC[j]

     #debug
   

    return Hij,Gij,DU1,DQ1,DU2,DQ2,Gij_dfn,DQ1_dfn,DQ2_dfn


######################## Solver Module-Matrix assemble and field point solve ########################
def build_matrix_const(panels, DDM=0, AB=[]):

    if(DDM == 1 and AB != 'none'):
        return update_matrix_const(panels, AB)

    N=len(panels)
    G_mat=np.empty((N, N), dtype=float)
    H_mat=np.empty((N, N), dtype=float)
    PI=3.141592653

    #Assemble G,H matrix
    for i, p_i in enumerate(panels): #nodes
        for j, p_j in enumerate(panels): #BEs
            if i==j:
                G_mat[i,j]=p_i.length*(np.log(2/p_i.length)+1)
                H_mat[i,j]=PI #Hii=1/2*2PI due to intergral=0
            if i!=j:
                xi,yi=p_i.xc,p_i.yc
                G,H,Gx,Gy,Hx,Hy=Analytical_Intergration_cython(xi,yi,p_j)
                H_mat[i,j]=H[0]
                G_mat[i,j]=G[0]
    
    H_origin=np.copy(H_mat)
    G_origin=np.copy(G_mat)

    
    #Boundary Condition enforcement
    #Assemble matrix A and reorder matrix H and G (Switching column)            
    for j in range(N):
        if (panels[j].bd_Indicator==0):#If boundary condition is Dirichlet then interchange G-H G*U=H*P
            for i in range(N):
                temp=H_mat[i,j]
                H_mat[i,j]=-G_mat[i,j]
                G_mat[i,j]=-temp
    
    #Robin Boundary Condition 
    #Reference https://www.researchgate.net/project/Extending-the-boundary-element-method-to-the-generalised-Robin-boundary-condition
    for j in range(N):
        for i in range(N):
            if (panels[j].bd_Indicator==1 and panels[j].Robin_alpha>0):#If boundary condition is Neumann then interchange G-H G*U=H*P
                H_mat[i,j]+=panels[j].Robin_alpha*G_mat[i,j]#beta*G
    
    print(H_mat)
    print(G_mat)
    
    #print(H_mat)
    #print(G_mat)
    A=H_mat
    #Assemble vector b
    b=np.empty(N, dtype=float)
    for i in range(N): #BE
        b[i]=0
        for j in range(N): #BEs
            b[i]=b[i]+G_mat[i,j]*panels[j].bd_value1
    
    print('Ab',A,b)

    return A, b, G  # ,G_origin,H_origin


def update_matrix_const(panels, AB=[]):

    N = len(panels)

    #Collecting prescribed BC values
    debug = 0
    A = AB[0]
    G = AB[1]

    #Assemble vector b
    b = np.empty(N, dtype=float)
    for i in range(N):  # BE
        b[i] = 0
        for j in range(N):  # BEs
            b[i] = b[i] + G[i, j] * panels[j].bd_value1
    
    return A, b, G

def solution_allocate_constant(panels,X,debug):
    if(debug): print("[Solution Results]")
    if(debug):print("Number of boundary elements:%s" % (len(panels)))
    if(debug):print("Coordinates&Boundary conditions of boundary elements:")
    if(debug):print("Point\tX\t\tY\t\tBD_P\t\tBD_Q")
    for i, p_i in enumerate(panels):
        if p_i.bd_Indicator==1: #if BE is a neumann boundary condition
            p_i.Q1=p_i.bd_value1
            p_i.P1=X[i]
            if(p_i.Robin_alpha>0):# This is a Robin boundary condition
                p_i.P1=X[i]
                p_i.Q1=(p_i.bd_value1-p_i.Robin_alpha*X[i])
        if p_i.bd_Indicator==0: #if BE is a dirichlet boundary condition
            p_i.P1=p_i.bd_value1
            p_i.Q1=X[i]
        if(debug):print("%s\t%s\t\t%s\t\t%s\t\t%s " % (i+1,p_i.xa,p_i.ya,p_i.P1,p_i.Q1))
    
    #Pressure derivate(u,v) on element nodes
    N=len(panels)
    for i in range(N):
        pl=panels[i]

        if(pl.Q1==0):#Zero-Neumann
            xc,yc=pl.xc,pl.yc
            temp=Field_Solve_constant(xc,yc,panels)
            pl.u1,pl.v1=temp[1],temp[2]
        else:#Non-zero Neumann
            pl.u1,pl.v1=pl.Q1*pl.nx,pl.Q1*pl.ny
    
def Field_Solve_constant(xi,yi,panels,elementID=-1):
    PI=3.141592653
    N=len(panels)
    p,u,v=0,0,0
    
    if (elementID==-1): #query point locate on internal domain
        for i, p_j in enumerate(panels):
            #puv=GHCalc(xi, yi, p_j)
            #puv=GHCalc_subdivision(xi, yi, p_j,'internal')
            G, H, Gx, Gy, Hx, Hy = Analytical_Intergration_cython(xi, yi, p_j)
            p = p + p_j.Q1 * G[0] - p_j.P1 * H[0]
            u = u + p_j.Q1 * Gx[0] - p_j.P1 * Hx[0]
            v = v + p_j.Q1 * Gy[0] - p_j.P1 * Hy[0]
        
        p=p/2/PI
        u=u/2/PI
        v=v/2/PI

    if (elementID!=-1): #query point locate on boundary
        #piecewise constant can not interpolation locally
        Element=panels[elementID]

        p=Element.P1
        u=Element.u1
        v=Element.v1
    
    #darcy flow -k/u*dp/dx
    return p,-u,-v


