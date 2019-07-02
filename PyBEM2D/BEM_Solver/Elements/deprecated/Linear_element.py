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
from Lib.Tools.Geometry import cosspace,Point2Segment,GaussLib,Global2Iso,Subdivision,point_in_panel

from .Exact_Integration import Analytical_Intergration_cython


######################## Solver Module-Matrix assemble and field point solve ########################
def build_matrix_linear(panels, DDM=0, AB=[]):
    
    if(DDM == 1 and AB != 'none'):
        return update_matrix_linear(panels, AB)
    
    N=len(panels)
    #!!!!index from 1!!!!
    G_mat=np.zeros((N+1, 2*N+1), dtype=float) #double node for flux term
    H_mat=np.zeros((N+1, N+1), dtype=float)
    PI=3.141592653

    for i in range(1,N+1):
        #1. off-diagonal
        #print("********Node(%s)\t (%s,%s)*******" % (i,panels[i-1].xa,panels[i-1].ya))
        #print('off-diagonal element')
        for jj in range(i+1,i+N-2+1): #for i in range(1,3) == i=1,2
            #i=1 element  off-diagonal element j(2-3,...,11-12)
            if jj-N<=0:
                j=jj
            else:
                j=jj-N
            #print("BE(%s-%s) (%s,%s)-(%s,%s)" % (j,j+1,panels[j-1].xa,panels[j-1].ya,panels[j-1].xb,panels[j-1].yb))
            #GHIJ=GHCalc_linear(panels[i-1].xa,panels[i-1].ya,panels[j-1])
            G,H,Gx,Gy,Hx,Hy=Analytical_Intergration_cython(panels[i-1].xa,panels[i-1].ya,panels[j-1])
            Hij1=H[0]
            Hij2=H[1]
            Gij1=G[0]
            Gij2=G[1]
            
            H_mat[i,j]=H_mat[i,j]+Hij1 #Hij1 belong to j
            if j-N<0:
                H_mat[i,j+1]=H_mat[i,j+1]+Hij2 #Hij belong to j+1
            else:
                H_mat[i,1]=H_mat[i,1]+Hij2
            #eg. Node 1, G has two node, H has only one
            G_mat[i,2*j-1]=Gij1 #odd number, left
            G_mat[i,2*j]=Gij2   #even number, right
            
        #2. diagonal element-singular element
            H_mat[i,i]=H_mat[i,i]-Hij1-Hij2
        #print('diagonal element')
        for jj in range(i+N-1,i+N+1): #for i in range(1,3) == i=1,2
            #print("i:%s\tjj(%s,%s) " % (i,i+N-1,i+N))
            #i=1 diagonal element (12-1)(1-2)
            if jj-N<=0:
                j=jj
            else:
                j=jj-N
            #print("BE(%s-%s) (%s,%s)-(%s,%s)" % (j,j+1,panels[j-1].xa,panels[j-1].ya,panels[j-1].xb,panels[j-1].yb))
            #Gii analytical integration
            Gij1=panels[j-1].length*(1.5-np.log(panels[j-1].length))/2
            Gij2=panels[j-1].length*(0.5-np.log(panels[j-1].length))/2
            
            if jj-(i+N-1)<=0: #left element, 
                #print("Left BE(%s-%s) (%s,%s)-(%s,%s)" % (j,j+1,panels[j-1].xa,panels[j-1].ya,panels[j-1].xb,panels[j-1].yb))
                temp=Gij1
                Gij1=Gij2
                Gij2=temp
            G_mat[i,2*j-1]=Gij1
            G_mat[i,2*j]=Gij2
        #3. External problem, additional line
        if H_mat[i,i]<0:
            H_mat[i,i]=2*PI+H_mat[i,i]
    
    
    #Dirichlet-Neumann Boundary Conditions
    #Assemble matrix A and reorder matrix H and G (searching column)            
    #Book P76, assemble algorithm
    #index from 1
    KODE=np.zeros(2*N+1, dtype=float) #bd_type, 1-neumann, 0-dirichlet, odd-left node, even-right node
    FI=np.zeros(N+1, dtype=float) #Dirichelt boundary condition   Left=Right 
    DFI=np.zeros(2*N+1, dtype=float) #Neumann boundary condition    Leff!=Right
    for i in range(1,N+1):
        KODE[2*i-1]=panels[i-1].bd_Indicator
        KODE[2*i]=KODE[2*i-1]
        DFI[2*i-1]=panels[i-1].bd_value1   #odd left
        DFI[2*i]=panels[i-1].bd_value2     #even right
    #print('KODE')
    #print(KODE)
    #print('DFI')
    #print(DFI)
    for i in range(1,N+1):
        for j in range(1,2+1):
            #print('Ele%s Node%s BC:%s Robin:%s'%(i,j,KODE[2*(i-1)+j],panels[i-1].Robin_alpha))
            if KODE[2*i-2+j]<=0: ##If boundary condition is dirichlet then interchange G-H G*U=H*P
                if (i!=N) or (j!=2):
                    if (i==1) or (j>1) or (KODE[2*i-2]==1):
                        for k in range(1,N+1):
                            temp=H_mat[k,i-1+j]
                            H_mat[k,i-1+j]=-G_mat[k,2*i-2+j]
                            G_mat[k,2*i-2+j]=-temp
                    else:
                        for k in range(1,N+1):
                            H_mat[k,i]=H_mat[k,i]-G_mat[k,2*i-1]
                            G_mat[k,2*i-1]=0
                    #print("G,Col%s<->H,Col%s"%(2*(i-1)+j,i-1+j))
                    continue
                if KODE[1]>0:
                    for k in range(1,N+1):
                        temp=H_mat[k,1]
                        H_mat[k,1]=-G_mat[k,2*N]
                        G_mat[k,2*N]=-temp
                continue
                if KODE[1]<=0:
                    for k in range(1,N+1):
                        H_mat[k,1]=H_mat[k,1]-G_mat[k,2*N]
                        G_mat[k,2*N]=0
                continue

    #print('\n')

    #Robin Boundary Condition 
    #Reference https://www.researchgate.net/project/Extending-the-boundary-element-method-to-the-generalised-Robin-boundary-condition
    for i in range(1,N+1):
        for j in range(1,2+1):
            #print('Ele%s Node%s BC:%s Robin:%s'%(i,j,KODE[2*(i-1)+j],panels[i-1].Robin_alpha))
            if (KODE[2*i-2+j]==1 and panels[i-1].Robin_alpha>0): ##If boundary condition is dirichlet then interchange G-H G*U=H*P
                if (i!=N) or (j!=2):
                    if (i==1) or (j>1) or (KODE[2*i-2]==1):
                        for k in range(1,N+1):
                            H_mat[k,i-1+j]+=panels[i-1].Robin_alpha*G_mat[k,2*(i-1)+j]
                    else:
                        for k in range(1,N+1):
                            H_mat[k,i]+=panels[i-1].Robin_alpha*G_mat[k,2*i-1]
                    continue
                if KODE[1]>0:
                    for k in range(1,N+1):
                        H_mat[k,1]+=panels[i-1].Robin_alpha*G_mat[k,2*N]
                continue
                if KODE[1]<=0:
                    for k in range(1,N+1):
                        H_mat[k,1]+=panels[i-1].Robin_alpha*G_mat[k,2*N]
                continue


    #print(H)
    #print(G)
    A=H_mat
    
    #Assemble vector b
    for i in range(1,N+1): #BE
        FI[i]=0
        for j in range(1,2*N+1): #BEs
                FI[i]=FI[i]+G_mat[i,j]*DFI[j]
    b=FI
    #Restore matrix as index from 0
    A=np.delete(A,0, axis=1)
    A=np.delete(A,0,axis=0)
    b=np.delete(b,0,axis=0) #axis=0 row axis=1 column
    G_mat = np.delete(G_mat, 0, axis=1)
    G_mat = np.delete(G_mat, 0, axis=0)

    return A,b,G_mat


def update_matrix_linear(panels, AB=[]):

    N = len(panels)

    #Collecting prescribed BC values
    debug = 0
    A = AB[0]
    G = AB[1]

    FI=np.zeros(N, dtype=float) #Dirichelt boundary condition   Left=Right 
    DFI=np.zeros(2*N, dtype=float) #Neumann boundary condition    Leff!=Right
    for i in range(N):
        DFI[2*i]=panels[i].bd_value1   #odd left
        DFI[2*i+1]=panels[i].bd_value2     #even right

    #Assemble vector b
    for i in range(N):  # BE
        FI[i] = 0
        for j in range(2*N):  # BEs
                FI[i] = FI[i] + G[i, j] * DFI[j]
    b = FI

    return A, b, G


def solution_allocate_linear(panels,X,debug=0):
    N=len(panels)
    #Reference code-BEM Introduction Course-1991-P81
    KODE=np.zeros(2*N+1, dtype=float) #bd_type, 1-neumann, 0-dirichlet, odd-left node, even-right node
    FI=np.zeros(N+1, dtype=float) #Dirichelt boundary condition   Left=Right 
    DFI=np.zeros(2*N+1, dtype=float) #Neumann boundary condition    Leff!=Right
    for i in range(1,N+1):
        KODE[2*i-1]=panels[i-1].bd_Indicator
        KODE[2*i]=KODE[2*i-1]
        DFI[2*i-1]=panels[i-1].bd_value1   #odd left
        DFI[2*i]=panels[i-1].bd_value2     #even right
        #In book's code, FI store matrix solution
        FI[i]=X[i-1]

    #print(DFI)


    #Robin Boundary condition convervation: Q=R-alpha*P
    '''
    for i in range(2,N+1):
        #print('All',i,panels[i-1].Robin_alpha)
        if (panels[i-1].Robin_alpha>0):
            if (KODE[2*i-2]==1 or KODE[2*i-1]==1):
                #print(i)
                #print(2*i-2,2*i-1)
                FIi_last=2*i+1
                if(i==N):
                    FIi_last=1
                DFI[2*i-2]=panels[i-1].bd_value1-panels[i-1].Robin_alpha*FI[i] #odd left
                DFI[2*i-1]=panels[i-1].bd_value2-panels[i-1].Robin_alpha*FI[i] #even right
    '''

    #Dirichlet and Neumann Manipulation switch row
    for i in range(1,N+1):
        for j in range(1,2+1):
            if KODE[2*i-2+j]<=0:#This is a Dirichlet BC
                if (i!=N) or (j!=2):
                    if (i==1) or (j==2) or (KODE[2*i-2]==1):
                        temp=FI[i-1+j]
                        FI[i-1+j]=DFI[2*i-2+j]
                        DFI[2*i-2+j]=temp
                    else:
                        DFI[2*i-1]=DFI[2*i-2]
                    continue
                if KODE[1]>0:
                    temp=FI[1]
                    FI[1]=DFI[2*N]
                    DFI[2*N]=temp
                else:
                    DFI[2*N]=DFI[1]   
                continue
    
    #print(FI)
    #Allocate real solution to BE class
    panels[0].P1=FI[1]
    panels[0].Q1=DFI[2*N]
    panels[0].Q2=DFI[1]
    for i in range(2,N+1):
        panels[i-1].P1=FI[i]
        panels[i-1].Q1=DFI[2*i-2]    #odd left
        panels[i-1].Q2=DFI[2*i-1]    #even right

    #Convert Robin flux to Neumann Flux
    for i, pl in enumerate(panels):
        if(i==0):
            pl_last=panels[N-1]
        else:
            pl_last=panels[i-1]
        if(pl_last.Robin_alpha>0):#This is a robin Boundary
            pl.Q1=pl.Q1-pl_last.Robin_alpha*pl.P1
        if(pl.Robin_alpha>0):#This is a robin Boundary
            pl.Q2=pl.Q2-pl.Robin_alpha*pl.P1

    #Special Case
    #Assign P2 for each element, which is shared with the 1st node of the next element
    for i in range(N):
        pl=panels[i]
        if(i==N-1):
            pl_next=panels[0]
        else:
            pl_next=panels[i+1]
        pl.P2=pl_next.P1
        pl.q1=pl.Q2 #right=q1
        pl.q2=pl_next.Q1 #next.left=q2
    
    #Pressure derivate(u,v) on element nodes
    for i in range(N):
        pl=panels[i]
        if(i==N-1):
            pl_next=panels[0]
        else:
            pl_next=panels[i+1]
        if(pl.q1==0 and pl_next.q2==0):# Zero Neumann has to be derived from pressure gradient in tangential direction
            pl.u1,pl.v1=calcUV_bd((pl.xa,pl.ya),pl,P_Pts=pl.P1)
            pl.u2,pl.v2=calcUV_bd((pl.xb,pl.yb),pl,P_Pts=pl.P2,backward=1)
        else:#(Q1,Q2,Q3!=0)
            pl.u1,pl.v1=pl.q1*pl.nx,pl.q1*pl.ny
            pl.u2,pl.v2=pl.q2*pl.nx,pl.q2*pl.ny

    if(debug):
        print("[Solution Results]")
        print("Number of boundary elements:%s" % (len(panels)))
        print("Coordinates&Boundary conditions of boundary elements:")
        print("Point\tX\t\tY\t\tBD_P\t\tLeft Flux\tRight flux")
        for i, pl in enumerate(panels):
            print("%s\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f" % (i+1,pl.xa,pl.ya,pl.P1,pl.Q1,pl.Q2))    

def Field_Solve_linear(xi,yi,panels,elementID=-1):
    PI=3.141592653
    N=len(panels)
    p,u,v=0,0,0
    
    if (elementID == -1):  # query point locate on internal domain

        for j in range(N-1):
            G,H,Gx,Gy,Hx,Hy=Analytical_Intergration_cython(xi,yi,panels[j])
            p=p+panels[j].Q2*G[0]+panels[j+1].Q1*G[1]-panels[j].P1*H[0]-panels[j+1].P1*H[1]
            u=u+panels[j].Q2*Gx[0]+panels[j+1].Q1*Gx[1]-panels[j].P1*Hx[0]-panels[j+1].P1*Hx[1]
            v=v+panels[j].Q2*Gy[0]+panels[j+1].Q1*Gy[1]-panels[j].P1*Hy[0]-panels[j+1].P1*Hy[1]
        
        G,H,Gx,Gy,Hx,Hy=Analytical_Intergration_cython(xi,yi,panels[N-1])
        p=p+panels[N-1].Q2*G[0]+panels[0].Q1*G[1]-panels[N-1].P1*H[0]-panels[0].P1*H[1]
        u=u+panels[N-1].Q2*Gx[0]+panels[0].Q1*Gx[1]-panels[N-1].P1*Hx[0]-panels[0].P1*Hx[1]
        v=v+panels[N-1].Q2*Gy[0]+panels[0].Q1*Gy[1]-panels[N-1].P1*Hy[0]-panels[0].P1*Hy[1]
        
        p=p/2/PI
        u=u/2/PI
        v=v/2/PI

    if (elementID!=-1): #query point locate on boundary
        Pts=(xi,yi) #query point
        Element=panels[elementID]
        #Element.eval_UV()

        #shape function & Node value
        phi=Element.get_ShapeFunc(Pts)
        Pi=[Element.P1,Element.P2]        
        ui=[Element.u1,Element.u2]
        vi=[Element.v1,Element.v2]
        
        for i in range(2):
            p+=phi[i]*Pi[i]
            u+=phi[i]*ui[i]
            v+=phi[i]*vi[i]
        
    #darcy flow -k/u*dp/dx
    return p,-u,-v


def calcP_bd(Pts, Element):
    #calculate the p on a specific element
    phi = Element.get_ShapeFunc(Pts)
    Pi = [Element.P1, Element.P2]

    p = 0.0
    for i in range(2):
        p += phi[i] * Pi[i]

    return p


def calcUV_bd(Pts, Element, P_Pts=-1, backward=0):
    #calculate the uv on a specific element using (Pts2-Pts)/dist
    #Default forward:Pts .--. Pts2 backward:Pts2 .--. Pts

    dist = 0.000001

    Pts = np.array(Pts)
    unit_vector = np.array((Element.tx, Element.ty))

    if (P_Pts == -1):
        P_Pts = calcP_bd(Pts, Element)

    if(backward != 1):  # forward
        Pts2 = Pts + dist * unit_vector
        temp = (calcP_bd(Pts2, Element) - P_Pts) / dist

    else:  # backward
        Pts2 = Pts - dist * unit_vector
        temp = (P_Pts - calcP_bd(Pts2, Element)) / dist

    u = temp * Element.tx
    v = temp * Element.ty

    return u, v
