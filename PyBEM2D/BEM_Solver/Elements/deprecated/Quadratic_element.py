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
from Lib.Tools.Geometry import cosspace,Point2Segment,GaussLib,Global2Iso,Subdivision,point_in_panel

from .Exact_Integration import Analytical_Intergration_cython


######################## Solver Module-Matrix assemble and field point solve ########################
def build_matrix_quadratic(panels, DDM=0, AB=[]):
    #All variables start from 1
    debug=0

    if(DDM == 1 and AB != 'none'):
        return update_matrix_quadratic(panels,AB)

    NE=len(panels) #number of elements
    N=2*NE #number of nodes
    #H IS A SQUARE MATRIX (2*NE,2*NE); G IS RECTANGULAR (2*NE,3*NE)
    #Index from 1
    G_mat=np.zeros((2*NE+1, 3*NE+1), dtype=float) #double node for flux term
    H_mat=np.zeros((2*NE+1, 2*NE+1), dtype=float)
    PI=3.141592653
    
    #prepare X,Y for Book's program,index from 1 and N+1=1
    X=np.zeros(N+2) 
    Y=np.zeros(N+2)
    for i, pl in enumerate(panels):
        X[2*i+1]=pl.xa
        Y[2*i+1]=pl.ya
        X[2*i+2]=pl.xc
        Y[2*i+2]=pl.yc
    X[N+1]=X[1]
    Y[N+1]=Y[1]
    #debug
    #for i in range(1,N+1+1):
    #    print("Node%s\t(%s-%s)"%(i,X[i],Y[i]))
    Hij=np.zeros(4)
    Gij=np.zeros(4)
    #1. Compute Gij[1,2,3] Hij[1,2,3] for each node and BE
    for LL in range(1,N+1): #The index of node
        for i in range(1,N-1+1,2): #The index of first node
            Node2Panel=int((i+1)/2)-1
            if((LL-i)*(LL-i-1)*(LL-i-2)*(LL-i+N-2)!=0): #off-diagonal
                #TEMP=GHCalc_quadratic(X[LL],Y[LL],panels[Node2Panel])
                #TEMP=GHCalc_quadratic_adapative(X[LL],Y[LL],panels[Node2Panel],'boundary')
                G,H,Gx,Gy,Hx,Hy=Analytical_Intergration_cython(X[LL],Y[LL],panels[Node2Panel])
                Hij[1:4], Gij[1:4] = H, G
            else:#Diagonal for Gii
                caseNo=LL-i+1
                if (LL==1) and (i==N-1):
                    caseNo=caseNo+N
                G,H,Gx,Gy,Hx,Hy=Analytical_Intergration_cython(X[LL],Y[LL],panels[Node2Panel])
                Hij[1:4], Gij[1:4] = H, G
                #Hij=GHCalc_quadratic(X[LL],Y[LL],panels[Node2Panel])[0]
                #Hij=GHCalc_quadratic_adapative(X[LL],Y[LL],panels[Node2Panel],'boundary')[0]
                #Gij=Gii_singular_quadratic(panels[Node2Panel],caseNo)
            for j in range(1,3+1):
                k=int(3*(i-1)/2)
                G_mat[LL,k+j]=G_mat[LL,k+j]+Gij[j]
                if (i-N+1==0):
                    if (j==3):
                        H_mat[LL,1]=H_mat[LL,1]+Hij[j]
                    else:
                        H_mat[LL,i-1+j]=H_mat[LL,i-1+j]+Hij[j]
                else:
                    H_mat[LL,i-1+j]=H_mat[LL,i-1+j]+Hij[j]
    
    
    #Compute the diagonal term
    for i in range(1,N+1):
        H_mat[i,i]=0
        for j in range(1,N+1):
            if(i!=j):
                H_mat[i,i]=H_mat[i,i]-H_mat[i,j]
        #For external problems:
        if (H_mat[i,i]<0):
            H_mat[i,i]=2*PI+H_mat[i,i]
    
    H_origin=np.copy(H_mat)
    G_origin=np.copy(G_mat)

    #2. Reorder the matrix based on Dirichlet and Neumann boundary condition
    #prepare KODE for Book's program,index from 1 and N+1=1
    KODE=np.zeros(3*NE+1, dtype=float) #bd_type, 1-neumann(robin), 0-dirichlet
    for i, pl in enumerate(panels):
        KODE[3*i+1]=panels[i].bd_Indicator
        KODE[3*i+2]=KODE[3*i+1]
        KODE[3*i+3]=KODE[3*i+1]

    for i in range(1,NE+1):
        for j in range(1,3+1):
            if(debug): print('Ele%s Node%s BC:%s'%(i,j,KODE[3*(i-1)+j]))
            #print(3*(i-1),3*(i-1)+j,KODE[3*(i-1)])
            if KODE[3*(i-1)+j]<=0:#This Ele's BC is Dirichelt
                if (i-NE!=0) or (j!=3):#This is not the last(3) node of last element 
                    if (i==1) or (j>1) or (KODE[3*(i-1)]==1):#If boundary condition is Neumann then interchange G-H G*U=H*P
                        #print('--Ele%s Node%s BC:%s'%(i,j,KODE[3*(i-1)+j]))
                        for k in range(1,N+1):
                            temp=H_mat[k,2*i-2+j]
                            H_mat[k,2*i-2+j]=-G_mat[k,3*i-3+j]
                            G_mat[k,3*i-3+j]=-temp
                        if(debug): print("G,Col%s<->H,Col%s"%(3*i-3+j,2*i-2+j))
                    else:#This is the first element and its first node and the previous ele's BC is Neumann, see Page 73
                        for k in range(1,N+1):
                            H_mat[k,2*i-1]=H_mat[k,2*i-1]-G_mat[k,3*i-2]
                            G_mat[k,3*i-2]=0
                    continue
                #Special case Dirichlet boundary is located at the last element
                if(debug): print('\nDirichlet on the last element')
                if KODE[1]>1e-6:#if the first node is Neumann
                    if(debug): print("Neumann")
                    for k in range(1,N+1):
                        temp=H_mat[k,1]
                        H_mat[k,1]=-G_mat[k,3*i]
                        G_mat[k,3*i]=-temp
                    if(debug): print("G,Col%s<->H,Col%s"%(3*i,1))
                    continue
                if KODE[1]<=1e-6:#if the first node is Dirichlet
                    if(debug): print('Dirichlet')
                    for k in range(1,N+1):
                        H_mat[k,1]=H_mat[k,1]-G_mat[k,3*i]
                        G_mat[k,3*i]=0
                    continue
    
    for i in range(1,NE+1):
        for j in range(1,3+1):
            if(debug): print('Ele%s Node%s BC:%s'%(i,j,KODE[3*(i-1)+j]))
            if (KODE[3*(i-1)+j]==1 and panels[i-1].Robin_alpha>0):#This Ele's BC is Dirichelt
                if (i-NE!=0) or (j!=3):#This is not the last(3) node of last element 
                    if (i==1) or (j>1) or (KODE[3*(i-1)]==1):#If boundary condition is Neumann then interchange G-H G*U=H*P
                        #print('--Ele%s Node%s BC:%s'%(i,j,KODE[3*(i-1)+j]))
                        for k in range(1,N+1):
                            H_mat[k,2*i-2+j]+=panels[i-1].Robin_alpha*G_mat[k,3*i-3+j]
                        if(debug): print("G,Col%s<->H,Col%s"%(3*i-3+j,2*i-2+j))
                    else:#This is the first element and its first node and the previous ele's BC is Neumann, see Page 73
                        for k in range(1,N+1):
                            H_mat[k,2*i-1]+=panels[i-1].Robin_alpha*G_mat[k,3*i-2]
                    continue
                #Special case Dirichlet boundary is located at the last element
                if(debug): print('\nDirichlet on the last element')
                if KODE[1]>1e-6:#if the first node is Neumann
                    if(debug): print("Neumann")
                    for k in range(1,N+1):
                        H_mat[k,1]+=panels[i-1].Robin_alpha*G_mat[k,3*i]
                    if(debug): print("G,Col%s<->H,Col%s"%(3*i,1))
                    continue
                if KODE[1]<=1e-6:#if the first node is Dirichlet
                    if(debug): print('Dirichlet')
                    for k in range(1,N+1):
                        H_mat[k,1]+=panels[i-1].Robin_alpha*G_mat[k,3*i]
                    continue


    H_mat=np.delete(H_mat,0,axis=1)
    H_mat=np.delete(H_mat,0,axis=0)
    A=H_mat

    #3.Assemble vector b
    #Prepare DFI for book's program
    DFI=np.zeros(3*NE+1, dtype=float) 
    for i, pl in enumerate(panels):
        DFI[3*i+1]=panels[i].bd_value1
        DFI[3*i+2]=panels[i].bd_value2
        DFI[3*i+3]=panels[i].bd_value3
    #print(DFI)
    
    b=np.zeros(2*NE+1, dtype=float)
    for i in range(1,N+1): #BE
        b[i]=0
        for j in range(1,3*NE+1): #BEs
                b[i]=b[i]+G_mat[i,j]*DFI[j]
    b=np.delete(b,0,axis=0)
    #debug
    H_origin=np.delete(H_origin,0,axis=1)
    H_origin=np.delete(H_origin,0,axis=0)
    G_origin=np.delete(G_origin,0,axis=1)
    G_origin=np.delete(G_origin,0,axis=0) #axis=0 row axis=1 column
    G_mat=np.delete(G_mat,0,axis=1)
    G_mat=np.delete(G_mat,0,axis=0)
    
    #print(H.shape)
    #print(H)
    #print(G.shape)
    #print(G)
    #print(A)
    #print(H_origin)
    return A,b,G,G_origin,H_origin

def solution_allocate_quadratic(panels,X,debug=0):
    NE=len(panels) #BE number
    N=2*NE #Node number
    
    #Reference code-BEM Introduction Course-1991-P81
    KODE=np.zeros(3*NE+1, dtype=float) #bd_type, 1-neumann, 0-dirichlet, odd-left node, even-right node
    FI=np.zeros(N+1, dtype=float) #Dirichelt boundary condition   Left=Right 
    DFI=np.zeros(3*NE+1, dtype=float) #Neumann boundary condition    Leff!=Right
    for i, pl in enumerate(panels):
        KODE[3*i+1]=panels[i].bd_Indicator
        KODE[3*i+2]=KODE[3*i+1]
        KODE[3*i+3]=KODE[3*i+1]
        DFI[3*i+1]=panels[i].bd_value1  
        DFI[3*i+2]=panels[i].bd_value2
        DFI[3*i+3]=panels[i].bd_value3     
        #In book's code, FI store matrix solution
    for i in range(1,N+1):
        FI[i]=X[i-1]

    #Robin Boundary condition convervation: Q=R-alpha*P
    for i in range(1,NE+1):
        if(panels[i-1].Robin_alpha>0):
            #print(panels[i-1].Robin_alpha)
            #print(i)
            #print(2*i-1,2*i,2*i+1)
            #print(3*(i-1)+1,3*(i-1)+2,3*(i-1)+3)            
            FIi_next=2*i+1
            if(i==NE):
                FIi_next=1
            if(KODE[3*(i-1)+1]==1 or KODE[3*(i-1)+2]==1 or KODE[3*(i-1)+3]==1):
                DFI[3*(i-1)+1]=panels[i-1].bd_value1-panels[i-1].Robin_alpha*FI[2*i-1]
                DFI[3*(i-1)+2]=panels[i-1].bd_value2-panels[i-1].Robin_alpha*FI[2*i]
                DFI[3*(i-1)+3]=panels[i-1].bd_value3-panels[i-1].Robin_alpha*FI[FIi_next]
                #aaa=1
    #print(DFI)

    for i in range(1,NE+1):
        for j in range(1,3+1):
            if KODE[3*i-3+j]<=0:
                if (i-NE!=0) or (j!=3):
                    if (i==1) or (j>1) or (KODE[3*i-3]==1):
                        temp=FI[2*i-2+j]
                        FI[2*i-2+j]=DFI[3*i-3+j]
                        DFI[3*i-3+j]=temp
                    else:
                        DFI[3*i-2]=DFI[3*i-3]
                    continue
                if KODE[1]>0:
                    temp=FI[1]
                    FI[1]=DFI[3*i]
                    DFI[3*i]=temp
                else:
                    DFI[3*i]=DFI[1]   
                continue

    
    for i in range(NE):
        panels[i].P1=FI[2*i+1]
        panels[i].P2=FI[2*i+2]
        panels[i].Q1=DFI[3*i+1]
        panels[i].Q2=DFI[3*i+2]
        panels[i].Q3=DFI[3*i+3]

    
    #Special Case
    #Assign P3 for each element, which is shared with the 1st node of the next element
    for i in range(NE):
        pl=panels[i]
        if(i==NE-1):
            pl_next=panels[0]
        else:
            pl_next=panels[i+1]
        pl.P3=pl_next.P1

    #Pressure derivate(u,v) on element nodes
    for i in range(NE):
        pl=panels[i]
        if(pl.Q1==0 and pl.Q2==0 and pl.Q3==0):# Zero Neumann has to be derived from pressure gradient in tangential direction
            pl.u1,pl.v1=calcUV_bd((pl.xa,pl.ya),pl,P_Pts=pl.P1)
            pl.u2,pl.v2=calcUV_bd((pl.xc,pl.yc),pl,P_Pts=pl.P2)
            pl.u3,pl.v3=calcUV_bd((pl.xb,pl.yb),pl,P_Pts=pl.P3,backward=1)
        else:#(Q1,Q2,Q3!=0)
            pl.u1,pl.v1=pl.Q1*pl.nx,pl.Q1*pl.ny
            pl.u2,pl.v2=pl.Q2*pl.nx,pl.Q2*pl.ny
            pl.u3,pl.v3=pl.Q3*pl.nx,pl.Q3*pl.ny

    if(debug):
        print("[Solution Results]")
        print("Number of boundary elements:%s" % (len(panels)))
        print("Coordinates&Boundary conditions of boundary elements:")
        print("Point\tX\t\tY\t\tPressure\tLeft Flux\tRight flux")
        for i in range(NE):
            pl=panels[i]
            if i==0:
                pl_p=panels[NE-1]
            else:
                pl_p=panels[i-1]
            print("(%s)%s\t%5.3f\t\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f" % (i+1,2*i+1,pl.xa,pl.ya,pl.P1,pl_p.Q3,pl.Q1))
            print("(%s)%s\t%5.3f\t\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f" % (i+1,2*i+2,pl.xc,pl.yc,pl.P2,pl.Q2,pl.Q2))


def Field_Solve_quadratic(xi,yi,panels,elementID=-1):
    #Calculate domain solution using BIE formulation-Page 105 @ BEM Introduction Course-1991
    #elementID=-1 internal point   elementID>=0 boundary point

    PI=3.141592653
    NE=len(panels)
    p,u,v=0,0,0

    Q = np.zeros(3, dtype=float)
    P = np.zeros(3, dtype=float)

    if (elementID==-1): #query point locate on internal domain
    
        for i in range(1,NE+1):
            pl=panels[i-1]
            G,H,Gx,Gy,Hx,Hy=Analytical_Intergration_cython(xi,yi,pl)

            Hij,Gij=H,G
            DUx,DQx=Gx,Hx
            DUy,DQy=Gy,Hy
            
            pl=panels[i-1]
            if i==NE:
                pl_next=panels[0]
            else:
                pl_next=panels[i]
            
            Q[0],Q[1],Q[2]=pl.Q1,pl.Q2,pl.Q3
            P[0],P[1],P[2]=pl.P1,pl.P2,pl_next.P1
            for j in range(3):
                p=p+Gij[j]*Q[j]-Hij[j]*P[j]
                u=u+DUx[j]*Q[j]-DQx[j]*P[j]
                v=v+DUy[j]*Q[j]-DQy[j]*P[j]
    
        p=p/2/PI
        u=u/2/PI
        v=v/2/PI
    
    if (elementID!=-1): #query point locate on boundary
        Pts=(xi,yi) #query point
        Element=panels[elementID]
    
        #shape function & Node value
        phi=Element.get_ShapeFunc(Pts)
        Pi=[Element.P1,Element.P2,Element.P3]        
        ui=[Element.u1,Element.u2,Element.u3]
        vi=[Element.v1,Element.v2,Element.v3]

        for i in range(3):
            p+=phi[i]*Pi[i]
            u+=phi[i]*ui[i]
            v+=phi[i]*vi[i]
        
        #debug
        #print('Point',Pts)
        #print('Element',elementID+1,Element.nx,Element.ny)
        #print(Pi,ui,vi)

    #darcy flow -k/u*dp/dx
    return p,-u,-v


def update_matrix_quadratic(panels,AB=[]):
    
    NE=len(panels) #number of elements
    N=2*NE #number of nodes
    
    #Collecting prescribed BC values
    debug = 0
    A = AB[0]
    G = AB[1]

    #3.Assemble vector b
    #Prepare DFI for book's program
    DFI = np.zeros(3 * NE, dtype=float)
    for i, pl in enumerate(panels):
        DFI[3 * i + 0] = panels[i].bd_value1
        DFI[3 * i + 1] = panels[i].bd_value2
        DFI[3 * i + 2] = panels[i].bd_value3
    #print(DFI)

    b = np.zeros(2 * NE, dtype=float)
    for i in range(N):  # BE
        b[i] = 0
        for j in range(3 * NE):  # BEs
                b[i] = b[i] + G[i, j] * DFI[j]
    
    return A,b,G




def calcP_bd(Pts, Element):
    #calculate the p on a specific element
    phi = Element.get_ShapeFunc(Pts)
    Pi = [Element.P1, Element.P2, Element.P3]

    p = 0.0
    for i in range(3):
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


    

