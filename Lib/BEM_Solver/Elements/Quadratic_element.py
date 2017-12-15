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
from .Quadratic_element_disct import *


######################## Solver Module-Kernal Integration ########################
def GHCalc_quadratic(xi,yi,panelj):
    #Off diagonal term calculation11
    #Page 132 in Katsikadelis'book 2016

    #Gauss Quadrature intergration,n=10
    #Gauss point-index from 1
    GI=[0,0.9739065285,-0.9739065285,0.8650633666,-0.8650633666,0.6794095683,-0.6794095682,0.4333953941,-0.4333953941,0.1488743389,-0.1488743389]  
    #weight cofficient-index from 1
    WC=[0,0.0666713443,0.0666713443,0.1494513491,0.1494513491,0.2190863625,0.2190863625,0.2692667193,0.2692667193,0.2955242247,0.2955242247]    
    X1,Y1=panelj.xa,panelj.ya
    X2,Y2=panelj.xc,panelj.yc
    X3,Y3=panelj.xb,panelj.yb
    
    #Integration interval transoform
    #Radius derivative coefficient A,B,C,D
    A=X3-2*X2+X1
    B=(X3-X1)/2
    C=Y3-2*Y2+Y1
    D=(Y3-Y1)/2
    XjG,YjG=0,0 #Intergation point along a boundary
    Rij=0  #Distance from node to Gauss point on boundary element
    Rd1=0  #Radius derivatives 
    Rd2=0  #Raidus derivatives
    Rd=0   #Rd=nx*Rd1+ny*Rd2
    nx,ny=0,0 #unit normal component on the boundary element
    JAC=0  #Jacobian for higher order element which replaced length/2 with linear element
    #Boundary solution variable-index start from 1
    phi=np.zeros(3+1) #shape function 1,2,3   phi(phi-1)/2, 1-phi^2,phi(phi+1)/2
    Hij=np.zeros(3+1)  #H matrrix value -velocity in terms of shape fucntion1,2,3 
    Gij=np.zeros(3+1)  #G matrrix value -pressure in terms of shape fucntion1,2,3 
    #Internal solution variable
    DUx=np.zeros(3+1)  #G derivative in x direction shape function1,2,3
    DUy=np.zeros(3+1)  #G derivative in y direction shape function1,2,3
    DQx=np.zeros(3+1)  #H derivative in x direction shape function1,2,3
    DQy=np.zeros(3+1)  #H derivative in y direction shape function1,2,3
    
    for j in range(1,10+1):
        phi[1]=GI[j]*(GI[j]-1)/2
        phi[2]=1-GI[j]**2
        phi[3]=GI[j]*(GI[j]+1)/2
        XjG=X1*phi[1]+X2*phi[2]+X3*phi[3]
        YjG=Y1*phi[1]+Y2*phi[2]+Y3*phi[3]
        JAC=np.sqrt((GI[j]*A+B)**2+(GI[j]*C+D)**2)
        nx=(GI[j]*C+D)/JAC
        ny=-(GI[j]*A+B)/JAC
        Rij=np.sqrt((xi-XjG)**2+(yi-YjG)**2)
        Rd1=(XjG-xi)/Rij
        Rd2=(YjG-yi)/Rij
        Rd=Rd1*nx+Rd2*ny
        
        for k in range(1,3+1):
            
            Hij[k]=Hij[k]-phi[k]*Rd/Rij*WC[j]*JAC
            Gij[k]=Gij[k]+phi[k]*np.log(1/Rij)*WC[j]*JAC
        
            DUx[k]=DUx[k]+phi[k]*Rd1/Rij*WC[j]*JAC
            DUy[k]=DUy[k]+phi[k]*Rd2/Rij*WC[j]*JAC
            DQx[k]=DQx[k]-phi[k]*(2*Rd1*Rd2*ny+(2*Rd1**2-1)*nx)/(Rij**2)*WC[j]*JAC
            DQy[k]=DQy[k]-phi[k]*(2*Rd1*Rd2*nx+(2*Rd2**2-1)*ny)/(Rij**2)*WC[j]*JAC

    return Hij,Gij,DUx,DQx,DUy,DQy

def GHCalc_quadratic_adapative(xi,yi,panelj,bd_int="internal"):
    #bd_inter- Position ID 1-boundary 0-interal domain1
    t0 = time.time()
    #Influence coefficient
    Hij=np.zeros(3+1)  #H matrrix value -velocity in terms of shape fucntion1,2,3 
    Gij=np.zeros(3+1)  #G matrrix value -pressure in terms of shape fucntion1,2,3 
    #Internal solution variable
    DUx=np.zeros(3+1)  #G derivative in x direction shape function1,2,3
    DUy=np.zeros(3+1)  #G derivative in y direction shape function1,2,3
    DQx=np.zeros(3+1)  #H derivative in x direction shape function1,2,3
    DQy=np.zeros(3+1)  #H derivative in y direction shape function1,2,3
        
    #BE element information
    X1,Y1=panelj.xa,panelj.ya
    X2,Y2=panelj.xc,panelj.yc
    X3,Y3=panelj.xb,panelj.yb
    length=panelj.length
    unit_vx=(X3-X1)/length
    unit_vy=(Y3-Y1)/length

    #Element subdivision+adapative gauss point selection
    Param1=Subdivision(xi,yi,X1,Y1,X3,Y3,1e-5,0,bd_int) #parameter [Subdivision trigger,NG_required]
    #No_sub=Param1[2]
    Trigger_sub=Param1[0]
    
    GaussOrder=[]
    if Trigger_sub==1:
        length_sub=[]
        No_sub=0
        Cum_length=0
        #length_sub.append(Param1[2]) #the first segment in sub element
        #GaussOrder.append(Param1[1]) #the NG pts in the first segment
        X1_sub,Y1_sub=X1,Y1 #the start point of the first segment
        X3_sub,Y3_sub=X3,Y3
        for itera in range(8000):
            #GaussOrder.append(GaussOrder[No_sub]) #last iteration
            #length_sub.append(length_sub[No_sub]) #last iteration
    
            Param2=Subdivision(xi,yi,X1_sub,Y1_sub,X3_sub,Y3_sub,1e-5,1,bd_int)
            GaussOrder.append(Param2[1])
            length_sub.append(Param2[2]) 
            Cum_length=Cum_length+length_sub[No_sub]
            X1_sub=X1_sub+length_sub[No_sub]*unit_vx
            Y1_sub=Y1_sub+length_sub[No_sub]*unit_vy
            #X3_sub=X1_sub-length_sub[No_sub]*unit_vx
            #Y3_sub=Y1_sub-length_sub[No_sub]*unit_vy
            No_sub=No_sub+1
            #print(X1_sub,X3_sub)
            #print('Cum_length:%s'%(Cum_length))
            if (Cum_length>=length-0.0001):
                #print('-------Subelement----%s'%(No_sub))
                temp=0
                for i in range(No_sub):
                    temp=temp+length_sub[i]
                    #print('No.%s length:%s NG pts:%s'%(i+1,length_sub[i],GaussOrder[i]))
                #print('Cum_length:%s'%(temp))
                break
        #store the endpoint of each sub-element
        endpoint_sub=Global2Iso(length_sub,length,3)
    elif Trigger_sub==0:
            GaussOrder.append(Param1[1])
            No_sub=1
    t1 = time.time()
    
    for m in range(No_sub): #loop for each subelement      
        GI,WC=GaussLib(GaussOrder[m]) #Gauss point and weight
        #if Trigger_sub==1:
            #endpoint_sub=Global2Iso(length_sub,length,3) #start, end point for the sub region in local coordinate(-1,1)
            #print(endpoint_sub)
            #x1_sub,x3_sub=endpoint_sub[m],endpoint_sub[m+1]
            #for j in range(1,GaussOrder[m]+1):
                #GI[j]=0.5*(x3_sub+x1_sub)+0.5*(x3_sub-x1_sub)*GI[j]
            #print('Subdomain:%s'%(m+1))
            #print(GI)
    #Radius derivative coefficient A,B,C,D
        A=X3-2*X2+X1
        B=(X3-X1)/2
        C=Y3-2*Y2+Y1
        D=(Y3-Y1)/2
        XjG,YjG=0,0 #Intergation point along a boundary, global coordiante
        Rij=0  #Distance from node to Gauss point on boundary element
        Rd1=0  #Radius derivatives 
        Rd2=0  #Raidus derivatives
        Rd=0   #Rd=nx*Rd1+ny*Rd2
        nx,ny=0,0 #unit normal component on the boundary element
        JAC=0  #Jacobian for higher order element which replaced length/2 with linear element
        JAC_bar=1 #subdivision jacobian transoform for local and global coordinate
        #if Trigger_sub==1: JAC_bar=0.5*(x3_sub-x1_sub) #John. T.Katsikadelis p146   Gernot Beer p148
    #Boundary solution variable-index start from 1
        phi=np.zeros(3+1) #shape function 1,2,3   phi(phi-1)/2, 1-phi^2,phi(phi+1)/2
        x1_sub,x3_sub=0,0
        
        for j in range(1,GaussOrder[m]+1):
            if Trigger_sub==1:
            #endpoint_sub=Global2Iso(length_sub,length,3) #start, end point for the sub region in local coordinate(-1,1)
            #print(endpoint_sub)
                x1_sub,x3_sub=endpoint_sub[m],endpoint_sub[m+1]
                GI[j]=0.5*(x3_sub+x1_sub)+0.5*(x3_sub-x1_sub)*GI[j]
                JAC_bar=0.5*(x3_sub-x1_sub)
            
            phi[1]=GI[j]*(GI[j]-1)/2
            phi[2]=1-GI[j]**2
            phi[3]=GI[j]*(GI[j]+1)/2
            XjG=X1*phi[1]+X2*phi[2]+X3*phi[3]
            YjG=Y1*phi[1]+Y2*phi[2]+Y3*phi[3]
            JAC=np.sqrt((GI[j]*A+B)**2+(GI[j]*C+D)**2)
            nx=(GI[j]*C+D)/JAC
            ny=-(GI[j]*A+B)/JAC
            Rij=np.sqrt((xi-XjG)**2+(yi-YjG)**2)
            Rd1=(XjG-xi)/Rij
            Rd2=(YjG-yi)/Rij
            Rd=Rd1*nx+Rd2*ny
            
            for k in range(1,3+1):
                Hij[k] += -phi[k]*Rd/Rij*WC[j]*JAC*JAC_bar
                Gij[k] += +phi[k]*np.log(1/Rij)*WC[j]*JAC*JAC_bar
        
                DUx[k] += +phi[k]*Rd1/Rij*WC[j]*JAC*JAC_bar
                DUy[k] += +phi[k]*Rd2/Rij*WC[j]*JAC*JAC_bar
                DQx[k] += -phi[k]*(2*Rd1*Rd2*ny+(2*Rd1**2-1)*nx)/(Rij**2)*WC[j]*JAC*JAC_bar
                DQy[k] += -phi[k]*(2*Rd1*Rd2*nx+(2*Rd2**2-1)*ny)/(Rij**2)*WC[j]*JAC*JAC_bar

    constant=1  #/2/np.pi
    t2 = time.time()
    #print(t1-t0,t2-t0)
    return Hij,Gij,DUx,DQx,DUy,DQy


def Gii_singular_quadratic(panelj,case):
    #Diagonal term
#Gii for a quadratic can not solved analytically,like linear element
#THE NON SINGULAR PART IS COMPUTED USING STANDARD GAUSS QUADRATURE,
#THE LOGARITHMIC PART IS COMPUTED USING A SPECIAL QUADRATURE FORMULA.
    
    #Gauss Quadrature intergration,n=10
    #Gauss point-index from 1
    GI=[0,0.9739065285,-0.9739065285,0.8650633666,-0.8650633666,0.6794095683,-0.6794095682,0.4333953941,-0.4333953941,0.1488743389,-0.1488743389]  
    #weight cofficient-index from 1
    WC=[0,0.0666713443,0.0666713443,0.1494513491,0.1494513491,0.2190863625,0.2190863625,0.2692667193,0.2692667193,0.2955242247,0.2955242247] 
    #Logarithmic Quadrature intergration,n=10
    GIL=[0,0.0090426309,0.0539712662,0.1353118246,0.2470524162,0.3802125396,0.5237923179,0.6657752055,0.7941904160,0.8981610912,0.9688479887]
    WCL=[0,0.1209551319,0.1863635425,0.1956608732,0.1735771421,0.1356956729,0.0936467585,0.0557877273,0.0271598109,0.0095151826,0.0016381576]

    Gii=np.zeros(3+1)  #G matrrix value -pressure in terms of shape fucntion1,2,3 
    XG1,YG1=panelj.xa,panelj.ya
    XG2,YG2=panelj.xc,panelj.yc
    XG3,YG3=panelj.xb,panelj.yb
    
    S1,S2,S3=0,0,0
    #Case1
    if case==1:
        #Local coornidate tranform
        X3=XG3-XG1
        Y3=YG3-YG1
        X2=XG2-XG1
        Y2=YG2-YG1
        A1=(X3-2*X2)*0.5
        B1=X2 
        A2=(Y3-2*Y2)*0.5
        B2=Y2
        #Geometrical properties
        for j in range(1,10+1): 
            XJA1=np.sqrt((4*A1*GIL[j]-2*A1+0.5*X3)**2+(4*A2*GIL[j]-2*A2+0.5*Y3)**2)*2 
            XJA2=np.sqrt((A1*GI[j]*2+0.5*X3)**2+(A2*GI[j]*2+0.5*Y3)**2)
            XLO=-np.log(2*np.sqrt((GI[j]*A1+B1)**2+(GI[j]*A2+B2)**2))
            #shape function
            F3=0.5*GI[j]*(GI[j]+1.)
            F2=1.-GI[j]**2
            F1=0.5*GI[j]*(GI[j]-1.)
            FL3=GIL[j]*(2.*GIL[j]-1.)
            FL2=4.*GIL[j]*(1.-GIL[j])
            FL1=(GIL[j]-1.)*(2.*GIL[j]-1.)
            #integration1
            S3=FL3*XJA1*WCL[j]+F3*XJA2*XLO*WC[j] 
            S2=FL2*XJA1*WCL[j]+F2*XJA2*XLO*WC[j] 
            S1=FL1*XJA1*WCL[j]+F1*XJA2*XLO*WC[j]
            
            Gii[1]=Gii[1]+S1
            Gii[2]=Gii[2]+S2
            Gii[3]=Gii[3]+S3
    if case==2:
        X3=XG3-XG2
        Y3=YG3-YG2
        X1=XG1-XG2
        Y1=YG1-YG2
        A1=X1+X3
        B1=X3-X1
        A2=Y1+Y3
        B2=Y3-Y1
        #Geometrical properties
        for j in range(1,10+1):
            XJA1=np.sqrt((0.5*B1-A1*GIL[j])**2+(0.5*B2-A2*GIL[j])**2)
            XJA11=np.sqrt((0.5*B1+A1*GIL[j])**2+(0.5*B2+A2*GIL[j])**2)
            XJA2=np.sqrt((0.5*B1+A1*GI[j])**2+(0.5*B2+A2*GI[j])**2)
            XLO=-0.5*np.log((GI[j]*A1*0.5+B1*0.5)**2+(GI[j]*A2*0.5+B2*0.5)**2)
            #shape function
            F3=0.5*GI[j]*(GI[j]+1.)
            F2=1.-GI[j]**2
            F1=0.5*GI[j]*(GI[j]-1.)
            FLN3=0.5*GIL[j]*(GIL[j]+1.)
            FLN2=1.-GIL[j]**2
            FLN1=0.5*GIL[j]*(GIL[j]-1.)
            #integration2
            S3=(FLN1*XJA1+FLN3*XJA11)*WCL[j]+F3*XJA2*XLO*WC[j] 
            S2=FLN2*(XJA1+XJA11)*WCL[j]+F2*XJA2*XLO*WC[j]
            S1=(FLN3*XJA1+FLN1*XJA11)*WCL[j]+F1*XJA2*XLO*WC[j] 
            
            Gii[1]=Gii[1]+S1
            Gii[2]=Gii[2]+S2
            Gii[3]=Gii[3]+S3
    if case==3:
        X2=XG2-XG3
        Y2=YG2-YG3
        X1=XG1-XG3
        Y1=YG1-YG3
        A1=(X1-2*X2)*0.5
        B1=-X2
        A2=(Y1-2*Y2)*0.5
        B2=-Y2
    #Geometrical properties
        for j in range(1,10+1):
            XJA1=np.sqrt((2*A1-4*A1*GIL[j]-0.5*X1)**2+(2*A2-4*A2*GIL[j]-0.5*Y1)**2)*2
            XJA2=np.sqrt((2*A1*GI[j]-0.5*X1)**2+(2*A2*GI[j]-0.5*Y1)**2)
            XLO=-np.log(2*np.sqrt((A1*GI[j]+B1)**2+(A2*GI[j]+B2)**2))
             #shape function
            F3=0.5*GI[j]*(GI[j]+1.)
            F2=1.-GI[j]**2
            F1=0.5*GI[j]*(GI[j]-1.)
            FL3=GIL[j]*(2.*GIL[j]-1.)
            FL2=4.*GIL[j]*(1.-GIL[j])
            FL1=(GIL[j]-1.)*(2.*GIL[j]-1.)
            #integration3
            S3=FL1*XJA1*WCL[j]+F3*XJA2*XLO*WC[j]     
            S2=FL2*XJA1*WCL[j]+F2*XJA2*XLO*WC[j] 
            S1=FL3*XJA1*WCL[j]+F1*XJA2*XLO*WC[j]
            
            Gii[1]=Gii[1]+S1
            Gii[2]=Gii[2]+S2
            Gii[3]=Gii[3]+S3
            
    return Gii


######################## Solver Module-Matrix assemble and field point solve ########################
def build_matrix_quadratic(panels):
    #All variables start from 1
    debug=0

    NE=len(panels) #number of elements
    N=2*NE #number of nodes
    #H IS A SQUARE MATRIX (2*NE,2*NE); G IS RECTANGULAR (2*NE,3*NE)
    #Index from 1
    G=np.zeros((2*NE+1, 3*NE+1), dtype=float) #double node for flux term
    H=np.zeros((2*NE+1, 2*NE+1), dtype=float)
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
    
    #1. Compute Gij[1,2,3] Hij[1,2,3] for each node and BE
    for LL in range(1,N+1): #The index of node
        for i in range(1,N-1+1,2): #The index of first node
            Node2Panel=int((i+1)/2)-1
            if((LL-i)*(LL-i-1)*(LL-i-2)*(LL-i+N-2)!=0): #off-diagonal
                TEMP=GHCalc_quadratic(X[LL],Y[LL],panels[Node2Panel])
                #TEMP=GHCalc_quadratic_adapative(X[LL],Y[LL],panels[Node2Panel],'boundary')
                Hij,Gij=TEMP[0],TEMP[1]
            else:#Diagonal for Gii
                caseNo=LL-i+1
                if (LL==1) and (i==N-1):
                    caseNo=caseNo+N
                Hij=GHCalc_quadratic(X[LL],Y[LL],panels[Node2Panel])[0]
                #Hij=GHCalc_quadratic_adapative(X[LL],Y[LL],panels[Node2Panel],'boundary')[0]
                Gij=Gii_singular_quadratic(panels[Node2Panel],caseNo)
            for j in range(1,3+1):
                k=int(3*(i-1)/2)
                G[LL,k+j]=G[LL,k+j]+Gij[j]
                if (i-N+1==0):
                    if (j==3):
                        H[LL,1]=H[LL,1]+Hij[j]
                    else:
                        H[LL,i-1+j]=H[LL,i-1+j]+Hij[j]
                else:
                    H[LL,i-1+j]=H[LL,i-1+j]+Hij[j]
    
    
    #Compute the diagonal term
    for i in range(1,N+1):
        H[i,i]=0
        for j in range(1,N+1):
            if(i!=j):
                H[i,i]=H[i,i]-H[i,j]
        #For external problems:
        if (H[i,i]<0):
            H[i,i]=2*PI+H[i,i]
    
    H_origin=np.copy(H)
    G_origin=np.copy(G)


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
                            temp=H[k,2*i-2+j]
                            H[k,2*i-2+j]=-G[k,3*i-3+j]
                            G[k,3*i-3+j]=-temp
                        if(debug): print("G,Col%s<->H,Col%s"%(3*i-3+j,2*i-2+j))
                    else:#This is the first element and its first node and the previous ele's BC is Neumann, see Page 73
                        for k in range(1,N+1):
                            H[k,2*i-1]=H[k,2*i-1]-G[k,3*i-2]
                            G[k,3*i-2]=0
                    continue
                #Special case Dirichlet boundary is located at the last element
                if(debug): print('\nDirichlet on the last element')
                if KODE[1]>1e-6:#if the first node is Neumann
                    if(debug): print("Neumann")
                    for k in range(1,N+1):
                        temp=H[k,1]
                        H[k,1]=-G[k,3*i]
                        G[k,3*i]=-temp
                    if(debug): print("G,Col%s<->H,Col%s"%(3*i,1))
                    continue
                if KODE[1]<=1e-6:#if the first node is Dirichlet
                    if(debug): print('Dirichlet')
                    for k in range(1,N+1):
                        H[k,1]=H[k,1]-G[k,3*i]
                        G[k,3*i]=0
                    continue
    
    for i in range(1,NE+1):
        for j in range(1,3+1):
            if(debug): print('Ele%s Node%s BC:%s'%(i,j,KODE[3*(i-1)+j]))
            if (KODE[3*(i-1)+j]==1 and panels[i-1].Robin_alpha>0):#This Ele's BC is Dirichelt
                if (i-NE!=0) or (j!=3):#This is not the last(3) node of last element 
                    if (i==1) or (j>1) or (KODE[3*(i-1)]==1):#If boundary condition is Neumann then interchange G-H G*U=H*P
                        #print('--Ele%s Node%s BC:%s'%(i,j,KODE[3*(i-1)+j]))
                        for k in range(1,N+1):
                            H[k,2*i-2+j]+=panels[i-1].Robin_alpha*G[k,3*i-3+j]
                        if(debug): print("G,Col%s<->H,Col%s"%(3*i-3+j,2*i-2+j))
                    else:#This is the first element and its first node and the previous ele's BC is Neumann, see Page 73
                        for k in range(1,N+1):
                            H[k,2*i-1]+=panels[i-1].Robin_alpha*G[k,3*i-2]
                    continue
                #Special case Dirichlet boundary is located at the last element
                if(debug): print('\nDirichlet on the last element')
                if KODE[1]>1e-6:#if the first node is Neumann
                    if(debug): print("Neumann")
                    for k in range(1,N+1):
                        H[k,1]+=panels[i-1].Robin_alpha*G[k,3*i]
                    if(debug): print("G,Col%s<->H,Col%s"%(3*i,1))
                    continue
                if KODE[1]<=1e-6:#if the first node is Dirichlet
                    if(debug): print('Dirichlet')
                    for k in range(1,N+1):
                        H[k,1]+=panels[i-1].Robin_alpha*G[k,3*i]
                    continue


    H=np.delete(H,0,axis=1)
    H=np.delete(H,0,axis=0)
    A=H

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
                b[i]=b[i]+G[i,j]*DFI[j]
    b=np.delete(b,0,axis=0)
    #debug
    H_origin=np.delete(H_origin,0,axis=1)
    H_origin=np.delete(H_origin,0,axis=0)
    G_origin=np.delete(G_origin,0,axis=1)
    G_origin=np.delete(G_origin,0,axis=0) #axis=0 row axis=1 column
    G=np.delete(G,0,axis=1)
    G=np.delete(G,0,axis=0)
    
    #print(H.shape)
    #print(H)
    #print(G.shape)
    #print(G)
    #print(A)
    return A,b,G_origin,H_origin,G

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

    if (elementID==-1): #query point locate on internal domain
    
        for i in range(1,NE+1):                   #  0   1   2   3   4   5
            pl=panels[i-1]

            #A=GHCalc__adaptive(xi,yi,panels[i-1],bd_int='internal') # Hij,Gij,DUx,DQx,DUy,DQy
            A=GHCalc_quadratic_adapative(xi,yi,pl,'internal') # Hij,Gij,DUx,DQx,DUy,DQy

            Hij,Gij=A[0],A[1]
            DUx,DQx=A[2],A[3]
            DUy,DQy=A[4],A[5]
        
            Q=np.zeros(3+1, dtype=float)
            P=np.zeros(3+1,dtype=float)
            pl=panels[i-1]
            if i==NE:
                pl_next=panels[0]
            else:
                pl_next=panels[i]
            
            Q[1],Q[2],Q[3]=pl.Q1,pl.Q2,pl.Q3
            P[1],P[2],P[3]=pl.P1,pl.P2,pl_next.P1
            for j in range(1,3+1):
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


def calcP_bd(Pts,Element):
    #calculate the p on a specific element
    phi=Element.get_ShapeFunc(Pts)
    Pi=[Element.P1,Element.P2,Element.P3]
    
    p=0.0
    for i in range(3):
        p+=phi[i]*Pi[i]

    return p

def calcUV_bd(Pts,Element,P_Pts=-1,backward=0):
    #calculate the uv on a specific element using (Pts2-Pts)/dist
    #Default forward:Pts .--. Pts2 backward:Pts2 .--. Pts

    dist=0.000001

    Pts=np.array(Pts)
    unit_vector=np.array((Element.tx,Element.ty))

    if (P_Pts==-1):
        P_Pts=calcP_bd(Pts,Element)

    if(backward!=1):#forward
        Pts2=Pts+dist*unit_vector
        temp=(calcP_bd(Pts2,Element)-P_Pts)/dist
        
    else:#backward
        Pts2=Pts-dist*unit_vector
        temp=(P_Pts-calcP_bd(Pts2,Element))/dist

    u=temp*Element.tx
    v=temp*Element.ty

    return u,v


    

