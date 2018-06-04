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

def GHCalc_subdivision(xi,yi,panelj,bd_int):
    
    #Boundary solution variable
    Hij=0  #H matrix value-velocity
    Gij=0  #G matrix value-pressure
    
    #Internal solution variable
    DU1=0  #G derivative in x direction
    DU2=0  #G derivative in y direction
    DQ1=0  #H derivative in x direction
    DQ2=0  #H derivative in y direction
    
    #BE element information
    X1,Y1=panelj.xa,panelj.ya
    X2,Y2=panelj.xc,panelj.yc
    X3,Y3=panelj.xb,panelj.yb
    length=panelj.length
    unit_vx=(X3-X1)/length
    unit_vy=(Y3-Y1)/length

    
    #Element subdivision+adapative gauss point selection
    Param1=Subdivision(xi,yi,X1,Y1,X3,Y3,3e-5,0,bd_int) #parameter [Subdivision trigger,NG_required]
    #No_sub=Param1[2]
    Trigger_sub=Param1[0]
    
    GaussOrder=[]
    if Trigger_sub==1:
        length_sub=[]
        No_sub=0
        Cum_length=0

        X1_sub,Y1_sub=X1,Y1 #the start point of the first segment
        X3_sub,Y3_sub=X3,Y3
        for itera in range(8000):
            Param2=Subdivision(xi,yi,X1_sub,Y1_sub,X3_sub,Y3_sub,3e-5,1,bd_int)
            GaussOrder.append(Param2[1])
            length_sub.append(Param2[2]) 
            Cum_length=Cum_length+length_sub[No_sub]
            X1_sub=X1_sub+length_sub[No_sub]*unit_vx
            Y1_sub=Y1_sub+length_sub[No_sub]*unit_vy
            No_sub=No_sub+1
            #print(X1_sub,X3_sub)
            if (Cum_length>=length-0.0001):
                #print('-------Subelement----%s'%(No_sub))
                temp=0
                for i in range(No_sub):
                    temp=temp+length_sub[i]
                    #print('No.%s length:%s NG pts:%s'%(i+1,length_sub[i],GaussOrder[i]))
                #print('Cum_length:%s'%(temp))
                break
        #store the endpoint of each sub-element in [-1,1] local coordinates
        endpoint_sub=Global2Iso(length_sub,length,2)
    elif Trigger_sub==0:
            GaussOrder.append(Param1[1])
            No_sub=1
       
    #Modified procedure to calculate H,G matrix
    for m in range(No_sub): #loop for each subelement      
        GI,WC=GaussLib(GaussOrder[m])[0],GaussLib(GaussOrder[m])[1] #Gauss point and weight
    
        Xj_G=0 #Gauss point transfer
        Yj_G=0 #Gauss point transfer
        Rij=0  #Distance from node to Gauss point on boundary element
        Rd1=0  #Radius derivatives 
        Rd2=0  #Raidus derivatives
        Rd=0   #Rd=nx*Rd1+ny*Rd2
        JAC_bar=1#subdivision jacobian transoform for local and global coordinate
             #if Trigger_sub==1: JAC_bar=0.5*(x3_sub-x1_sub) #John. T.Katsikadelis p146   Gernot Beer p148
        
        for j in range(1,GaussOrder[m]+1):
            if Trigger_sub==1:
            #endpoint_sub=Global2Iso(length_sub,length,3) #start, end point for the sub region in local coordinate(-1,1)
            #print(endpoint_sub)
                x1_sub,x3_sub=endpoint_sub[m],endpoint_sub[m+1]
                GI[j]=0.5*(x3_sub+x1_sub)+0.5*(x3_sub-x1_sub)*GI[j]  #Eq. 5.87 #John. T.Katsikadelis p138 
                JAC_bar=0.5*(x3_sub-x1_sub)                          #Eq. 5.87 #John. T.Katsikadelis p138
            
            Xj_G=(X3-X1)*GI[j]/2+(X3+X1)/2
            Yj_G=(Y3-Y1)*GI[j]/2+(Y3+Y1)/2
            Rij=np.sqrt((xi-Xj_G)**2+(yi-Yj_G)**2)
            Rd1=(Xj_G-xi)/Rij
            Rd2=(Yj_G-yi)/Rij
            Rd=Rd1*panelj.nx+Rd2*panelj.ny
            
            Hij=Hij-Rd/Rij*WC[j]*panelj.length/2*JAC_bar
            Gij=Gij+np.log(1/Rij)*WC[j]*panelj.length/2*JAC_bar
            
            DU1=DU1+Rd1/Rij*WC[j]*panelj.length/2*JAC_bar
            DU2=DU2+Rd2/Rij*WC[j]*panelj.length/2*JAC_bar
            DQ1=DQ1-(2*Rd1*Rd2*panelj.ny+(2*Rd1**2-1)*panelj.nx)/(Rij**2)*WC[j]*panelj.length/2*JAC_bar
            DQ2=DQ2-(2*Rd1*Rd2*panelj.nx+(2*Rd2**2-1)*panelj.ny)/(Rij**2)*WC[j]*panelj.length/2*JAC_bar
            
            #print('(%s,%s)'%(Xj_G,Yj_G))
            #print(Hij)


    return Hij,Gij,DU1,DQ1,DU2,DQ2

def GHCalc_analytical(xi,yi,panelj):
    #The algorithm is followed with Yijun Liu FMBEM book-code, p178
    #Appendix.A in H-F G-G DU-K DQ-H in Yijun Liu's book
    PI=3.141592653

    #Boundary solution variable
    Hij=0  #H matrix value-velocity
    Gij=0  #G matrix value-pressure
    
    #Internal solution variable
    DU1=0  #G derivative in x direction
    DU2=0  #G derivative in y direction
    DQ1=0  #H derivative in x direction
    DQ2=0  #H derivative in y direction
    
    #BE geometry element information
    X1,Y1=panelj.xa,panelj.ya
    X2,Y2=panelj.xc,panelj.yc
    X3,Y3=panelj.xb,panelj.yb
    length=panelj.length
    dnorm_x=panelj.nx
    dnorm_y=panelj.ny
    #tnorm_x=(X3-X1)/length
    #tnorm_y=(Y3-Y1)/length
    tnorm_x=panelj.tx
    tnorm_y=panelj.ty
    
    #print('BE(%s,%s)-(%s,%s)'%(X1,Y1,X3,Y3))
    #panel(pts a,pts b) node(xi,yi)
    distx_a=X1-xi
    disty_a=Y1-yi
    distx_b=X3-xi
    disty_b=Y3-yi
    
    r1=np.sqrt(distx_a**2+disty_a**2)
    r2=np.sqrt(distx_b**2+disty_b**2)
    d=distx_a*dnorm_x+disty_a*dnorm_y
    t1=-distx_a*dnorm_y+disty_a*dnorm_x
    t2=-distx_b*dnorm_y+disty_b*dnorm_x
    
    
    ds=abs(d)
    theta1=np.arctan2(t1,ds)
    theta2=np.arctan2(t2,ds)
    dtheta=theta2-theta1
    
    if (r1==0 or r2==0): #for the case of edge node concide with element node
        print('node',xi,yi)
        print('BE(%s,%s)-(%s,%s)'%(X1,Y1,X3,Y3))
        Hij=PI #Hii
        Gij=length*(np.log(2/length)+1) #Gii
    else: #normal case
        Gij=(-dtheta*ds + length + t1*np.log(r1)-t2*np.log(r2))
        if d<-0.0000000001:
            dtheta=-dtheta
        Hij=-dtheta
    
    #My derivation
    #DU1=dnorm_x*dtheta+dnorm_y*np.log(np.cos(theta2)/np.cos(theta1)) #DU in x
    #DU2=dnorm_y*dtheta-dnorm_x*np.log(np.cos(theta2)/np.cos(theta1)) #DU in y
    
    #Appendix.A in Yijun Liu's book
    DU1=dnorm_x*dtheta+tnorm_x*np.log(r2/r1) #DU in x
    DU2=dnorm_y*dtheta+tnorm_y*np.log(r2/r1) #DU in y
    
    '''
    #My poor derivation
    #print('dnorm',dnorm_x,dnorm_y,d)
    #print('tnorm',tnorm_x,tnorm_y,d)
    dcos=np.cos(2*theta2)-np.cos(2*theta1)
    dsin=np.sin(2*theta2)-np.sin(2*theta1)
    
    if d!=0:
        if (d<-0.000000001):#outside
            DQ1=1/2/ds*(-dnorm_x*dsin+dnorm_y*dcos)
            DQ2=1/2/ds*(-dnorm_x*dcos-dnorm_y*dsin)
        else: #inside
            DQ1=1/2/ds*(-dnorm_x*dsin-dnorm_y*dcos)
            DQ2=1/2/ds*(dnorm_x*dcos-dnorm_y*dsin)
    else: #d==0
        #print("d==0")
        DQ1=1/length
        DQ2=1/length
    '''

    #Appendix.A in Yijun Liu's book
    DQ1=-(t2/r2**2-t1/r1**2)*dnorm_x+ds*(1/r2**2-1/r1**2)*tnorm_x
    DQ2=-(t2/r2**2-t1/r1**2)*dnorm_y+ds*(1/r2**2-1/r1**2)*tnorm_y

    return Hij,Gij,DU1,DQ1,DU2,DQ2



######################## Solver Module-Matrix assemble and field point solve ########################
def build_matrix_const(panels, DDM=0, AB=[]):

    if(DDM == 1 and AB != 'none'):
        return update_matrix_const(panels, AB)

    N=len(panels)
    G=np.empty((N, N), dtype=float)
    H=np.empty((N, N), dtype=float)
    PI=3.141592653

    #Assemble G,H matrix
    for i, p_i in enumerate(panels): #nodes
        for j, p_j in enumerate(panels): #BEs
            if i==j:
                G[i,j]=p_i.length*(np.log(2/p_i.length)+1)
                H[i,j]=PI #Hii=1/2*2PI due to intergral=0
            if i!=j:
                xi,yi=p_i.xc,p_i.yc
                #TEMP=GHCalc(xi,yi,p_j)
                #TEMP=GHCalc_subdivision(xi,yi,p_j,'internal')
                TEMP=GHCalc_analytical(xi,yi,p_j)
                H[i,j]=TEMP[0]
                G[i,j]=TEMP[1]
    
    H_origin=np.copy(H)
    G_origin=np.copy(G)
    
    #Boundary Condition enforcement
    #Assemble matrix A and reorder matrix H and G (Switching column)            
    for j in range(N):
        if (panels[j].bd_Indicator==0):#If boundary condition is Dirichlet then interchange G-H G*U=H*P
            for i in range(N):
                temp=H[i,j]
                H[i,j]=-G[i,j]
                G[i,j]=-temp
    
    #Robin Boundary Condition 
    #Reference https://www.researchgate.net/project/Extending-the-boundary-element-method-to-the-generalised-Robin-boundary-condition
    for j in range(N):
        for i in range(N):
            if (panels[j].bd_Indicator==1 and panels[j].Robin_alpha>0):#If boundary condition is Neumann then interchange G-H G*U=H*P
                H[i,j]+=panels[j].Robin_alpha*G[i,j]#beta*G
    #print(H)
    #print(G)
    A=H
    #Assemble vector b
    b=np.empty(N, dtype=float)
    for i in range(N): #BE
        b[i]=0
        for j in range(N): #BEs
            b[i]=b[i]+G[i,j]*panels[j].bd_value1

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
            puv=GHCalc_analytical(xi,yi,p_j)
            p=p+p_j.Q1*puv[1]-p_j.P1*puv[0]
            u=u+p_j.Q1*puv[2]-p_j.P1*puv[3]
            v=v+p_j.Q1*puv[4]-p_j.P1*puv[5]
        
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


