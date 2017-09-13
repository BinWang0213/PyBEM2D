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
from .ShapeFunc import *
from numba import jit


@jit(nopython=True) #23X faster than original one
def Integr2D_GHdGdH(Pts_i,Pts_a,Pts_c,Pts_b,N_GaussPts=[],Sub_Pts=[],d1=1.0,d2=1.0):
    """Calculation of non-singular intergration(off-diagonal term) in CBIE
       Reference: Eq 3.49, Page55 on The Boundary Element Method with Programming,2008
        
       2D kernel integration with adaptive element subdivision 
    
             a    c    b
        |---|*|--|*|--|*|---|  Discontinuous Quad element
    
    order=2 Quad element
    
    Arguments
    ---------
    Integration side -- notation j
    Target side      -- notation i
    phi              -- Element shape function
    Kernel           -- G,H,dG,dH kernel which varies with PDE
    Jac              -- Jacobian to convert global coords(curve) to local coords(line)[-1,1]
    Jac_bar          -- Jacobian to convert local coords[0.2,0.3] to sub-element coords[-1,1]
    GI,WC            -- Numerical integration points and corrsponding weights 
    order            -- Order of the element, 1-linear element 2-quadratic element
    Num_nodes        -- Number of nodes in a element
    
    Integration={phi[xi]*Kernel[i,j]*Jac[xi]*Jac_bar[xi]*WC} within [-1,1]
    
    Author:Bin Wang(binwang.0213@gmail.com)
    Date: July. 2017
    """
    X1,Y1=Pts_a[0],Pts_a[1]
    X2,Y2=Pts_c[0],Pts_c[1]
    X3,Y3=Pts_b[0],Pts_b[1]
    xi,yi=Pts_i[0],Pts_i[1]
    xjG,yjG=0.0,0.0

    
    Num_nodes=3
    Hij=np.zeros(Num_nodes)  #H matrrix value -velocity in terms of shape fucntion1,2,3 
    Gij=np.zeros(Num_nodes)  #G matrrix value -pressure in terms of shape fucntion1,2,3 
    DUx=np.zeros(Num_nodes)  #G derivative in x direction shape function1,2,3
    DUy=np.zeros(Num_nodes)  #G derivative in y direction shape function1,2,3
    DQx=np.zeros(Num_nodes)  #H derivative in x direction shape function1,2,3
    DQy=np.zeros(Num_nodes)  #H derivative in y direction shape function1,2,3
    
    N_sub=len(N_GaussPts)
    for s in range(N_sub):
        GI,WC=GaussLib2(N_GaussPts[s])#Get the Gauss point for each subdivision
        for j in range(N_GaussPts[s]):
            Jac_bar=1
            GI_new=GI[j]
            if(N_sub>1):#Subdivision integration
                x1_sub,x3_sub=Sub_Pts[s],Sub_Pts[s+1]
                GI_new=0.5*(x3_sub+x1_sub)+0.5*(x3_sub-x1_sub)*GI[j]
                Jac_bar=0.5*(x3_sub-x1_sub)
            #Normal integration
            phi=ShapeFunc(GI_new,d1,d2,order=2)
            XjG,YjG=X1*phi[0]+X2*phi[1]+X3*phi[2],Y1*phi[0]+Y2*phi[1]+Y3*phi[2]
            Jac_nxny_txty=Jac_NT(GI_new,Pts_a,Pts_c,Pts_b,order=2)#Jacobian,nx,ny,tx,ty
            Jac=Jac_nxny_txty[0]
            nx,ny=Jac_nxny_txty[1]
            Rij=np.sqrt((xi-XjG)**2+(yi-YjG)**2) #(1/r) ln(1/r) in kernel
            drdx,drdy=(XjG-xi)/Rij,(YjG-yi)/Rij
            Rd=drdx*nx+drdy*ny

            for k in range(Num_nodes):#node number=element order+1
                Hij[k]=Hij[k]-phi[k]*Rd/Rij*WC[j]*Jac*Jac_bar
                Gij[k]=Gij[k]+phi[k]*np.log(1/Rij)*WC[j]*Jac*Jac_bar
        
                DUx[k]=DUx[k]+phi[k]*drdx/Rij*WC[j]*Jac*Jac_bar
                DUy[k]=DUy[k]+phi[k]*drdy/Rij*WC[j]*Jac*Jac_bar
                DQx[k]=DQx[k]-phi[k]*(2*drdx*drdy*ny+(2*drdx**2-1)*nx)/(Rij**2)*WC[j]*Jac*Jac_bar
                DQy[k]=DQy[k]-phi[k]*(2*drdx*drdy*nx+(2*drdy**2-1)*ny)/(Rij**2)*WC[j]*Jac*Jac_bar
    
    return Hij,Gij,DUx,DQx,DUy,DQy

@jit(nopython=True) #23X faster than original one
def Integr2D_Gii(NodeID,Pts_a,Pts_c,Pts_b,d1=1.0,d2=1.0):
    """Calculation of singular intergration(diagonal term) in CBIE
       Hii=-sum(Hij) Gii=subdivide logarithmic integration

       Reference:
       Eq 12.8, Page333 on The Boundary Element Method with Programming,2008
       Eq 2.99, Page102 on Boundary Elements An Introductory Course

       2D kernel integration with adaptive element subdivision 
    
             a    c    b
        |---|*|--|*|--|*|---|  Discontinuous Quad element
             1    2    3  
       
    Arguments
    ---------
    Integration side -- notation j
    Target side      -- notation i
    phi              -- Element shape function
    Kernel           -- G,H,dG,dH kernel which varies with PDE
    Jac              -- Jacobian to convert global coords(curve) to local coords(line)[-1,1]
    Jac_bar1         -- Jacobian to convert local coords[0.2,0.3] to sub-element coords[0,1]
    Jac_bar2         -- Jacobian to convert sub-element [0,1] to [-1,1] sub-element
    GI,WC            -- Numerical integration points and corrsponding weights 
    order            -- Order of the element, 1-linear element 2-quadratic element
    Num_nodes        -- Number of nodes in a element
    d1,d2            -- offset distance for Discontinuous element,d1=d2=1 is continuous element
    
    sider            -- Switcher for local Gauss point equation(left(-1),right(1))
    
    Integration={phi[xi]*Kernel[i,j]*Jac[xi]*Jac_bar[xi]*WC} within [-1,1]
    
    Author:Bin Wang(binwang.0213@gmail.com)
    Date: July. 2017
    """
    #Gauss quad integration with n=10
    GI=[-0.9739065285171717,-0.8650633666889845,-0.6794095682990244,-0.4333953941292472,-0.1488743389816312 
        , 0.1488743389816312,0.4333953941292472, 0.6794095682990244,0.8650633666889845,0.9739065285171717]
    WC=[0.0666713443086881,0.1494513491505806,0.2190863625159820,0.2692667193099963,0.2955242247147529 
        ,0.2955242247147529,0.2692667193099963, 0.2190863625159820,0.1494513491505806,0.0666713443086881]
    #Logarithmic quad integration with n=10
    GIL=[0.0090426309,0.0539712662,0.1353118246,0.2470524162,0.3802125396
         ,0.5237923179,0.6657752055,0.7941904160,0.8981610912,0.9688479887]
    WCL=[0.1209551319,0.1863635425,0.1956608732,0.1735771421,0.1356956729
         ,0.0936467585,0.0557877273,0.0271598109,0.0095151826,0.0016381576]
    
    X1,Y1=Pts_a[0],Pts_a[1]
    X2,Y2=Pts_c[0],Pts_c[1]
    X3,Y3=Pts_b[0],Pts_b[1]
    
    Num_nodes=3
    Gii=np.zeros(Num_nodes)  #G matrrix value -pressure in terms of shape fucntion1,2,3 
    
    ###Gauss Part###
    N_sub=2#The integration performed on two sub-region(left,right)
    for s in range(N_sub):
        for j in range(10):
            if(s==0): sider=1#left
            if(s==1): sider=-1#right
                
            if(NodeID==1):#Singular at Node 1    
                GI_new=-d1+0.5*(1+d1*sider)*(1+GI[j]*sider)*sider
                Jac_bar1=d1+1*sider 
                Jac_bar2=0.5*sider
            elif(NodeID==2):#Singular at Node 2(middle)
                GI_new=0.5*(1+GI[j]*sider)*sider
                Jac_bar1=1*sider
                Jac_bar2=0.5*sider
            elif(NodeID==3):#Singular at Node 3
                GI_new=d2+0.5*(1-d2*sider)*(1+GI[j]*sider)*sider
                Jac_bar1=-d2+1*sider
                Jac_bar2=0.5*sider
            
            phi=ShapeFunc(xi=GI_new,d1=d1,d2=d2,order=2)
            Jac_nxny_txty=Jac_NT(GI_new,Pts_a,Pts_c,Pts_b,order=2)#Jacobian,nx,ny,tx,ty
            Jac=Jac_nxny_txty[0]
            Rij1=Jac*abs(Jac_bar1)#1991 version
            Rij=0.5*np.sqrt((X1-X3)**2+(Y1-Y3)**2)*abs(Jac_bar1)#2008 programming version
            #print('RRRij',Rij1,Rij)
            
            for k in range(Num_nodes):#node number=element order+1
                if(Jac_bar1!=0):#case d1 or d2==1
                    Gii[k]=Gii[k]+phi[k]*np.log(1/Rij)*WC[j]*Jac*Jac_bar1*Jac_bar2
        #print('Gauss')
        #print("GI",GI)
        #print('Phi',phi,Jac,Jac_bar1,Jac_bar2)
        #print(Gii)
    
    ###Gauss-Laguerre Part(log)###
    for s in range(N_sub):
        for j in range(10):
            if(s==0): sider=-1#left
            if(s==1): sider=1#right
                
            if(NodeID==1):#Singular at Node 1 
                GIL_new=-d1+(1+d1*sider)*GIL[j]*sider
                Jac_bar=1+d1*sider
            elif(NodeID==2):#Singular at Node 2(middle)
                GIL_new=GIL[j]*sider
                Jac_bar=1
            elif(NodeID==3):#Singular at Node 3
                GIL_new=d2+(1-d2*sider)*GIL[j]*sider
                Jac_bar=1-d2*sider
            
            phi=ShapeFunc(GIL_new,d1=d1,d2=d2,order=2)
            Jac_nxny_txty=Jac_NT(GIL_new,Pts_a,Pts_c,Pts_b,order=2)#Jacobian,nx,ny,tx,ty
            Jac=Jac_nxny_txty[0]
            
            for k in range(Num_nodes):#node number=element order+1
                if(Jac_bar!=0):#case d1 or d2==1
                    Gii[k]=Gii[k]+phi[k]*WCL[j]*Jac*Jac_bar
        #print('Gauss-Laguerre')
        #print("GIL",GIL)
        #print('phi',phi,Jac,Jac_bar)
        #print(Gii)
    
    return Gii

@jit(nopython=True) #23X faster than original one
def Subdivision_Indicator(Pts_i,Pts_a,Pts_b,TOL=1e-5,point_int=1):
    """Calculation of non-singular intergration(off-diagonal term) in CBIE
       Reference: Page80 on Advanced BEM method,2014(Xiaowei Gao)
       
    
    Arguments
    ---------
    point_int       -- point_type=1 target point on boundary, =2 on internal domain
                        called lamada in Gao et al's book
    TOL             -- required accurancy for near singular integration (1e-5,1e-8)
    NG_required     -- Estimated gauss point to achieve the required TOL
    
    Author:Bin Wang(binwang.0213@gmail.com)
    Date: July. 2017
    """
    
    X1,Y1=Pts_a[0],Pts_a[1]
    X3,Y3=Pts_b[0],Pts_b[1]
    xi,yi=Pts_i[0],Pts_i[1]
    
    #print(Subdivision(xi, yi, X1, Y1, X3, Y3, TOL, 0, "internal"))
    
    length = np.sqrt((X3 - X1) ** 2 + (Y3 - Y1) ** 2)
    mindist=Point2Segment(xi, yi, X1, Y1, X3, Y3)
    p_prime=np.sqrt(point_int * 2 / 3 + 2 / 5)
    
    if length > 3.9 * mindist: #length limiter
        length_temp = 3.9 * mindist  #special trick to trigger subdivision
    else:
        length_temp = length
    
    NG_required = p_prime * np.log(TOL / 2) * 0.5 / np.log(length_temp / 4 / mindist) #Eq. 3-6-79
    
    NG_required=NG_required+point_int#increase the subdivision probability
    #print(NG_required)
    
    if(NG_required>10):
        return True
    
    return False


@jit(nopython=True) #23X faster than original one
def Element_Subdivision(Pts_i,Pts_a,Pts_b,TOL=1e-5,point_int=2):
    """Calculation of non-singular intergration(off-diagonal term) in CBIE
        Reference: Page80 on Advanced BEM method,2014(Xiaowei Gao)

        a                       b
        |-------Pts_rest--------|  Element
                  -->
    Arguments
    ---------
    point_int       -- point_type=1 target point on boundary, =2 on internal domain
                        called lamada in Gao et al's book
    TOL             -- required accurancy for near singular integration (1e-5,1e-8)
    NG_required     -- Estimated gauss point to achieve the required TOL in a sub-element
    length_sub      -- Estimated length of a sub element
    N_GaussPts      -- [Output] Number of gauss points for each sub element e.g. [10,8,7,6]
    local_coords    -- [Output] Local coords(-1,1) for each sub element     e.g. [-1,-0.9,0.2,1.0]

    Author:Bin Wang(binwang.0213@gmail.com)
    Date: July. 2017
    """
    X1,Y1=Pts_a[0],Pts_a[1]
    X3,Y3=Pts_b[0],Pts_b[1]
    xi,yi=Pts_i[0],Pts_i[1]
        
    length_original = np.sqrt((X3 - X1) ** 2 + (Y3 - Y1) ** 2)#total length of element
    unit_vx=(X3-X1)/length_original #unit vector in element direction
    unit_vy=(Y3-Y1)/length_original
    
    Num_sub=0
    Cum_length=0.0
    N_GaussPts=[0]#for numba type infer
    Length_subs=[0.0]#for numba type infer
    N_GaussPts.pop(0)
    Length_subs.pop(0)
    
    X1_rest,Y1_rest=X1,Y1
    length_sub=0.0#length of sub element
    count=0
    length=0.0 #rest length of element
    while Cum_length<=length_original-0.0000001:
        
        #[Gao's algorithm]
        #print(Subdivision(xi,yi,X1_rest,Y1_rest,X3, Y3,1e-5,1,"internal"))
        
        length=np.sqrt((X3 - X1_rest) ** 2 + (Y3 - Y1_rest) ** 2)
        mindist=Point2Segment(xi, yi, X1_rest,Y1_rest, X3, Y3)
        p_prime=np.sqrt(point_int * 2 / 3 + 2 / 5)
        if length > 3.9 * mindist: #length limiter
            length_temp = 3.9 * mindist  #special trick to trigger subdivision
        else:
            length_temp = length
        
        NG_required = p_prime * np.log(TOL / 2) * 0.5 / np.log(length_temp / 4 / mindist) #Eq. 3-6-79
        NG_required = int(NG_required) + point_int*2+1#increase the subdivision probability
            
        #sub-interval length and division
        if NG_required > 10:  #subdivision trigger, 10=maximum gauss point in subdivision
            NG_required = 10  #maximum number of gauss point 
            length_sub = 4. * mindist * (TOL / 2.) ** (0.5 * p_prime / NG_required)  #minimum length of subdivision element
            #print('Required sublength:%s'%(length_sub))
            #print('Element length:%s'%(length))
            if length_sub >= length-0.0000001:  #if the required length > remainning length
                length_sub = length
        elif NG_required < 2:  #minimum gauss point=2 in sub division
            NG_required = 2
            length_sub = 4. * mindist * (TOL / 2.) ** (0.5 * p_prime / NG_required)#Eq. 3-6-79
            No_sub = int(length / length_sub + 0.95)
            length_sub = length / No_sub
        else:  #No sub-division is required
            No_sub = 1
            length_sub = length
        
        #[Collect sub_length and Gauss point]
        N_GaussPts.append(NG_required)
        Length_subs.append(length_sub)
        X1_rest+=Length_subs[count]*unit_vx
        Y1_rest+=Length_subs[count]*unit_vy
        
        Cum_length+=length_sub
        count+=1
        
    #[Convert length coords to element coords[-1,1]]
    local_coords=np.zeros(count+1)
    local_coords[0]=-1.0
    temp=0.0
    for i in range(count):
        temp=temp+Length_subs[i]
        local_coords[i+1]=-1+2*temp/length_original
    
    local_coords[-1]=1.0 #make sure the end of coords
    
    #print(N_GaussPts,Length_subs)
    return N_GaussPts,local_coords


#2.3X faster in plot solution
def GHCalc_quad_adaptive(xi,yi,panelj,bd_int="internal"):
    """Calculation of non-singular intergration(off-diagonal term) in CBIE
       For Quad element and adapative element subdivision
       
    Arguments
    ---------
    point_int        -- point_type=1 target point on boundary, =2 on internal domain
                        called lamada in Gao et al's book
    
    Author:Bin Wang(binwang.0213@gmail.com)
    Date: July. 2017
    """
    #Element J,integration part
    Pts_a=np.array([panelj.xa,panelj.ya])
    Pts_c=np.array([panelj.xc,panelj.yc])
    Pts_b=np.array([panelj.xb,panelj.yb])
    #Target Point I
    Pts_i=np.array([xi,yi])
    #Element subdivision coeff
    if(bd_int=="internal"):#Target point locate on internal domain
        pts_int=2
    elif(bd_int=="boundary"):
        pts_int=1
    
    #Step1. Element Subdivision Check
    Button=Subdivision_Indicator(Pts_i,Pts_a,Pts_b,TOL=1e-5,point_int=pts_int)
    #STep2. Integration
    if(Button==True):#Adaptive element subdivision integration
        N_GaussPts,Sub_Pts=Element_Subdivision(Pts_i,Pts_a,Pts_b,TOL=1e-5,point_int=2)
        return Integr2D_GHdGdH(Pts_i,Pts_a,Pts_c,Pts_b,N_GaussPts,Sub_Pts,d1=1.0,d2=1.0)
    elif(Button==False):#Const 10 Gauss point integration
        return Integr2D_GHdGdH(Pts_i,Pts_a,Pts_c,Pts_b,N_GaussPts=[10],Sub_Pts=[-1],d1=1.0,d2=1.0)
