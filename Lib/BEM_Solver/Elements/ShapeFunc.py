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

from Lib.Tools.Geometry import *
import numpy as np
from numba import jit

@jit(nopython=True)
def ShapeFunc(xi=0.0,d1=1.0,d2=1.0,order=1):
    """Calculation Shape function term for higher-order element
        Reference: Eq 3.36-3.37, Page51 on The Boundary Element Method with Programming,2008
        
             0         1
        |---|*|-------|*|---|  Linear element
             0    1    2
        |---|*|--|*|--|*|---|  Quad element

        Arguments
        ---------
        xi      -- local coordinates within a element [-1,1], greek /xi
        order   -- 1 order=linear element 2 order=quad element
        disct   -- indicator of discontinuous element
        d1/d2   -- offset distance from pts_a/pts_b in discontinous element [0,1]
                   normal shape function is a special case of discontinuous(d1=d2=1)
                   So, default type is continuous element

    Author:Bin Wang(binwang.0213@gmail.com)
    Date: July. 2017
    """
    
    if(order==1):#linear element
        phi=np.zeros(2)
        phi[0]=1/(d1+d2)*(d2-xi)
        phi[1]=1/(d1+d2)*(d1+xi)
    if(order==2):#Quad element
        phi=np.zeros(3)
        phi[0]=1/(d1+d2)/d1*(-d2+xi)*xi
        phi[1]=1/d1/d2*(d1+xi)*(d2-xi)
        phi[2]=1/(d1+d2)/d2*(d1+xi)*xi
    
    return phi

@jit(nopython=True) 
def ShapeFunc_Interp(Variable,Weight): #slow, not recommend
    """Calculation Shape function interpolation 
       Example:
       Interp=Variable[0]*Weight[0]+Variable[1]*Weight[1]....
       
       Variable can be multiple variables in each row
       Input:        Output:
       [[x1,x2,x3]   [x_interp,y_interp]
        [y1,y2,y3]]
       
    Author:Bin Wang(binwang.0213@gmail.com)
    Date: July. 2017
    """
    return np.dot(Variable,Weight)

@jit(nopython=True)
def Jacobian(xi=0.0,Pts_a=(0.0,0.0),Pts_c=(0.5,0.0),Pts_b=(1.0,0.0),order=1):
    """Calculation Jacobian term for higher-order element
        Reference: Eq 2.78, Page91 on The Boundary Elements An Introduction Course,1991 (Brebbia)
                   Eq 5.68, Page140 on BEM theory and application,2016 (Katsikadelis)
                   Eq 3.49, Page55 on The Boundary Element Method with Programming,2008
        Jacobian is the |norm| of the normal vector

        The jacobian is the same on both continuous and discontnuous elements

        J=sqrt((dx/dxi)^2+(dy/dxi)^2) for 2D problem

        Linear:
        x(xi)=b0+b1*xi  dx/dxi=b1=(xb-xa)/2
        y(xi)=c0+c1*xi  dy/dxi=c1=(yb-ya)/2
        J=Length/2

        Quad:
        x(xi)=b0+b1*xi+b2*(xi^2)    dx/dxi=b1+2*b2*xi
        y(xi)=c0+c1*xi+c2*(xi^2)    dy/dxi=c1+2*c2*xi

    Author:Bin Wang(binwang.0213@gmail.com)
    Date: July. 2017
    """
    Jac=0.0

    if(order==1):
        Jac=calcDist(Pts_a,Pts_b)/2
    if(order==2):
        b1=(Pts_b[0]-Pts_a[0])/2
        b2=(Pts_a[0]-2*Pts_c[0]+Pts_b[0])/2
        c1=(Pts_b[1]-Pts_a[1])/2
        c2=(Pts_a[1]-2*Pts_c[1]+Pts_b[1])/2
        dxdxi=b1+2*b2*xi
        dydxi=c1+2*c2*xi
        Jac=np.sqrt(dxdxi**2+dydxi**2)
    return Jac

@jit(nopython=True)
def Jac_NT(xi=0.0,Pts_a=(0.0,0.0),Pts_c=(0.5,0.0),Pts_b=(1.0,0.0),order=1):
    """Calculation Jacobian, unit Norm and Tangent vector term for higher-order element (Curve shape)
        Reference: Eq 3.49, Page55 on The Boundary Element Method with Programming,2008
                   
        Jacobian is the |norm| of the normal vector

        The jacobian is the same on both continuous and discontnuous elements

        J=sqrt((dx/dxi)^2+(dy/dxi)^2) for 2D problem

        Linear:
        x(xi)=b0+b1*xi  dx/dxi=b1=(xb-xa)/2   nx=(dy/dxi)/Jac=(yb-ya)/Length     tx=-ny
        y(xi)=c0+c1*xi  dy/dxi=c1=(yb-ya)/2   ny=-(dx/dxi)/Jac=-(xb-xa)/Length   ty=nx
        Jac=Length/2

        Quad:
        x(xi)=b0+b1*xi+b2*(xi^2)    dx/dxi=b1+2*b2*xi
        y(xi)=c0+c1*xi+c2*(xi^2)    dy/dxi=c1+2*c2*xi

    Author:Bin Wang(binwang.0213@gmail.com)
    Date: July. 2017
    """
    Jac=0.0

    if(order==1):
        Jac=calcDist(Pts_a,Pts_b)/2
        dxdxi=(Pts_b[0]-Pts_a[0])/2
        dydxi=(Pts_b[1]-Pts_a[1])/2
    if(order==2):
        b1=(Pts_b[0]-Pts_a[0])/2
        b2=(Pts_a[0]-2*Pts_c[0]+Pts_b[0])/2
        c1=(Pts_b[1]-Pts_a[1])/2
        c2=(Pts_a[1]-2*Pts_c[1]+Pts_b[1])/2
        dxdxi=b1+2*b2*xi
        dydxi=c1+2*c2*xi
        Jac=np.sqrt(dxdxi**2+dydxi**2)
    #Calculate norm and tangent vector
    nx=dydxi/Jac
    ny=-dxdxi/Jac
    tx=-ny
    ty=nx

    return Jac,(nx,ny),(tx,ty)


@jit(nopython=True)
def calcDist(Pts0=(0,0),Pts1=(1,1)):
    '''Calculating distance of two points
    '''
    return np.sqrt((Pts1[0]-Pts0[0])**2+(Pts1[1]-Pts0[1])**2)

@jit(nopython=True)
def GaussLib2(Gaussorder):
    '''Gauss point in numerical integration
    '''
    if (Gaussorder==2):
        GI=[-0.5773502691896257,0.5773502691896257]
        WC=[1.,1.]
    elif (Gaussorder==3):
        GI=[-0.7745966692414834 ,0.,0.7745966692414834]
        WC=[0.5555555555555556,0.8888888888888889,0.5555555555555556] 
    elif (Gaussorder==4):
        GI=[-0.8611363115940526,-0.3399810435848563,0.3399810435848563,0.8611363115940526]
        WC=[0.3478548451374538,0.6521451548625461,0.6521451548625461,0.3478548451374538]
    elif (Gaussorder==5):
        GI=[-0.9061798459386640 ,-0.5384693101056831,  0.  ,-0.5384693101056831, 0.9061798459386640]
        WC=[0.2369268850561891 , 0.4786286704993665 , 0.5688888888888889 , 0.4786286704993665 , 0.2369268850561891] 
    elif (Gaussorder==6):
        GI=[-0.9324695142031521 ,-0.6612093864662645, -0.2386191860831969 , 0.2386191860831969 , 0.6612093864662645 , 0.9324695142031521]
        WC=[0.1713244923791704 ,0.3607615730481386 ,0.4679139345726910 ,0.4679139345726910, 0.3607615730481386 , 0.1713244923791704]
    elif (Gaussorder==7):
        GI=[-0.9491079123427585, -0.7415311855993945, -0.4058451513773972 , 0.     
            ,0.4058451513773972, 0.7415311855993945 ,0.9491079123427585]
        WC=[0.1294849661688697 , 0.2797053914892766 , 0.3818300505051189 , 0.4179591836734694 
            , 0.3818300505051189 , 0.2797053914892766,0.1294849661688697]
    elif (Gaussorder==8):
        GI=[-0.9602898564975363, -0.7966664774136267, -0.5255324099163290, -0.1834346424956498 
            , 0.1834346424956498 , 0.5255324099163290, 0.7966664774136267 , 0.9602898564975363]
        WC=[0.1012285362903763 , 0.2223810344533745 ,0.3137066458778873 , 0.3626837833783620
            ,0.3626837833783620 ,0.3137066458778873,0.2223810344533745 , 0.1012285362903763]
    elif (Gaussorder==9):
        GI=[-0.9681602395076261, -0.8360311073266358 ,-0.6133714327005904 ,-0.3242534234038089, 0.     
            ,0.3242534234038089,0.6133714327005904 , 0.8360311073266358 , 0.9681602395076261]
        WC=[0.0812743883615744 , 0.1806481606948574 , 0.2606106964029354 ,  0.3123470770400029 , 0.3302393550012598 
            , 0.3123470770400029, 0.2606106964029354  , 0.1806481606948574 , 0.0812743883615744]
    elif (Gaussorder==10):
        GI=[-0.9739065285171717 ,-0.8650633666889845 ,-0.6794095682990244, -0.4333953941292472, -0.1488743389816312 
            , 0.1488743389816312,0.4333953941292472 , 0.6794095682990244 , 0.8650633666889845 , 0.9739065285171717]
        WC=[0.0666713443086881 , 0.1494513491505806 ,0.2190863625159820 , 0.2692667193099963 , 0.2955242247147529 
            , 0.2955242247147529,0.2692667193099963 , 0.2190863625159820 , 0.1494513491505806 , 0.0666713443086881]
    
    return GI,WC





    