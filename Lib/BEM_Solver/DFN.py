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

import numpy
import math
import time
from matplotlib import pyplot
from .Constant_element import GHCalc_analytical,GHCalc
from .Quadratic_element import GHCalc_quadratic_adapative,GHCalc_quadratic
from Lib.Tools.Geometry import Point2Segment,GaussLib,Global2Iso,Subdivision,point_in_panel


######################## Preprocess Module-Boundary Discretization ########################
def DiscretCheck_DFN_trace(x_length,y_length,interval,panels):
    #Constant element used to discrete the trace
    print("Number of boundary elements:%s" % (len(panels)))
    print("Coordinates&Boundary conditions of boundary elements:")
    print("Point\tX\t\tY\t\tBD_type\t\tBD_value")
    print("BD_type: Neumann=1 Dirichlet=0")
    for i, pl in enumerate(panels):
        #pyplot.annotate(i+1, (pl.xa+0.1,pl.ya+0.1),fontsize=12)
        print("%s\t%s\t\t%s\t\t%s\t\t%s " % (i+1,pl.xc,pl.yc,pl.bd_Indicator,pl.bd_value1))

def DiscretCheck_DFN_boundary(x_length,y_length,interval,panels):
    #Qu element used to discrete the trace
    print("Number of boundary elements:%s" % (len(panels)))
    print("Coordinates&Boundary conditions of boundary elements:")
    print("Point\tX\t\tY\t\tBD_type\t\tBD_value1\tBD_value2\tBD_value3")
    print("BD_type: Neumann=1 Dirichlet=0 Double node method to treat the corner issue")
    print("For Dirichlet P1=P2 For Neumann Flux1 != Flux2")
    for i, pl in enumerate(panels):
        #pyplot.annotate(2*i+1, (pl.xa+0.1,pl.ya+0.1),fontsize=12) #1,3,5...
        #pyplot.annotate(2*i+2, (pl.xc+0.1,pl.yc+0.1),fontsize=12) #2,4,6...
        print("(%s)%s\t%5.3f\t\t%.3f\t\t%.3s\t\t%.3f\t\t%.3f\t%.3f" % (i+1,2*i+1,pl.xa,pl.ya,pl.bd_Indicator,pl.bd_value1,pl.bd_value2,pl.bd_value3))
        print("(%s)%s\t%5.3f\t\t%.3f\t\t%.3s\t\t%.3f\t\t%.3f\t%.3f" % (i+1,2*i+2,pl.xc,pl.yc,pl.bd_Indicator,pl.bd_value1,pl.bd_value2,pl.bd_value3))
        print("(%s)%s\t%5.3f\t\t%.3f\t\t%.3s\t\t%.3f\t\t%.3f\t%.3f" % (i+1,'Xb',pl.xb,pl.yb,pl.bd_Indicator,pl.bd_value1,pl.bd_value2,pl.bd_value3))

def Discretplot_DFN(x_length,y_length,interval,panels,traces,Ntp):

    val_x, val_y = 0.2, 0.2
    x_min, x_max = min(panel.xa for panel in panels), max(panel.xa for panel in panels)
    y_min, y_max = min(panel.ya for panel in panels), max(panel.ya for panel in panels)
    x_start, x_end = x_min-val_x*(x_max-x_min), x_max+val_x*(x_max-x_min)
    y_start, y_end = y_min-val_y*(y_max-y_min), y_max+val_y*(y_max-y_min)

    size = 10
    pyplot.figure(figsize=(size, (y_end-y_start)/(x_end-x_start)*size))
    pyplot.grid(True)
    pyplot.xlabel('X (m)', fontsize=15)
    pyplot.ylabel('X (m)', fontsize=15)
    pyplot.xlim(x_start, x_end)
    pyplot.ylim(y_start, y_end)

    tick_interval=interval
    major_ticks_y = numpy.arange(0, x_length+0.1, tick_interval)                                              
    major_ticks_x = numpy.arange(0, y_length+0.1, tick_interval)                                              

    pyplot.xticks(major_ticks_x)
    pyplot.yticks(major_ticks_y)
    pyplot.tick_params(direction='in',labelsize=16)
    pyplot.grid(True)

    #Plot fracture 
    X=numpy.zeros(2*len(panels)+1)
    Y=numpy.zeros(2*len(panels)+1)
    for i, pl in enumerate(panels):
        X[2*i]=pl.xa
        X[2*i+1]=pl.xc
        Y[2*i]=pl.ya
        Y[2*i+1]=pl.yc
    X[-1]=X[0]
    Y[-1]=Y[0]

    pyplot.plot(X, Y, linestyle='-', linewidth=3, marker='o', markersize=5, color='#CD2305');    

    #annotate gap
    gap=x_length/60
    #for i, pl in enumerate(panels):
        #pyplot.annotate(2*i+1, (pl.xa+gap,pl.ya+gap),fontsize=12) #1,3,5...
        #pyplot.annotate(2*i+2, (pl.xc+gap,pl.yc+gap),fontsize=12) #2,4,6...

    #Plot trace
    Ntrace=len(traces) # Num of trace
    #Ntp number of element per trace
    for i in range(Ntrace):
         for j in range(Ntp):
            if j<Ntp-1:
                pyplot.plot([traces[i][j].xa,traces[i][j+1].xb],[traces[i][j].ya,traces[i][j+1].yb], 
                    linestyle='-', linewidth=3, marker='o', markersize=5, color='g');
            #pyplot.annotate(j+1, (traces[i][j].xa+gap,traces[i][j].ya+gap),fontsize=12) #1,3,5...
            #if j==Ntp-1:
                #pyplot.annotate(Ntp+1, (traces[i][j].xb+gap,traces[i][j].yb+gap),fontsize=12) #1,3,5...




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
    '''
    GI=[0.8611363116,0.3399810436,-0.3399810436,-0.8611363116]  #Gauss point
    WC=[0.3478548451,0.6521451549,0.6521451549,0.3478548451]    #weight cofficient
    '''
    GausOrder=10
        #Integration interval transoform
    Xj_G=numpy.empty(GausOrder) #Gauss point transfer
    Yj_G=numpy.empty(GausOrder) #Gauss point transfer
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
    DU1_dfn=0 #G for DFN
    DU2_dfn=0 #G for DFN

    for j in range(GausOrder):
        Xj_G[j]=(panelj.xb-panelj.xa)*GI[j]/2+(panelj.xb+panelj.xa)/2
        Yj_G[j]=(panelj.yb-panelj.ya)*GI[j]/2+(panelj.yb+panelj.ya)/2
        Rij=numpy.sqrt((xi-Xj_G[j])**2+(yi-Yj_G[j])**2)
        Rd1=(Xj_G[j]-xi)/Rij
        Rd2=(Yj_G[j]-yi)/Rij
        Rd=Rd1*panelj.nx+Rd2*panelj.ny
        Hij=Hij-Rd/Rij*WC[j]*panelj.length/2
        Gij=Gij+numpy.log(1/Rij)*WC[j]*panelj.length/2
        
        Gij_dfn=Gij_dfn+numpy.log(1/Rij)*WC[j]/2 #Gij for DFN requires Gij/panelj.length
        
        DU1=DU1+Rd1/Rij*WC[j]*panelj.length/2
        DU2=DU2+Rd2/Rij*WC[j]*panelj.length/2
        DQ1=DQ1-(2*Rd1*Rd2*panelj.ny+(2*Rd1**2-1)*panelj.nx)/(Rij**2)*WC[j]*panelj.length/2
        DQ2=DQ2-(2*Rd1*Rd2*panelj.nx+(2*Rd2**2-1)*panelj.ny)/(Rij**2)*WC[j]*panelj.length/2
        
        DU1_dfn=DU1_dfn+Rd1/Rij*WC[j]/2 #Coefficient for DFN requires divided by panelj.length
        DU2_dfn=DU2_dfn+Rd2/Rij*WC[j]/2 #Coefficient for DFN requires divided by panelj.length
        DQ1_dfn=DQ1_dfn-(2*Rd1*Rd2*panelj.ny+(2*Rd1**2-1)*panelj.nx)/(Rij**2)*WC[j]/2
        DQ2_dfn=DQ2_dfn-(2*Rd1*Rd2*panelj.nx+(2*Rd2**2-1)*panelj.ny)/(Rij**2)*WC[j]/2
        

    return Hij,Gij,DU1,DQ1,DU2,DQ2,Gij_dfn,DU1_dfn,DQ1_dfn,DU2_dfn,DQ2_dfn



######################## Solver Module-Matrix assemble and field point solve ########################
def build_matrix_DFN(panels,traces):
    NE=len(panels) #number of elements
    N=2*NE #number of nodes
    #H IS A SQUARE MATRIX (2*NE,2*NE); G IS RECTANGULAR (2*NE,3*NE)
    #Index from 1
    G=numpy.zeros((2*NE+1, 3*NE+1), dtype=float) #double node for flux term
    H=numpy.zeros((2*NE+1, 2*NE+1), dtype=float)
    PI=3.141592653
    
    #HEE
    #prepare X,Y for Book's program,index from 1 and N+1=1
    X=numpy.zeros(N+2) 
    Y=numpy.zeros(N+2)
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
                Hij=TEMP[0]
                #Gij=TEMP[1]
            else:#Diagonal for Gii
                caseNo=LL-i+1
                if (LL==1) and (i==N-1):
                    caseNo=caseNo+N
                Hij=GHCalc_quadratic(X[LL],Y[LL],panels[Node2Panel])[0]
                #Hij=GHCalc_quadratic_adapative(X[LL],Y[LL],panels[Node2Panel],'boundary')[0]
                #Gij=Gii_singular_quadratic(panels[Node2Panel],caseNo)
            for j in range(1,3+1):
                k=int(3*(i-1)/2)
                #G[LL,k+j]=G[LL,k+j]+Gij[j]
                if (i-N+1==0):#the N-1 node
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
    H=numpy.delete(H,0,axis=1)
    H=numpy.delete(H,0,axis=0)
    HEE=H
    

    #2. ATE-HTE-Influence factor from edge to trace
    Ne=2*NE
    Nt=len(traces)*len(traces[0])
    HTE=numpy.zeros((Nt+1, Ne+1), dtype=float)
    
    tii=1
    for t in range(len(traces)):# which trace
        for ti in range(len(traces[0])):#which node on a trace
            
            for j in range(NE):#store Hij into element
                Temp_Hij=GHCalc_quadratic_adapative(traces[t][ti].xc,traces[t][ti].yc,panels[j],'internal')[0]
                panels[j].H1=Temp_Hij[1]
                panels[j].H2=Temp_Hij[2]
                panels[j].H3=Temp_Hij[3]
            
            for j in range(NE):#assemble matrix
                pl_current=panels[j]
                if j==0:
                    pl_previous=panels[NE-1]
                else:
                    pl_previous=panels[j-1]
                HTE[tii,2*j+1]=pl_current.H1+pl_previous.H3
                HTE[tii,2*j+2]=pl_current.H2
            tii=tii+1#node index for trace

    HTE=numpy.delete(HTE,0,axis=1)
    HTE=numpy.delete(HTE,0,axis=0)        
    
    #3. AET-HET if the trace is not on the boundary-all is zero
    HET=numpy.zeros((Ne, Nt), dtype=float)
    
    #4. HTT-if the trace is not on the boundary-all is zero
    HTT=numpy.zeros((Nt, Nt), dtype=float)
    #compute the diagonal term
    for i in range(Nt):
        HTT[i,i]=0
        for j in range(Ne):
            HTT[i,i]=HTT[i,i]-HTE[i,j]
        for j in range(Nt):
            if(i!=j):
                HTT[i,i]=HTT[i,i]-HTT[i,j]
    
    #5. GET-BET
    GET=numpy.zeros((Ne+1, Nt+1), dtype=float)
    for i in range(NE):
        tii=1
        for t in range(len(traces)):# which trace
            for ti in range(len(traces[0])):#which node on a trace
                #Temp_Gij_odd=GHCalc(panels[i].xa,panels[i].ya,traces[t][ti])[1]/traces[t][ti].length
                #Temp_Gij_even=GHCalc(panels[i].xc,panels[i].yc,traces[t][ti])[1]/traces[t][ti].length
                Temp_Gij_odd=GHCalc_analytical(panels[i].xa,panels[i].ya,traces[t][ti])[1]/traces[t][ti].length
                Temp_Gij_even=GHCalc_analytical(panels[i].xc,panels[i].yc,traces[t][ti])[1]/traces[t][ti].length
                GET[2*i+1,tii]=Temp_Gij_odd  
                GET[2*i+2,tii]=Temp_Gij_even
                tii=tii+1
    GET=numpy.delete(GET,0,axis=1)
    GET=numpy.delete(GET,0,axis=0)  
    

    #6. GTT-BTT
    GTT=numpy.zeros((Nt+1, Nt+1), dtype=float)
    TII=1
    tii=1
    for T in range(len(traces)):
        for TI in range(len(traces[0])):
            tii=1
            for t in range(len(traces)):# which trace
                for ti in range(len(traces[0])):#which node on a trace
                    if tii!=TII:
                        #Temp_Gij=GHCalc(traces[T][TI].xc,traces[T][TI].yc,traces[t][ti])[6]
                        Temp_Gij=GHCalc_analytical(traces[T][TI].xc,traces[T][TI].yc,traces[t][ti])[1]/traces[t][ti].length
                        GTT[TII,tii]=Temp_Gij
                    #compute the diagonal term
                    if tii==TII:
                        #GTT[TII,tii]=traces[t][ti].length*(numpy.log(2/traces[t][ti].length)+1)/traces[t][ti].length
                        GTT[TII,tii]=(numpy.log(2/traces[t][ti].length)+1)
                    tii=tii+1
            TII=TII+1

        
    GTT=numpy.delete(GTT,0,axis=1)
    GTT=numpy.delete(GTT,0,axis=0)  
    
    
    #Conductance in paper
    Mat1=numpy.linalg.inv(GTT)
    Mat2=numpy.dot(Mat1,HTE)
    Mat3=numpy.dot(Mat1,HTT)
    Mat4=numpy.dot(GET,Mat2)
    Mat5=numpy.dot(GET,Mat3)
    Mat6=HEE-Mat4
    Mat6=numpy.linalg.inv(Mat6)
    Mat8=numpy.dot(Mat6,Mat5)
    Mat9=numpy.dot(Mat2,Mat8)
    Mat_C=Mat9+Mat3
    MatHe=numpy.dot(Mat6,Mat5)


    #Continuity diagonal term calculation
    for i in range(len(Mat_C)):
        Mat_C[i,i]=0
        for j in range(len(Mat_C)):
            if(i!=j):
                Mat_C[i,i]=Mat_C[i,i]-Mat_C[j,i]
    
    #BEM general matrix G and H (A,B)  A*head=B*flux
    """
    MatA_t1=numpy.concatenate((HEE,-GET),axis=1) #horizontal
    MatA_t2=numpy.concatenate((HTE,-GTT),axis=1) #horizontal
    MatA=numpy.concatenate((MatA_t1,MatA_t2),axis=0)#Vertical
    MatB=numpy.concatenate((-HET,-HTT),axis=0)
    """
    MatA_t1=numpy.concatenate((HEE,-GET),axis=1) #horizontal
    MatA_t2=numpy.concatenate((HTE,-GTT),axis=1) #horizontal
    MatA=numpy.concatenate((MatA_t1,MatA_t2),axis=0)#Vertical
    MatB=numpy.concatenate((-HET,-HTT),axis=0)


    #Prepare the RHS based on given boundary conditions
    #interchange column between A,B based on boundary conditions, edge are always no-flow boundary conditions




    return HEE,HTE,HET,HTT,GET,GTT,MatA,MatB,Mat_C,MatHe
    #return Mat_C,MatHe,MatA,MatB


def solution_allocate_DFN(panels,traces,Xp,Pt,Qt):
    NE=len(panels) #BE number
    NT=len(traces) #trace number
    Nbd_pt=len(traces[0]) #number of element per trace
    N=2*NE #Node number
    #Xp=-Xp[:N]
    
    for i in range(NE):
        panels[i].P1=Xp[2*i] #odd
        panels[i].P2=Xp[2*i+1] #even
        panels[i].bd_value1=Xp[2*i] #odd
        panels[i].bd_value2=Xp[2*i+1] #even
        
    for i in range(NT):
        for j in range(Nbd_pt):#4 elements per trace
            traces[i][j].bd_value=Qt[Nbd_pt*i+j]
            traces[i][j].Q=Qt[Nbd_pt*i+j]
            traces[i][j].P=Pt[Nbd_pt*i+j]


    
    print("Number of boundary elements:%s" % (len(panels)))
    print("Coordinates&Boundary conditions of boundary elements:")
    print("Point\tX\t\tY\t\tPressure\tLeft Flux\tRight flux")
    for i in range(NE):
        pl=panels[i]
        if i==1:
            pl_p=panels[NE-1]
        else:
            pl_p=panels[i-1]
        print("(%s)%s\t%5.3f\t\t%.3f\t\t%.3f\t\t%.4f\t\t%.4f" % (i+1,2*i+1,pl.xa,pl.ya,pl.P1,pl_p.Q3,pl.Q1))
        print("(%s)%s\t%5.3f\t\t%.3f\t\t%.3f\t\t%.4f\t\t%.4f" % (i+1,2*i+2,pl.xc,pl.yc,pl.P2,pl.Q2,pl.Q2))
    
    print("Number of boundary elements:%s" % (len(panels)))
    print("Coordinates&Boundary conditions of boundary elements:")
    print("Point\tX\t\tY\t\tQ\t\tP")
    for i in range(NT):
        for j, pl in enumerate(traces[i]):
            print("%s\t%s\t\t%s\t\t%s\t\t%s " % (j+1,pl.xc,pl.yc,pl.Q,pl.P))
        print('\n')


def Field_Solve_DFN(xi,yi,panels,traces):
    PI=3.141592653
    NE=len(panels) #BE number
    NT=len(traces) #trace number
    Nbd_pt=len(traces[0]) #number of element per trace
    p,u,v=0,0,0
    
    
    #HpE- Hmatrix edge-point(xi,yi)
    for i in range(NE):                   
        pl=panels[i]                 #  0   1   2   3   4   5
        A=GHCalc_quadratic_adapative(xi,yi,pl,'internal') # Hij,Gij,DUx,DQx,DUy,DQy
        #A=GHCalc_quadratic(xi,yi,pl)
        HpE,Gij=A[0],A[1]
        DUx,DQx=A[2],A[3]
        DUy,DQy=A[4],A[5]
        
        #Q=numpy.zeros(3+1, dtype=float)
        P=numpy.zeros(3+1,dtype=float)
        if i==NE-1:
            pl_next=panels[0]
        else:
            pl_next=panels[i+1]
            
        #Q[1],Q[2],Q[3]=pl.Q1,pl.Q2,pl.Q3 edge Q=0
        P[1],P[2],P[3]=pl.P1,pl.P2,pl_next.P1
        for j in range(1,3+1):
            p=p-HpE[j]*P[j]
            u=u-DQx[j]*P[j]
            v=v-DQy[j]*P[j]
    
    #GpT- Gmatrix trace-point(xi,yi)
    for i in range(NT):
        for j in range(Nbd_pt): #4 elements per trace
            pl=traces[i][j]
            
            TEMP=GHCalc_analytical(xi,yi,pl)
            GpT=TEMP[1]/pl.length
            #DU1,DU2=B[2],B[4]
            DU1,DU2=TEMP[2]/pl.length,TEMP[4]/pl.length
            DQ1,DQ2=TEMP[3]/pl.length,TEMP[5]/pl.length
            
            Q=pl.Q
            P=pl.P
            
            p=p+Q*GpT
            u=u+Q*DU1
            v=v+Q*DU2      
    
    p=p/2/PI
    u=-u/2/PI
    v=-v/2/PI
    return p,u,v
