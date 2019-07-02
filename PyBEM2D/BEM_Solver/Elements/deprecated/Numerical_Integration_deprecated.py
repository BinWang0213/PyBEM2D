
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


def GHCalc_linear(xi,yi,panelj):
    #Gauss Quadrature intergration,n=4
    GI=[0.8611363116,0.3399810436,-0.3399810436,-0.8611363116]  #Gauss point
    WC=[0.3478548451,0.6521451549,0.6521451549,0.3478548451]    #weight cofficient
    
        #Integration interval transoform
    Xj_G=np.empty(4) #Gauss point transfer
    Yj_G=np.empty(4) #Gauss point transfer
    Rij=0  #Distance from node to Gauss point on boundary element
    Rd1=0  #Radius derivatives 
    Rd2=0  #Raidus derivatives
    Rd=0   #Rd=nx*Rd1+ny*Rd2
    #Boundary solution variable
    phi1=0  #shape function1 (1-phi)/2
    phi2=0  #shape function2 (1+phi)/2
    Hij1=0  #H1 matrix value-velocity shape function H1
    Hij2=0  #H2 matrix value-velocity shape function H2
    Gij1=0  #G1 matrix value-pressure shape function G1
    Gij2=0  #G2 matrix value-pressure shape function G2
    #Internal solution variable
    DUx1=0  #G derivative in x direction shape function1
    DUx2=0  #shape function2
    DUy1=0  #G derivative in y direction shape function2
    DUy2=0  #shape function2
    DQx1=0  #H derivative in x direction shape function1
    DQx2=0  #shape function2
    DQy1=0  #H derivative in y direction shape function1
    DQy2=0  #shape function2
    for j in range(4):
        Xj_G[j]=(panelj.xb-panelj.xa)*GI[j]/2+(panelj.xb+panelj.xa)/2
        Yj_G[j]=(panelj.yb-panelj.ya)*GI[j]/2+(panelj.yb+panelj.ya)/2
        Rij=np.sqrt((xi-Xj_G[j])**2+(yi-Yj_G[j])**2)
        Rd1=(Xj_G[j]-xi)/Rij
        Rd2=(Yj_G[j]-yi)/Rij
        Rd=Rd1*panelj.nx+Rd2*panelj.ny
        phi1=(1-GI[j])/2
        phi2=(1+GI[j])/2

        Hij1=Hij1-phi1*Rd/Rij*WC[j]*panelj.length/2
        Hij2=Hij2-phi2*Rd/Rij*WC[j]*panelj.length/2
        Gij1=Gij1+phi1*np.log(1/Rij)*WC[j]*panelj.length/2
        Gij2=Gij2+phi2*np.log(1/Rij)*WC[j]*panelj.length/2

        DUx1=DUx1+phi1*Rd1/Rij*WC[j]*panelj.length/2
        DUx2=DUx2+phi2*Rd1/Rij*WC[j]*panelj.length/2
        DUy1=DUy1+phi1*Rd2/Rij*WC[j]*panelj.length/2
        DUy2=DUy2+phi2*Rd2/Rij*WC[j]*panelj.length/2
        DQx1=DQx1-phi1*(2*Rd1*Rd2*panelj.ny+(2*Rd1**2-1)*panelj.nx)/(Rij**2)*WC[j]*panelj.length/2
        DQx2=DQx2-phi2*(2*Rd1*Rd2*panelj.ny+(2*Rd1**2-1)*panelj.nx)/(Rij**2)*WC[j]*panelj.length/2
        DQy1=DQy1-phi1*(2*Rd1*Rd2*panelj.nx+(2*Rd2**2-1)*panelj.ny)/(Rij**2)*WC[j]*panelj.length/2
        DQy2=DQy2-phi2*(2*Rd1*Rd2*panelj.nx+(2*Rd2**2-1)*panelj.ny)/(Rij**2)*WC[j]*panelj.length/2

    return Hij1,Hij2,Gij1,Gij2,DUx1,DQx1,DUx2,DQx2,DUy1,DQy1,DUy2,DQy2


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