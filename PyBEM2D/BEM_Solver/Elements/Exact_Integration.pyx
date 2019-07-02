from libc.math cimport *
from libc.math cimport M_PI as pi
cimport cython

# Exact Integration for point source element
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef Analytical_Intergration_source_cython(xi,yi,source,k_tensor):
    #Note: this well source function only support anistropic perm
    cdef double G
    cdef double x0, y0, r2
    cdef double k11,k12,k22,k_det

    x0,y0=source.x0, source.y0
    k11,k12,k22=k_tensor[0],k_tensor[1], k_tensor[2]

    #Anistropic properties
    k_det = k11 * k22 - k12*k12 #k11*k22-k12^2
    s11=k22/k_det
    s22=k11/k_det
    s12=-k12/k_det

    r2=s11*(xi-x0)*(xi-x0)+2*s12*(xi-x0)*(yi-y0)+s22*(yi-y0)*(yi-y0)

    G=log(r2)/2.0
    Gx=(xi-x0)/k11/r2
    Gy=(yi-y0)/k22/r2
    
    return G,Gx,Gy
    #print(panel.element_type, G, H, Gx, Gy, Hx, Hy)


# Exact Integration for Const,Linear and Quad element

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef Analytical_Intergration_cython(xi,yi,panel,k_tensor):
    cdef double G[3],H[3],Gx[3],Gy[3],Hx[3],Hy[3]

    for i in range(3):
        G[i],H[i],Gx[i],Gy[i],Hx[i],Hy[i]=-1.0,-1.0,-1.0,-1.0,-1.0,-1.0
    
    if(panel.isPtsOnElement(xi, yi)):
        OnElement_Intergration(xi,yi,panel,G,H,Gx,Gy,Hx,Hy,k_tensor[0],k_tensor[1],k_tensor[2])
    else:
        OffElement_Intergration(xi,yi,panel,G,H,Gx,Gy,Hx,Hy,k_tensor[0],k_tensor[1],k_tensor[2])
    
    return G,H,Gx,Gy,Hx,Hy
    #print(panel.element_type, G, H, Gx, Gy, Hx, Hy)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void OffElement_Intergration(double xi, double yi, object panel, double * G, double * H, double * Gx, double * Gy, double * Hx, double * Hy,
                                 double k11,double k12,double k22):

    #Variables initlization
    cdef double x1, x2, y1, y2
    cdef double Lj,J
    cdef double Cx,Cy,Dx,Dy
    cdef double k_det,s11,s12,s22
    cdef double a,b,c,e,dis

    cdef double F_0,F_1,F_2,F_3,F_4,F_5
    cdef double G_0,G_1,G_2,G_3
    cdef double d
    cdef double A0,A1,A2
    cdef double Ex0,Ex1,Ex2,Ey0,Ey1,Ey2
    cdef double Ix0,Ix1,Ix2,Iy0,Iy1,Iy2
    cdef double Esx0,Esx1,Esx2,Esy0,Esy1,Esy2
    cdef double Isx0,Isx1,Isx2,Isy0,Isy1,Isy2

    cdef str element_type = panel.element_type
    #Basic element info
    x1, x2, y1, y2 = panel.xa, panel.xb, panel.ya, panel.yb
    Lj = panel.length
    J = Lj / 2.0  # Constant Jacobian for stright line
    d=panel.d

    #Geometric constants
    Cx = (x2 + x1) / 2.0 - xi
    Cy = (y2 + y1) / 2.0 - yi
    Dx = (x2 - x1) / 2.0
    Dy = (y2 - y1) / 2.0

    #Anistropic properties
    k_det = k11 * k22 - k12*k12 #k11*k22-k12^2
    s11=k22/k_det
    s22=k11/k_det
    s12=-k12/k_det

    #r=a*zi**2+b*zi+c
    a = s11*Dx*Dx + 2*s12*Dx*Dy+ s22*Dy*Dy
    b = 2.0 * (s11*Cx * Dx + s12*(Cx*Dy+Cy*Dx)+s22*Cy * Dy)
    c = s11*Cx*Cx + 2*s12*Cx*Cy + s22*Cy*Cy
    e = Cx*Dy-Cy*Dx
    dis = 4.0 * a * c - b*b

    if(abs(dis)<1e-15): dis=0.0 #For Computer, 0 is not really zero

    #F integrals
    F_0=0.0
    if(dis > 0):
        F_0 = 2.0 / sqrt(dis) * (atan((2.0 * a + b) /sqrt(dis)) - atan((-2.0 * a + b) / sqrt(dis)))
    if(abs(dis) ==0.0):
        F_0 = 2.0 / (b - 2.0 * a) - 2.0 / (b + 2.0 * a)

    F_1 = 1.0/2.0/a*log((a + b + c)/(a - b + c)) - b/2.0/a*F_0
    F_2 = 2.0/a - c/a * F_0 - b/a*F_1
    F_3 = -c/a*F_1 - b/a*F_2
    F_4 = 2.0/3.0/a - c/a*F_2 - b/a*F_3
    F_5 = -c/a * F_3 - b/a*F_4
    
    #G integrals
    if(dis!=0.0):
        G_0=(2.0*a+b)/(dis)/(a+b+c)-(-2.0*a+b)/(dis)/(a-b+c)+2.0*a/(dis)*F_0
        G_1=-(b+2.0*c)/(dis)/(a+b+c)+(-b+2.0*c)/(dis)/(a-b+c)-b/(dis)*F_0

    if(dis==0.0):
        G_0=8.0*a/3.0/pow((b-2.0*a),3.0)-8.0*a/3.0/pow((b+2.0*a),3)
        G_1=-(8.0*a/3/pow((2.0*a+b),3)+8.0*a/3/pow((-2.0*a+b),3))-2.0/3.0*(1.0/pow((2.0*a+b),2)-1.0/pow((-2.0*a+b),2))

    G_2=-1.0/(a*(a-b+c))-1.0/(a*(a+b+c))+c/a*G_0
    G_3=1.0/2.0/a/a*(log((a+b+c)/(a-b+c))-3.0*a*b*G_2-(2.0*a*c+b*b)*G_1-b*c*G_0)


    #1. Constant Element

    # Common used integrals
    A0 = log((a + c)*(a+c) - b*b) - 2 * a * F_2 - b * F_1

    Ex0, Ey0 = (Cx * F_0 + Dx * F_1), (Cy * F_0 + Dy * F_1)
    Ix0, Iy0 = 2.0 * e * (Cx * G_0 + Dx * G_1), 2.0 * e * (Cy * G_0 + Dy * G_1)
    Esx0, Esy0 = Ex0 * s11 + Ey0 * s12, Ex0 * s12 + Ey0 * s22
    Isx0, Isy0 = Ix0 * s11 + Iy0 * s12 - Dy * F_0, Ix0 * s12 + Iy0 * s22 + Dx * F_0

    if(element_type == "Const"):
        G[0]=-J/2.0*A0
        H[0]=-e*F_0
        Gx[0]=J*Esx0
        Gy[0]=J*Esy0
        Hx[0]=-Isx0
        Hy[0]=-Isy0
        return
        #return G1,H1,Gx1,Gy1,Hx1,Hy1
    
    #2. Discontinuous Linear Element
    
    #discontinous element local offset distance (1.0/2,1)
    
    # Common used integrals
    A1=1.0/2.0*(log((a+b+c)/(a-b+c))-2.0*a*F_3-b*F_2)
    Ex1, Ey1 = (Cx * F_1 + Dx * F_2), (Cy * F_1 + Dy * F_2)
    Ix1, Iy1 = (2.0 * e * (Cx * G_1 + Dx * G_2)), (2.0 * e * (Cy * G_1 + Dy * G_2))
    Esx1, Esy1 = Ex1 * s11 + Ey1 * s12, Ex1 * s12 + Ey1 * s22
    Isx1, Isy1 = Ix1 * s11 + Iy1 * s12 - Dy * F_1, Ix1 * s12 + Iy1 * s22 + Dx * F_1


    if(element_type == "Linear"):
        G[0],G[1]=-J/2.0/2.0*(A0-A1/d),-J/2.0/2.0*(A0+A1/d)
        H[0],H[1]=-e/2.0*(F_0-F_1/d),-e/2.0*(F_0+F_1/d)
        Gx[0],Gx[1]=J/2.0*(Esx0-Esx1/d),J/2.0*(Esx0+Esx1/d)
        Gy[0],Gy[1]=J/2.0*(Esy0-Esy1/d),J/2.0*(Esy0+Esy1/d)
        Hx[0],Hx[1]=-1.0/2.0*(Isx0-Isx1/d),-1.0/2.0*(Isx0+Isx1/d)
        Hy[0],Hy[1]=-1.0/2.0*(Isy0-Isy1/d),-1.0/2.0*(Isy0+Isy1/d)
        return
        #return G1, G2, Gx1,Gx2, Gy1, Gy2, Hx1, Hx2, Hy1,Hy2

    #3. Discontinuous Quadratic Element
    
    # Common used integrals
    A2 =1.0/3.0*( log((a + c)*(a+c) - b*b) - 2 * a * F_4 - b * F_3 )
    Ex2, Ey2 = (Cx * F_2 + Dx * F_3), (Cy * F_2 + Dy * F_3)
    Ix2, Iy2 = (2.0 * e * (Cx * G_2 + Dx * G_3)), (2.0 * e * (Cy * G_2 + Dy * G_3))
    Esx2, Esy2 = Ex2 * s11 + Ey2 * s12, Ex2 * s12 + Ey2 * s22
    Isx2, Isy2 = Ix2 * s11 + Iy2 * s12 - Dy * F_2, Ix2 * s12 + Iy2 * s22 + Dx * F_2

    if(element_type == "Quad"):
        G[0],G[1],G[2]=-J/2.0/2.0*(A2/d/d-A1/d),-J/2.0*(A0-A2/d/d),-J/2.0/2.0*(A2/d/d+A1/d)
        H[0],H[1],H[2]=-e/2.0*(F_2/d/d-F_1/d),-e*(F_0-F_2/d/d),-e/2.0*(F_2/d/d+F_1/d)
        Gx[0],Gx[1],Gx[2]=J/2.0*(Esx2/d/d-Esx1/d),J*(Esx0-Esx2/d/d),J/2.0*(Esx2/d/d+Esx1/d)
        Gy[0],Gy[1],Gy[2]=J/2.0*(Esy2/d/d-Esy1/d),J*(Esy0-Esy2/d/d),J/2.0*(Esy2/d/d+Esy1/d)
        Hx[0],Hx[1],Hx[2]=-1.0/2.0*(Isx2/d/d-Isx1/d),-(Isx0-Isx2/d/d),-1.0/2.0*(Isx2/d/d+Isx1/d)
        Hy[0],Hy[1],Hy[2]=-1.0/2.0*(Isy2/d/d-Isy1/d),-(Isy0-Isy2/d/d),-1.0/2.0*(Isy2/d/d+Isy1/d)
        return
        #return G1, G2, G3, H1, H2, H3, Gx1, Gx2, Gx3, Gy1, Gy2, Gy3, Hx1, Hx2, Hx3, Hy1, Hy2, Hy3

    #print(G1,G2,G3,H1,H2,H3,Gx1,Gx2,Gx3,Gy1,Gy2,Gy3,Hx1,Hx2,Hx3,Hy1,Hy2,Hy3)
    
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void OnElement_Intergration(double xi, double yi, object panel, double * G, double * H, double * Gx, double * Gy, double * Hx, double * Hy,
                                 double k11,double k12,double k22):

    cdef double x1, x2, y1, y2
    cdef double Lj,J
    cdef double k_det,s11,s12,s22
    cdef double a
    cdef double d
    cdef double Dx,Dy
    cdef str element_type = panel.element_type

    cdef double zi
    cdef double S0,S1,S2,T0,T1,T2,R0,R1,R2

    #Basic geometric info
    x1, x2, y1, y2 = panel.xa, panel.xb, panel.ya, panel.yb
    Lj = panel.length
    J = Lj / 2.0  # Constant Jacobian for stright line
    d=panel.d

    #Geometric constants
    Dx = (x2 - x1) / 2.0
    Dy = (y2 - y1) / 2.0

    a = Dx*Dx + Dy*Dy

    #Anistropic properties
    k_det = k11 * k22 - k12*k12 #k11*k22-k12^2
    s11=k22/k_det
    s22=k11/k_det
    s12=-k12/k_det

    a = s11*Dx*Dx + 2*s12*Dx*Dy+ s22*Dy*Dy

    #Local coordinates
    zi = panel.get_LocalGeometricCoord((xi, yi))

    #Simple way to fix the log(0) issue when node approching two end
    if(abs(1.0 + zi) < 1e-15):
        zi = zi + 1e-15
    elif(abs(1.0 - zi) < 1e-15):
        zi = zi - 1e-15

    # Common integrals
    R0=-1.0/(1.0+zi)-1.0/(1.0-zi)
    S0 = (1.0 - zi) * (log(sqrt(a) * (1.0 - zi)) - 1.0) + \
        (1.0 + zi) * (log(sqrt(a) * (1.0 + zi)) - 1.0)
    T0 = log((1.0 - zi) / (1.0 + zi))


    if(element_type == "Const"):
        #On element
        G[0] = -J * S0
        #G=Lj*(np.log(2/Lj)+1) test when zi=0
        Gx[0] = J * (s11*Dx+s12*Dy) / a * T0 
        Gy[0] = J *(s12*Dx+s22*Dy) / a * T0
        Hx[0] = 1.0*Dy/a*R0
        Hy[0] = -1.0*Dx/a*R0
        H[0] = 0.0
        return


    # Common integrals \ is line changer
    #S1=-1/4*(1+zi)**2*(2*np.log(Lj/2*(1+zi))-1) \
    #    +1/4*(1-zi)**2*(2*np.log(Lj/2*(1-zi))-1) \
    #    +zi*S0
    R1 = 1/(1+zi)-1/(1-zi) + T0
    S1 = 1.0 / 2.0 * (1.0 - zi * zi) * log((1.0 - zi) / (1.0 + zi)) - zi
    T1 = zi * log((1.0 - zi) / (1.0 + zi)) + 2.0

    if(element_type == "Linear"):
        G[0], G[1] = -J / 2.0 * (S0 - S1 / d), -J / 2.0 * (S0 + S1 / d)
        Gx[0] = J * (s11*Dx+s12*Dy) / a * 0.5 * (T0 - T1 / d) 
        Gx[1] = J * (s11*Dx+s12*Dy) / a * 0.5 * (T0 + T1 / d) 
        Gy[0] = J * (s12*Dx+s22*Dy) / a * 0.5 * (T0 - T1 / d) 
        Gy[1] = J * (s12*Dx+s22*Dy) / a * 0.5 * (T0 + T1 / d) 
        Hx[0] = 1*Dy/a*0.5*(R0-R1/d)
        Hx[1] = 1*Dy/a*0.5*(R0+R1/d)
        Hy[0] = -1.0*Dx/a*0.5*(R0-R1/d)
        Hy[1] = -1.0*Dx/a*0.5*(R0+R1/d)
        H[0], H[1]= 0.0, 0.0
        return

    # Common integrals \ is line changer
    R2= R0+2*T1
    S2 = 1.0 / 9.0 * (1.0 + zi)**3.0 * (3.0 * log(sqrt(a) * (1.0 + zi)) - 1) \
        + 1.0 / 9.0 * (1.0 - zi)**3.0 * (3.0 * log(sqrt(a) * (1.0 - zi)) - 1) \
        + 2.0 * zi * S1 - zi * zi * S0
    T2 = zi * zi * log((1.0 - zi) / (1.0 + zi)) + 2 * zi

    if(element_type == "Quad"):
        G[0], G[1], G[2] = -J / 2.0 * (S2 / d / d - S1 / d), -J * \
            (S0 - S2 / d / d), -J / 2.0 * (S2 / d / d + S1 / d)
        Gx[0] = J * (s11*Dx+s12*Dy) / a * 0.5 * (T2 / d / d - T1 / d)
        Gx[1] = J * (s11*Dx+s12*Dy) / a * (T0 - T2 / d / d) 
        Gx[2] = J * (s11*Dx+s12*Dy) / a * 0.5 * (T2 / d / d + T1 / d) 
        Gy[0] = J * (s12*Dx+s22*Dy) / a * 0.5 * (T2 / d / d - T1 / d) 
        Gy[1] = J * (s12*Dx+s22*Dy) / a * (T0 - T2 / d / d) 
        Gy[2] = J * (s12*Dx+s22*Dy) / a * 0.5 * (T2 / d / d + T1 / d) 
        Hx[0] = 1 * Dy / a * 0.5 * (R2 / d / d - R1 / d)
        Hx[1] = 1 * Dy / a * (R0-R2 / d / d)
        Hx[2] = 1 * Dy / a * 0.5 * (R2 / d / d + R1 / d)
        Hy[0] = -1 * Dx / a * 0.5 * (R2 / d / d - R1 / d)
        Hy[1] = -1 * Dx / a * (R0 - R2 / d / d)
        Hy[2] = -1 * Dx / a * 0.5 * (R2 / d / d + R1 / d)
        H[0], H[1], H[2]= 0.0, 0.0, 0.0
        return
