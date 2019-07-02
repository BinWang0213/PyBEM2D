# Exact Integration for Const,Linear and Quad element
from math import *

def Analytical_Intergration_python(xi, yi, panel, k_tensor):

    G, H, Gx, Gy, Hx, Hy = [-1] * 3, [-1] * \
        3, [-1] * 3, [-1] * 3, [-1] * 3, [-1] * 3

    #print('Pts',xi,yi)
    #print("Panel",panel)
    if(panel.isPtsOnElement(xi, yi)):
        OnElement_Intergration(xi, yi, panel, G, H, Gx, Gy,Hx, Hy, k_tensor[0], k_tensor[1], k_tensor[2])
    else:
        OffElement_Intergration(xi, yi, panel, G, H, Gx, Gy, Hx, Hy, k_tensor[0], k_tensor[1], k_tensor[2])

    return G, H, Gx, Gy, Hx, Hy
    #print(panel.element_type, G, H, Gx, Gy, Hx, Hy)



def OffElement_Intergration( xi,  yi, panel,  G, H, Gx, Gy, Hx,Hy,
                                   k11,  k12,  k22):

    element_type = panel.element_type
    #Basic element info
    x1, x2, y1, y2 = panel.xa, panel.xb, panel.ya, panel.yb
    Lj = panel.length
    J = Lj / 2.0  # Constant Jacobian for stright line
    d = panel.d

    #Geometric constants
    Cx = (x2 + x1) / 2.0 - xi
    Cy = (y2 + y1) / 2.0 - yi
    Dx = (x2 - x1) / 2.0
    Dy = (y2 - y1) / 2.0

    #Anistropic properties
    k_det = k11 * k22 - k12 * k12  # k11*k22-k12^2
    s11 = k22 / k_det
    s22 = k11 / k_det
    s12 = -k12 / k_det

    #r=a*zi**2+b*zi+c
    a = s11 * Dx * Dx + 2 * s12 * Dx * Dy + s22 * Dy * Dy
    b = 2.0 * (s11 * Cx * Dx + s12 * (Cx * Dy + Cy * Dx) + s22 * Cy * Dy)
    c = s11 * Cx * Cx + 2 * s12 * Cx * Cy + s22 * Cy * Cy
    e = Cx * Dy - Cy * Dx
    dis = 4.0 * a * c - b * b
    
    if(abs(dis) < 1e-15):
        dis = 0.0  # For Computer, 0 is not really zero

    #F integrals
    F_0 = 0.0
    if(dis > 0):
        F_0 = 2.0 / sqrt(dis) * (atan((2.0 * a + b) / sqrt(dis)
                                      ) - atan((-2.0 * a + b) / sqrt(dis)))
    if(abs(dis) == 0.0):
        F_0 = 2.0 / (b - 2.0 * a) - 2.0 / (b + 2.0 * a)

    F_1 = 1.0 / 2.0 / a * log((a + b + c) / (a - b + c)) - b / 2.0 / a * F_0
    F_2 = 2.0 / a - c / a * F_0 - b / a * F_1
    F_3 = -c / a * F_1 - b / a * F_2
    F_4 = 2.0 / 3.0 / a - c / a * F_2 - b / a * F_3
    F_5 = -c / a * F_3 - b / a * F_4

    #G integrals
    if(dis != 0.0):
        G_0 = (2.0 * a + b) / (dis) / (a + b + c) - (-2.0 * a + b) / \
            (dis) / (a - b + c) + 2.0 * a / (dis) * F_0
        G_1 = -(b + 2.0 * c) / (dis) / (a + b + c) + \
            (-b + 2.0 * c) / (dis) / (a - b + c) - b / (dis) * F_0

    if(dis == 0.0):
        G_0 = 8.0 * a / 3.0 / pow((b - 2.0 * a), 3.0) - 8.0 * a / 3.0 / pow((b + 2.0 * a), 3)
        G_1 = -(8.0 * a / 3 / pow((2.0 * a + b), 3) + 8.0 * a / 3 / pow((-2.0 * a + b), 3)
                ) - 2.0 / 3.0 * (1.0 / pow((2.0 * a + b), 2) - 1.0 / pow((-2.0 * a + b), 2))
    G_2 = -1.0 / (a * (a - b + c)) - 1.0 / (a * (a + b + c)) + c / a * G_0
    G_3 = 1.0 / 2.0 / a / a * (log((a + b + c) / (a - b + c)) -
                               3.0 * a * b * G_2 - (2.0 * a * c + b * b) * G_1 - b * c * G_0)

    #1. Constant Element
    # Common used integrals
    A0 = log((a + c) * (a + c) - b * b) - 2 * a * F_2 - b * F_1

    Ex0, Ey0 = (Cx * F_0 + Dx * F_1), (Cy * F_0 + Dy * F_1)
    Ix0, Iy0 = 2.0 * e * (Cx * G_0 + Dx * G_1), 2.0 * e * (Cy * G_0 + Dy * G_1)
    Esx0, Esy0 = Ex0 * s11 + Ey0 * s12, Ex0 * s12 + Ey0 * s22
    Isx0, Isy0 = Ix0 * s11 + Iy0 * s12 - Dy * F_0, Ix0 * s12 + Iy0 * s22 + Dx * F_0

    #debug_numericalQuad_Anistropic('hello', k11, k12, k22, s11, s12, s22, Lj, a, b, c,
    #                               d, e, dis, J, Cx, Cy, Dx, Dy, F_0, F_1, F_2, F_3, F_4, F_5, G_0, G_1, G_2, G_3)

    if(element_type == "Const"):
        G[0] = -J / 2.0 * A0
        H[0] = -e * F_0
        Gx[0] = J * Esx0
        Gy[0] = J * Esy0
        Hx[0] = -Isx0
        Hy[0] = -Isy0
        return
        #return G1,H1,Gx1,Gy1,Hx1,Hy1

    #2. Discontinuous Linear Element

    #discontinous element local offset distance (1.0/2,1)

    # Common used integrals
    A1 = 1.0 / 2.0 * (log((a + b + c) / (a - b + c)) - 2.0 * a * F_3 - b * F_2)
    Ex1, Ey1 = (Cx * F_1 + Dx * F_2), (Cy * F_1 + Dy * F_2)
    Ix1, Iy1 = (2.0 * e * (Cx * G_1 + Dx * G_2)), (2.0 * e * (Cy * G_1 + Dy * G_2))
    Esx1, Esy1 = Ex1 * s11 + Ey1 * s12, Ex1 * s12 + Ey1 * s22
    Isx1, Isy1 = Ix1 * s11 + Iy1 * s12 - Dy * F_1, Ix1 * s12 + Iy1 * s22 + Dx * F_1

    if(element_type == "Linear"):
        G[0], G[1] = -J / 2.0 / 2.0 * \
            (A0 - A1 / d), -J / 2.0 / 2.0 * (A0 + A1 / d)
        H[0], H[1] = -e / 2.0 * (F_0 - F_1 / d), -e / 2.0 * (F_0 + F_1 / d)
        Gx[0], Gx[1] = J / 2.0 * (Esx0 - Esx1 / d), J / 2.0 * (Esx0 + Esx1 / d)
        Gy[0], Gy[1] = J / 2.0 * (Esy0 - Esy1 / d), J / 2.0 * (Esy0 + Esy1 / d)
        Hx[0], Hx[1] = -1.0 / 2.0 * \
            (Isx0 - Isx1 / d), -1.0 / 2.0 * (Isx0 + Isx1 / d)
        Hy[0], Hy[1] = -1.0 / 2.0 * \
            (Isy0 - Isy1 / d), -1.0 / 2.0 * (Isy0 + Isy1 / d)
        return
        #return G1, G2, Gx1,Gx2, Gy1, Gy2, Hx1, Hx2, Hy1,Hy2

    #3. Discontinuous Quadratic Element

    # Common used integrals
    A2 = 1.0 / 3.0 * (log((a + c) * (a + c) - b * b) - 2 * a * F_4 - b * F_3)
    Ex2, Ey2 = (Cx * F_2 + Dx * F_3), (Cy * F_2 + Dy * F_3)
    Ix2, Iy2 = (2.0 * e * (Cx * G_2 + Dx * G_3)), (2.0 * e * (Cy * G_2 + Dy * G_3))
    Esx2, Esy2 = Ex2 * s11 + Ey2 * s12, Ex2 * s12 + Ey2 * s22
    Isx2, Isy2 = Ix2 * s11 + Iy2 * s12 - Dy * F_2, Ix2 * s12 + Iy2 * s22 + Dx * F_2


    if(element_type == "Quad"):
        G[0], G[1], G[2] = -J / 2.0 / 2.0 * (A2 / d / d - A1 / d), -J / 2.0 * (
            A0 - A2 / d / d), -J / 2.0 / 2.0 * (A2 / d / d + A1 / d)
        H[0], H[1], H[2] = -e / 2.0 * (F_2 / d / d - F_1 / d), -e * (
            F_0 - F_2 / d / d), -e / 2.0 * (F_2 / d / d + F_1 / d)
        Gx[0], Gx[1], Gx[2] = J / 2.0 * (Esx2 / d / d - Esx1 / d), J * (
            Esx0 - Esx2 / d / d), J / 2.0 * (Esx2 / d / d + Esx1 / d)
        Gy[0], Gy[1], Gy[2] = J / 2.0 * (Esy2 / d / d - Esy1 / d), J * (
            Esy0 - Esy2 / d / d), J / 2.0 * (Esy2 / d / d + Esy1 / d)
        Hx[0], Hx[1], Hx[2] = -1.0 / 2.0 * (Isx2 / d / d - Isx1 / d), -(
            Isx0 - Isx2 / d / d), -1.0 / 2.0 * (Isx2 / d / d + Isx1 / d)
        Hy[0], Hy[1], Hy[2] = -1.0 / 2.0 * (Isy2 / d / d - Isy1 / d), -(
            Isy0 - Isy2 / d / d), -1.0 / 2.0 * (Isy2 / d / d + Isy1 / d)
        return
        #return G1, G2, G3, H1, H2, H3, Gx1, Gx2, Gx3, Gy1, Gy2, Gy3, Hx1, Hx2, Hx3, Hy1, Hy2, Hy3

    #print(G1,G2,G3,H1,H2,H3,Gx1,Gx2,Gx3,Gy1,Gy2,Gy3,Hx1,Hx2,Hx3,Hy1,Hy2,Hy3)


def OnElement_Intergration(xi,yi, panel,G,H,Gx,Gy,Hx,Hy,k11,  k12,  k22):
    
    
    element_type = panel.element_type

    #Basic geometric info
    x1, x2, y1, y2 = panel.xa, panel.xb, panel.ya, panel.yb
    Lj = panel.length
    J = Lj / 2.0  # Constant Jacobian for stright line
    d = panel.d

    #Geometric constants
    Dx = (x2 - x1) / 2.0
    Dy = (y2 - y1) / 2.0

    a = Dx * Dx + Dy * Dy

    #Anistropic properties
    k_det = k11 * k22 - k12 * k12  # k11*k22-k12^2
    s11 = k22 / k_det
    s22 = k11 / k_det
    s12 = -k12 / k_det

    a = s11 * Dx * Dx + 2 * s12 * Dx * Dy + s22 * Dy * Dy

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

def debug_numericalQuad_Anistropic(mode,k11,k12,k22,s11,s12,s22,Lj,a,b,c,d,e,dis,J,Cx,Cy,Dx,Dy,F_0,F_1,F_2,F_3,F_4,F_5,G_0,G_1,G_2,G_3):
    from scipy.integrate import quad

    def Switch_ShapeFunc(N, z, d):
            if(N == 1):
                N = 1
            if(N == 2):
                N = 0.5 * (1 - z / d)
            if(N == 3):
                N = 0.5 * (1 + z / d)
            if(N == 4):
                N = 0.5 * z / d * (z / d - 1)
            if(N == 5):
                N = (1 - z / d) * (1 + z / d)
            if(N == 6):
                N = 0.5 * z / d * (z / d + 1)
            return N
    
    def G_integral(z, N):
            kernal = -J / 2 * log(a * z**2 + b * z + c)
            return kernal * Switch_ShapeFunc(N, z, d)
    def H_integral(z,N):
        kernal = -1 * e / (a * z**2 + b * z + c)
        return kernal * Switch_ShapeFunc(N, z, d)
    
    def dudx_integral(z, N):
            kernal = J * (s11*(Cx + Dx * z)+s12*(Cy+Dy*z)) / (a * z**2 + b * z + c)
            return kernal * Switch_ShapeFunc(N, z, d)
    def dudy_integral(z, N):
            kernal = J * (s12*(Cx + Dx * z)+s22*(Cy+Dy*z)) / (a * z**2 + b * z + c)
            return kernal * Switch_ShapeFunc(N, z, d)
    def dudxx_integral(z,N):
        kernal = J * ( 2*dudx_integral(z, N)**2 +s11/(a * z**2 + b * z + c) )
        return kernal * Switch_ShapeFunc(N, z, d)
    def dudxy_integral(z,N):
        kernal = J * (2*dudx_integral(z,N)*dudy_integral(z,N)+s12/(a * z**2 + b * z + c))
        return kernal * Switch_ShapeFunc(N, z, d)
    def dudyy_integral(z,N):
        kernal = J * (2*dudy_integral(z, N)**2+s22/(a * z**2 + b * z + c))
        return kernal * Switch_ShapeFunc(N, z, d)

    G = quad(G_integral, -1, 1, args=(1), points=[-1, 0, 1], epsabs=1e-15)[0]
    print('G',G)

    H_direct=quad(H_integral, -1, 1, args=(1), points=[-1, 0, 1], epsabs=1e-15)[0]
    dudx=quad(dudx_integral, -1, 1, args=(1), points=[-1, 0, 1], epsabs=1e-15)[0]
    dudy=quad(dudy_integral, -1, 1, args=(1), points=[-1, 0, 1], epsabs=1e-15)[0]

    Gx=k11*dudx+k12*dudy
    Gy=k12*dudx+k22*dudy

    nx,ny=2*Dy/Lj,-2*Dx/Lj
    print('H',-1*(Gx*nx+Gy*ny))
    print('H_direct', H_direct)


    print('Gx', Gx)
    print('Gy', Gy)

    dudxx=quad(dudxx_integral, -1, 1, args=(1), points=[-1, 0, 1], epsabs=1e-15)
    dudxy=quad(dudxy_integral, -1, 1, args=(1), points=[-1, 0, 1], epsabs=1e-15)
    dudyy=quad(dudyy_integral, -1, 1, args=(1), points=[-1, 0, 1], epsabs=1e-15)

    print(k11,k12,k22,s11,s12,s22)
    print('error',dudxx[1],dudxy[1],dudyy[1])
    dudxx, dudxy, dudyy = dudxx[0], dudxy[0], dudyy[0]
    Hx=(k11*dudxx+k12*dudxy)*nx+(k12*dudxx+k22*dudxy)*ny
    Hy=(k11*dudxy+k12*dudyy)*nx+(k12*dudxy+k22*dudyy)*ny
    
    print('Hx',Hx)
    print('Hy',Hy)

def debug_numericalQuad_Dimensionless(mode,a,b,c,d,e,dis,J,Cx,Cy,Dx,Dy,F_0,F_1,F_2,F_3,F_4,F_5,G_0,G_1,G_2,G_3):
        print("---Debug is open---", dis)
        from scipy.integrate import quad

        debug=0

        if(debug):
            def F_func(z, n, a, b, c):
                return z**n / (a * z**2 + b * z + c)

            F_00 = quad(F_func, -1, 1, args=(0, a, b, c), points=[-1, 0, 1])
            F_10 = quad(F_func, -1, 1, args=(1, a, b, c), points=[-1, 0, 1])
            F_20 = quad(F_func, -1, 1, args=(2, a, b, c), points=[-1, 0, 1])
            F_30 = quad(F_func, -1, 1, args=(3, a, b, c), points=[-1, 0, 1])
            F_40 = quad(F_func, -1, 1, args=(4, a, b, c), points=[-1, 0, 1])
            F_50 = quad(F_func, -1, 1, args=(5, a, b, c), points=[-1, 0, 1])
            print('F')
            print(F_0, F_00)
            print(F_1, F_10)
            print(F_2, F_20)
            print(F_3, F_30)
            print(F_4, F_40)
            print(F_5, F_50)

            def G_func(z, n, a, b, c):
                if(n == 0):
                    return 1 / (a * z**2 + b * z + c)**2
                return z**n / (a * z**2 + b * z + c)**2

            G_00 = quad(G_func, -1, 1, args=(0, a, b, c), points=[-1, 0, 1])
            G_10 = quad(G_func, -1, 1, args=(1, a, b, c), points=[-1, 0, 1])
            G_20 = quad(G_func, -1, 1, args=(2, a, b, c), points=[-1, 0, 1])
            G_30 = quad(G_func, -1, 1, args=(3, a, b, c), points=[-1, 0, 1])

            print('G')
            print(G_0, G_00)
            print(G_1, G_10)
            print(G_2, G_20)
            print(G_3, G_30)

        def Switch_ShapeFunc(N,z,d):
            if(N == 1):
                N = 1
            if(N == 2):
                N = 0.5 * (1 - z / d)
            if(N == 3):
                N = 0.5 * (1 + z / d)
            if(N == 4):
                N = 0.5 * z/d * (z/d - 1)
            if(N == 5):
                N = (1 - z / d) * (1 + z / d)
            if(N == 6):
                N = 0.5 * z/d * (z/d + 1)
            return N

        def G_integral(z, N, J, a, b, c):
            
            kernal = -J / 2 * np.log(a * z**2 + b * z + c)
            return kernal * Switch_ShapeFunc(N, z, d)

        def H_integral(z, N, e, a, b, c):
            kernal= -1 * e / (a * z**2 + b * z + c)
            return kernal * Switch_ShapeFunc(N, z, d)

        def Gx_integral(z, N, J, Cx, Dx, a, b, c):
            kernal= J * (Cx + Dx * z) / (a * z**2 + b * z + c)
            return kernal * Switch_ShapeFunc(N, z, d)

        def Gy_integral(z, N, J, Cy, Dy, a, b, c):
            kernal= J * (Cy + Dy * z) / (a * z**2 + b * z + c)
            return kernal * Switch_ShapeFunc(N, z, d)

        def Hx_integral(z, N, e, Cx, Dx, Dy, a, b, c):
            kernal= -1 * (2 * e * (Cx + Dx * z) / ((a * z**2 + b * z + c)**2) - Dy / (a * z**2 + b * z + c))
            return kernal * Switch_ShapeFunc(N, z, d)

        def Hy_integral(z, N, e, Cy, Dy, Dx, a, b, c):
            kernal= -1 * (2 * e * (Cy + Dy * z) / ((a * z**2 + b * z + c)**2) + Dx / (a * z**2 + b * z + c))
            return kernal * Switch_ShapeFunc(N, z, d)

        if(mode=="Const" or mode=="All"):
            print('Const Element')
            print('G', quad(G_integral, -1, 1,args=(1, J, a, b, c), points=[-1, 0, 1],epsabs=1e-15))
            print('H', quad(H_integral, -1, 1,args=(1, e, a, b, c), points=[-1, 0, 1],epsabs=1e-15))
            print('Gx', quad(Gx_integral, -1, 1,args=(1, J, Cx, Dx, a, b, c), points=[-1, 0, 1],epsabs=1e-15))
            print('Gy', quad(Gy_integral, -1, 1,args=(1, J, Cy, Dy, a, b, c), points=[-1, 0, 1],epsabs=1e-15))
            print('Hx', quad(Hx_integral, -1, 1, args=(1, e,Cx, Dx, Dy, a, b, c), points=[-1, 0, 1],epsabs=1e-15))
            print('Hy', quad(Hy_integral, -1, 1, args=(1, e,Cy, Dy, Dx, a, b, c), points=[-1, 0, 1],epsabs=1e-15))

        if(mode == "Linear"or mode == "All"):
            print("Linear Element")
            print('G1', quad(G_integral, -1, 1,args=(2, J, a, b, c), points=[-1, 0, 1],epsabs=1e-15))
            print('G2', quad(G_integral, -1, 1,args=(3, J, a, b, c), points=[-1, 0, 1],epsabs=1e-15))
            print('H1', quad(H_integral, -1, 1,args=(2, e, a, b, c), points=[-1, 0, 1],epsabs=1e-15))
            print('H2', quad(H_integral, -1, 1,args=(3, e, a, b, c), points=[-1, 0, 1],epsabs=1e-15))
            print('Gx1', quad(Gx_integral, -1, 1,args=(2, J, Cx, Dx, a, b, c), points=[-1, 0, 1],epsabs=1e-15))
            print('Gx2', quad(Gx_integral, -1, 1,args=(3, J, Cx, Dx, a, b, c), points=[-1, 0, 1],epsabs=1e-15))
            print('Gy1', quad(Gy_integral, -1, 1,args=(2, J, Cy, Dy, a, b, c), points=[-1, 0, 1],epsabs=1e-15))
            print('Gy2', quad(Gy_integral, -1, 1,args=(3, J, Cy, Dy, a, b, c), points=[-1, 0, 1],epsabs=1e-15))
            print('Hx1', quad(Hx_integral, -1, 1, args=(2, e,Cx, Dx, Dy, a, b, c), points=[-1, 0, 1],epsabs=1e-15))
            print('Hx2', quad(Hx_integral, -1, 1, args=(3, e,Cx, Dx, Dy, a, b, c), points=[-1, 0, 1],epsabs=1e-15))
            print('Hy1', quad(Hy_integral, -1, 1, args=(2, e,Cy, Dy, Dx, a, b, c), points=[-1, 0, 1],epsabs=1e-15))
            print('Hy2', quad(Hy_integral, -1, 1, args=(3, e,Cy, Dy, Dx, a, b, c), points=[-1, 0, 1],epsabs=1e-15))

        if(mode == "Quad" or mode == "All"):
            print("Quad Element")
            print('G1', quad(G_integral, -1, 1,args=(4, J, a, b, c), points=[-1, 0, 1],epsabs=1e-15))
            print('G2', quad(G_integral, -1, 1,args=(5, J, a, b, c), points=[-1, 0, 1],epsabs=1e-15))
            print('G3', quad(G_integral, -1, 1,args=(6, J, a, b, c), points=[-1, 0, 1],epsabs=1e-15))
            print('H1', quad(H_integral, -1, 1,args=(4, e, a, b, c), points=[-1, 0, 1],epsabs=1e-15))
            print('H2', quad(H_integral, -1, 1,args=(5, e, a, b, c), points=[-1, 0, 1],epsabs=1e-15))
            print('H3', quad(H_integral, -1, 1,args=(6, e, a, b, c), points=[-1, 0, 1],epsabs=1e-15))
            print('Gx1', quad(Gx_integral, -1, 1,args=(4, J, Cx, Dx, a, b, c), points=[-1, 0, 1],epsabs=1e-15))
            print('Gx2', quad(Gx_integral, -1, 1,args=(5, J, Cx, Dx, a, b, c), points=[-1, 0, 1],epsabs=1e-15))
            print('Gx3', quad(Gx_integral, -1, 1,args=(6, J, Cx, Dx, a, b, c), points=[-1, 0, 1],epsabs=1e-15))
            print('Gy1', quad(Gy_integral, -1, 1,args=(4, J, Cy, Dy, a, b, c), points=[-1, 0, 1],epsabs=1e-15))
            print('Gy2', quad(Gy_integral, -1, 1,args=(5, J, Cy, Dy, a, b, c), points=[-1, 0, 1],epsabs=1e-15))
            print('Gy3', quad(Gy_integral, -1, 1,args=(6, J, Cy, Dy, a, b, c), points=[-1, 0, 1],epsabs=1e-15))
            print('Hx1', quad(Hx_integral, -1, 1, args=(4, e,Cx, Dx, Dy, a, b, c), points=[-1, 0, 1],epsabs=1e-15))
            print('Hx2', quad(Hx_integral, -1, 1, args=(5, e,Cx, Dx, Dy, a, b, c), points=[-1, 0, 1],epsabs=1e-15))
            print('Hx3', quad(Hx_integral, -1, 1, args=(6, e,Cx, Dx, Dy, a, b, c), points=[-1, 0, 1],epsabs=1e-15))
            print('Hy1', quad(Hy_integral, -1, 1, args=(4, e,Cy, Dy, Dx, a, b, c), points=[-1, 0, 1],epsabs=1e-15))
            print('Hy2', quad(Hy_integral, -1, 1, args=(5, e, Cy, Dy, Dx, a,b, c), points=[-1, -0.5, 0, 0.5, 1], limit=1000, epsabs=1e-15))
            print('Hy3', quad(Hy_integral, -1, 1, args=(6, e, Cy, Dy, Dx,a, b, c), points=[-1, -0.5, 0, 0.5, 1], epsabs=1e-15))

        print("---Debug is end---")
