import numpy as np

from .Constant_element import GHCalc_analytical, GHCalc
from .Linear_element import GHCalc_linear
from .Quadratic_element import GHCalc_quadratic_adapative
from Lib.Tools.Geometry import *


def Analytical_Intergration(xi,yi,panel):

    if(panel.isPtsOnElement(xi, yi)):
        OnElement_Intergration(xi,yi,panel)
    else:
        OffElement_Intergration(xi,yi,panel)



def OffElement_Intergration(xi,yi,panel):

    debug=0
    #Basic element info
    x1, x2, y1, y2 = panel.xa, panel.xb, panel.ya, panel.yb
    Lj = panel.length
    J = Lj / 2  # Constant Jacobian for stright line

    #Geometric constants
    Ex = (x2 + x1) / 2
    Ey = (y2 + y1) / 2
    Cx = Ex - xi
    Cy = Ey - yi
    Dx = (x2 - x1) / 2
    Dy = (y2 - y1) / 2
    

    #r=a*zi**2+b*zi+c
    a = Dx**2 + Dy**2
    b = 2 * (Cx * Dx + Cy * Dy)
    c = Cx**2 + Cy**2
    e = Cx*Dy-Cy*Dx
    dis = 4 * a * c - b**2

    if(abs(dis)<1e-15): dis=0.0 #For Computer, 0 is not really zero

    #F integrals
    F_0=0.0
    if(dis > 0):
        F_0 = 2 / np.sqrt(dis) * (np.arctan((2 * a + b) /np.sqrt(dis)) - np.arctan((-2 * a + b) / np.sqrt(dis)))
    if(abs(dis) ==0.0):
        F_0 = 2 / (b - 2 * a) - 2 / (b + 2 * a)

    F_1 = 1/2/a*np.log((a + b + c)/(a - b + c)) - b/2/a*F_0
    F_2 = 2/a - c/a * F_0 - b/a*F_1
    F_3 = -c/a*F_1 - b/a*F_2
    F_4 = 2/3/a - c/a*F_2 - b/a*F_3
    F_5 = -c/a * F_3 - b/a*F_4
    

    #G integrals
    if(dis!=0.0):
        G_0=(2*a+b)/(dis)/(a+b+c)-(-2*a+b)/(dis)/(a-b+c)+2*a/(dis)*F_0
        G_1=-(b+2*c)/(dis)/(a+b+c)+(-b+2*c)/(dis)/(a-b+c)-b/(dis)*F_0

    if(dis==0.0):
        G_0=8*a/3/(b-2*a)**3-8*a/3/(b+2*a)**3
        G_1=-(8*a/3/(2*a+b)**3+8*a/3/(-2*a+b)**3)-2/3*(1/(2*a+b)**2-1/(-2*a+b)**2)

    G_2=-1/(a*(a-b+c))-1/(a*(a+b+c))+c/a*G_0
    G_3=1/2/a/a*(np.log((a+b+c)/(a-b+c))-3*a*b*G_2-(2*a*c+b**2)*G_1-b*c*G_0)


    d=0.8 #discontinous offset

    #1. Constant Element

    # Common used integrals
    A0 = np.log((a + c)**2 - b**2) - 2 * a * F_2 - b * F_1

    Ex0, Ey0 = (Cx * F_0 + Dx * F_1),(Cy*F_0+Dy*F_1)
    Ix0, Iy0 = (2*(Cx*Dy-Cy*Dx)*(Cx*G_0+Dx*G_1)-Dy*F_0),(2*(Cx*Dy-Cy*Dx)*(Cy*G_0+Dy*G_1)+Dx*F_0)

    panel.element_type="Const"
    if(panel.element_type == "Const"):
        G1=-J/2*A0
        H1=-e*F_0
        Gx1=J*Ex0
        Gy1=J*Ey0
        Hx1=-Ix0
        Hy1=-Iy0
    
    print(G1,H1,Gx1,Gy1,Hx1,Hy1)

    #2. Discontinuous Linear Element
    
    #discontinous element local offset distance (1/2,1)
    
    # Common used integrals
    A1=1/2*(np.log((a+b+c)/(a-b+c))-2*a*F_3-b*F_2)
    Ex1, Ey1 = (Cx*F_1+Dx*F_2),(Cy*F_1+Dy*F_2)
    Ix1, Iy1 = (2*(Cx*Dy-Cy*Dx)*(Cx*G_1+Dx*G_2)-Dy*F_1),(2*(Cx*Dy-Cy*Dx)*(Cy*G_1+Dy*G_2)+Dx*F_1)

    panel.element_type = "Linear"
    if(panel.element_type == "Linear"):
        G1,G2=-J/2/2*(A0-A1/d),-J/2/2*(A0+A1/d)
        H1,H2=-e/2*(F_0-F_1/d),-e/2*(F_0+F_1/d)
        Gx1,Gx2=J/2*(Ex0-Ex1/d),J/2*(Ex0+Ex1/d)
        Gy1,Gy2=J/2*(Ey0-Ey1/d),J/2*(Ey0+Ey1/d)
        Hx1,Hx2=-1/2*(Ix0-Ix1/d),-1/2*(Ix0+Ix1/d)
        Hy1,Hy2=-1/2*(Iy0-Iy1/d),-1/2*(Iy0+Iy1/d)

    print(G1,G2,H1,H2,Gx1,Gx2,Gy1,Gy2,Hx1,Hx2,Hy1,Hy2)
    #3. Discontinuous Quadratic Element
    
    # Common used integrals
    A2 =1/3*( np.log((a + c)**2 - b**2) - 2 * a * F_4 - b * F_3 )
    Ex2, Ey2 = (Cx*F_2+Dx*F_3),(Cy*F_2+Dy*F_3)
    Ix2, Iy2 = (2*(Cx*Dy-Cy*Dx)*(Cx*G_2+Dx*G_3)-Dy*F_2),(2*(Cx*Dy-Cy*Dx)*(Cy*G_2+Dy*G_3)+Dx*F_2)

    panel.Type = "Quad"
    if(panel.Type == "Quad"):
        G1,G2,G3=-J/2/2*(A2/d/d-A1/d),-J/2*(A0-A2/d/d),-J/2/2*(A2/d/d+A1/d)
        H1,H2,H3=-e/2*(F_2/d/d-F_1/d),-e*(F_0-F_2/d/d),-e/2*(F_2/d/d+F_1/d)
        Gx1,Gx2,Gx3=J/2*(Ex2/d/d-Ex1/d),J*(Ex0-Ex2/d/d),J/2*(Ex2/d/d+Ex1/d)
        Gy1,Gy2,Gy3=J/2*(Ey2/d/d-Ey1/d),J*(Ey0-Ey2/d/d),J/2*(Ey2/d/d+Ey1/d)
        Hx1,Hx2,Hx3=-1/2*(Ix2/d/d-Ix1/d),-(Ix0-Ix2/d/d),-1/2*(Ix2/d/d+Ix1/d)
        Hy1,Hy2,Hy3=-1/2*(Iy2/d/d-Iy1/d),-(Iy0-Iy2/d/d),-1/2*(Iy2/d/d+Iy1/d)

    
    print(G1,G2,G3,H1,H2,H3,Gx1,Gx2,Gx3,Gy1,Gy2,Gy3,Hx1,Hx2,Hx3,Hy1,Hy2,Hy3)


def OnElement_Intergration(xi, yi, panel):

    debug=0

    #Basic element info
    x1, x2, y1, y2 = panel.xa, panel.xb, panel.ya, panel.yb
    Lj = panel.length
    J = Lj / 2  # Constant Jacobian for stright line

    #Geometric constants
    Dx = (x2 - x1) / 2
    Dy = (y2 - y1) / 2

    #Local coordinates
    zi = panel.get_LocalGeometricCoord((xi, yi))
    if(debug): print("Singular integral", zi)


    #Simple way to fix the log(0) issue when node approching two end
    if(abs(1+zi)<1e-15):
        zi=zi+1e-15
    elif(abs(1 - zi) < 1e-15):
        zi=zi-1e-15
    
    # Common integrals
    S0=(1-zi)*(np.log(Lj/2*(1-zi))-1) + (1+zi)*(np.log(Lj/2*(1+zi))-1)
    T0=np.log((1-zi)/(1+zi))

    d=0.8

    panel.element_type = "Const"
    if(panel.element_type == "Const"):
        #On element
        G1=-J*S0
        #G=Lj*(np.log(2/Lj)+1) test when zi=0
        Gx1=J*Dx/(Lj/2)/(Lj/2)*T0
        Gy1=J*Dy/(Lj/2)/(Lj/2)*T0
        H1,Hx1,Hy1=0.0,0.0,0.0

    print(G1, H1, Gx1, Gy1, Hx1, Hy1)
    if(debug):
        debug_oldcode(panel.Type, xi, yi, panel, G1, H1, Gx1, Gy1, Hx1, Hy1)

    # Common integrals \ is line changer
    #S1=-1/4*(1+zi)**2*(2*np.log(Lj/2*(1+zi))-1) \
    #    +1/4*(1-zi)**2*(2*np.log(Lj/2*(1-zi))-1) \
    #    +zi*S0
    S1=1/2*(1-zi*zi)*np.log((1-zi)/(1+zi))-zi
    T1=zi*np.log((1-zi)/(1+zi))+2

    panel.element_type = "Linear"
    if(panel.element_type == "Linear"):
        G1,G2=-J/2*(S0-S1/d),-J/2*(S0+S1/d)
        Gx1=J*Dx/(Lj/2)/(Lj/2)*0.5*(T0-T1/d)
        Gx2=J*Dx/(Lj/2)/(Lj/2)*0.5*(T0+T1/d)
        Gy1=J*Dy/(Lj/2)/(Lj/2)*0.5*(T0-T1/d)
        Gy2=J*Dy/(Lj/2)/(Lj/2)*0.5*(T0+T1/d)
        H1,H2,Hx1,Hx2,Hy1,Hy2=0.0,0.0,0.0,0.0,0.0,0.0
    
    print(G1, G2, H1, H2, Gx1, Gx2, Gy1, Gy2, Hx1, Hx2, Hy1, Hy2)

    if(debug): debug_oldcode(panel.Type, xi, yi, panel, G1, H1, Gx1, Gy1, Hx1, Hy1,
                  G2, H2, Gx2, Gy2, Hx2, Hy2)

    # Common integrals \ is line changer
    S2=1/9*(1+zi)**3*(3*np.log(Lj/2*(1+zi))-1) \
        +1/9*(1-zi)**3*(3*np.log(Lj/2*(1-zi))-1) \
        +2*zi*S1-zi*zi*S0
    T2=zi*zi*np.log((1-zi)/(1+zi))+2*zi
    
    panel.element_type = "Quad"
    if(panel.element_type == "Quad"):
        G1,G2,G3=-J/2*(S2/d/d-S1/d),-J*(S0-S2/d/d),-J/2*(S2/d/d+S1/d)
        Gx1=J*Dx/(Lj/2)/(Lj/2)*0.5*(T2/d/d-T1/d)
        Gx2=J*Dx/(Lj/2)/(Lj/2)*(T0-T2/d/d)
        Gx3=J*Dx/(Lj/2)/(Lj/2)*0.5*(T2/d/d+T1/d)
        Gy1=J*Dy/(Lj/2)/(Lj/2)*0.5*(T2/d/d-T1/d)
        Gy2=J*Dy/(Lj/2)/(Lj/2)*(T0-T2/d/d)
        Gy3=J*Dy/(Lj/2)/(Lj/2)*0.5*(T2/d/d+T1/d)
        H1,H2,H3,Hx1,Hx2,Hx3,Hy1,Hy2,Hy3=0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0

    if(debug): debug_oldcode(panel.Type, xi, yi, panel, G1, H1, Gx1, Gy1, Hx1, Hy1,
                  G2, H2, Gx2, Gy2, Hx2, Hy2, G3, H3, Gx3, Gy3, Hx3, Hy3)
    
    print(G1,G2,G3,H1,H2,H3,Gx1,Gx2,Gx3,Gy1,Gy2,Gy3,Hx1,Hx2,Hx3,Hy1,Hy2,Hy3)





##--------------Debug Functions-------------------
def debug_oldcode(mode,xi, yi, panel,G1,H1,Gx1,Gy1,Hx1,Hy1,
                  G2=0,H2=0,Gx2=0,Gy2=0,Hx2=0,Hy2=0,
                  G3=0,H3=0,Gx3=0,Gy3=0,Hx3=0,Hy3=0):
    
    print("---Solution---", mode)

    if(mode=="Const"):
        Const_Sol = GHCalc_analytical(xi, yi, panel)

        print("Constant Element")
        print("Exact\tReference")
        print('G', G1, Const_Sol[1])
        print('H', H1, Const_Sol[0])
        print('Gx', Gx1, Const_Sol[2])
        print('Gy', Gy1, Const_Sol[4])
        print('Hx', Hx1, Const_Sol[3])
        print('Hy', Hy1, Const_Sol[5])
    
    if(mode=="Linear"):
        Linear_Sol = GHCalc_linear(xi, yi, panel)

        print("Linear Element")
        print("Exact\tReference")
        print('G1', G1, Linear_Sol[2])
        print('G2', G2, Linear_Sol[3])
        print('H1', H1, Linear_Sol[0])
        print('H2', H2, Linear_Sol[1])
        print('Gx1', Gx1, Linear_Sol[4])
        print('Gx2', Gx2, Linear_Sol[6])
        print('Gy1', Gy1, Linear_Sol[8])
        print('Gy2', Gy2, Linear_Sol[10])
        print('Hx1', Hx1, Linear_Sol[5])
        print('Hx2', Hx2, Linear_Sol[7])
        print('Hy1', Hy1, Linear_Sol[9])
        print('Hy2', Hy2, Linear_Sol[11])

    
    if(mode=="Quad"):
        if(panel.isPtsOnElement(xi, yi)):
            Quad_Sol = np.zeros((6, 4))
        else:
            Quad_Sol = GHCalc_quadratic_adapative(xi, yi, panel)
        print("Quadratic Element")
        print("Exact\tReference")
        print('G1', G1, Quad_Sol[1][1])
        print('G2', G2, Quad_Sol[1][2])
        print('G3', G3, Quad_Sol[1][3])
        print('H1', H1, Quad_Sol[0][1])
        print('H2', H2, Quad_Sol[0][2])
        print('H2', H3, Quad_Sol[0][3])
        print('Gx1', Gx1, Quad_Sol[2][1])
        print('Gx2', Gx2, Quad_Sol[2][2])
        print('Gx3', Gx3, Quad_Sol[2][3])
        print('Gy1', Gy1, Quad_Sol[4][1])
        print('Gy2', Gy2, Quad_Sol[4][2])
        print('Gy3', Gy3, Quad_Sol[4][3])
        print('Hx1', Hx1, Quad_Sol[3][1])
        print('Hx2', Hx2, Quad_Sol[3][2])
        print('Hx3', Hx3, Quad_Sol[3][3])
        print('Hy1', Hy1, Quad_Sol[5][1])
        print('Hy2', Hy2, Quad_Sol[5][2])
        print('Hy3', Hy3, Quad_Sol[5][3])


def debug_numericalQuad(mode,a,b,c,d,e,dis,J,Cx,Cy,Dx,Dy,F_0,F_1,F_2,F_3,F_4,F_5,G_0,G_1,G_2,G_3):
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
