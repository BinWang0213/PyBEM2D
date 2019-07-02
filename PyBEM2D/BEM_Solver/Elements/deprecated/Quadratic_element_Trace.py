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

from Lib.Tools.Geometry import cosspace, Point2Segment, GaussLib, Global2Iso, Subdivision, point_in_panel
from .Quadratic_element import *
from .Linear_element import *
from .Constant_element import *


######################## Solver Module-Matrix assemble and field point solve ########################
def build_matrix_quadratic_Trace(panels, traces, mesh):
    #Reference A BEM solution of steady-state ﬂow problems in discrete fracture networks with minimization of core storage
    #Edge use quad element and trace use constant element

    #All variables start from 1
    debug = 0

    NE = len(panels)  # number of elements
    N = mesh.Ndof_edge  # number of nodes
    #H IS A SQUARE MATRIX (2*NE,2*NE); G IS RECTANGULAR (2*NE,3*NE)
    #Index from 1
    # single node for flux term
    G = np.zeros((N + 1, N + 1), dtype=float)
    H = np.zeros((N + 1, N + 1), dtype=float)
    PI = 3.141592653

    #prepare X,Y for Book's program,index from 1 and N+1=1
    X = np.zeros(N + 2)
    Y = np.zeros(N + 2)
    for i, pl in enumerate(panels):
        X[2 * i + 1] = pl.xa
        Y[2 * i + 1] = pl.ya
        X[2 * i + 2] = pl.xc
        Y[2 * i + 2] = pl.yc
    X[N + 1] = X[1]
    Y[N + 1] = Y[1]
    #debug
    #for i in range(1,N+1+1):
    #    print("Node%s\t(%s-%s)"%(i,X[i],Y[i]))

    #1. Compute Gij[1,2,3] Hij[1,2,3] for each node and BE
    for LL in range(1, N + 1):  # The index of node
        for i in range(1, N - 1 + 1, 2):  # The index of first node
            Node2Panel = int((i + 1) / 2) - 1
            if((LL - i) * (LL - i - 1) * (LL - i - 2) * (LL - i + N - 2) != 0):  # off-diagonal
                TEMP = GHCalc_quadratic(X[LL], Y[LL], panels[Node2Panel])
                #TEMP=GHCalc_quadratic_adapative(X[LL],Y[LL],panels[Node2Panel],'boundary') #Not accurate at all
                Hij, Gij = TEMP[0], TEMP[1]
            else:  # Diagonal for Gii
                caseNo = LL - i + 1
                if (LL == 1) and (i == N - 1):
                    caseNo = caseNo + N
                Hij = GHCalc_quadratic(X[LL], Y[LL], panels[Node2Panel])[0]
                #Hij=GHCalc_quadratic_adapative(X[LL],Y[LL],panels[Node2Panel],'boundary')[0]
                Gij = Gii_singular_quadratic(panels[Node2Panel], caseNo)
            for j in range(1, 3 + 1):#Local node id
                #k = int(3 * (i - 1) / 2)
                if(debug):
                    print(Node2Panel, 'G', LL, k + j, j, Gij[j])
                #G[LL, k + j] = G[LL, k + j] + Gij[j]
                if (i - N + 1 == 0):#The last element
                    if (j == 3):#The last node
                        if(debug):
                            print(Node2Panel, 'H', LL, 1, j, Hij[j])
                        H[LL, 1] = H[LL, 1] + Hij[j]
                        G[LL, 1] = G[LL, 1] + Gij[j]
                    else:
                        if(debug):
                            print(Node2Panel, 'H', LL, i - 1 + j, j, Hij[j])
                        H[LL, i - 1 + j] = H[LL, i - 1 + j] + Hij[j]
                        G[LL, i - 1 + j] = G[LL, i - 1 + j] + Gij[j]
                else:
                    if(debug):
                        print(Node2Panel, 'H', LL, i - 1 + j, j, Hij[j])
                    H[LL, i - 1 + j] = H[LL, i - 1 + j] + Hij[j]
                    G[LL, i - 1 + j] = G[LL, i - 1 + j] + Gij[j]

            if(debug):
                print('----')

    #Compute the diagonal term
    for i in range(1, N + 1):
        H[i, i] = 0
        for j in range(1, N + 1):
            if(i != j):
                H[i, i] = H[i, i] - H[i, j]
        #For external problems:
        if (H[i, i] < 0):
            H[i, i] = 2 * PI + H[i, i]

    HEE = np.delete(H, 0, axis=1)
    HEE = np.delete(HEE, 0, axis=0)
    GEE = np.delete(G, 0, axis=1)
    GEE = np.delete(GEE, 0, axis=0)

    #2. ATE-HTE BTE-GTE -Influence factor from edge to trace
    Ne = mesh.Ndof_edge
    Nt = mesh.Ndof_trace
    HTE = np.zeros((Nt + 1, Ne + 1), dtype=float)
    GTE = np.zeros((Nt + 1, Ne + 1), dtype=float)

    Hij_temp = np.zeros((Ne, 3), dtype=float) #Temporay variable which store the Hij
    Gij_temp = np.zeros((Ne, 3), dtype=float) #Temporay variable which store the Gij

    tii = 1
    for t in range(len(traces)):  # which trace
        for ti in range(len(traces[t])):  # which node on a trace

            for j in range(NE):  # store Hij into element
                Temp = GHCalc_quadratic_adapative(
                    traces[t][ti].xc, traces[t][ti].yc, panels[j], 'internal')
                Temp_Hij=Temp[0]
                Temp_Gij=Temp[1]
                Hij_temp[j][0] = Temp_Hij[1]
                Hij_temp[j][1] = Temp_Hij[2]
                Hij_temp[j][2] = Temp_Hij[3]
                Gij_temp[j][0] = Temp_Gij[1]
                Gij_temp[j][1] = Temp_Gij[2]
                Gij_temp[j][2] = Temp_Gij[3]

            for j in range(NE):  # assemble matrix
                pl_currentH1 = Hij_temp[j][0]
                pl_currentH2 = Hij_temp[j][1]
                pl_currentG1 = Hij_temp[j][0]
                pl_currentG2 = Hij_temp[j][1]
                if j == 0:
                    pl_previousH3 = Hij_temp[NE - 1][2]
                    pl_previousG3 = Gij_temp[NE - 1][2]
                else:
                    pl_previousH3 = Hij_temp[j - 1][2]
                    pl_previousG3 = Gij_temp[j - 1][2]
                HTE[tii, 2 * j + 1] = pl_currentH1 + pl_previousH3
                HTE[tii, 2 * j + 2] = pl_currentH2
                GTE[tii, 2 * j + 1] = pl_currentG1 + pl_previousG3
                GTE[tii, 2 * j + 2] = pl_currentG2
            tii = tii + 1  # node index for trace
    HTE=np.delete(HTE,0,axis=1)
    HTE=np.delete(HTE,0,axis=0)
    GTE=np.delete(GTE,0,axis=1)
    GTE=np.delete(GTE,0,axis=0) 

    #3. AET-HET if the trace is not on the boundary-all is zero
    HET = np.zeros((Ne, Nt), dtype=float)

    #4. HTT-if the trace is not on the boundary-all is zero
    HTT = np.zeros((Nt, Nt), dtype=float)
    #compute the diagonal term
    for i in range(Nt):
        HTT[i, i] = 0
        for j in range(Ne):
            HTT[i, i] = HTT[i, i] - HTE[i, j]
        for j in range(Nt):
            if(i != j):
                HTT[i, i] = HTT[i, i] - HTT[i, j]

    #5. GET-BET
    GET = np.zeros((Ne + 1, Nt + 1), dtype=float)
    for i in range(NE):
        tii = 1
        for t in range(len(traces)):  # which trace
            for ti in range(len(traces[t])):  # which node on a trace
                #Temp_Gij_odd=GHCalc(panels[i].xa,panels[i].ya,traces[t][ti])[1]/traces[t][ti].length
                #Temp_Gij_even=GHCalc(panels[i].xc,panels[i].yc,traces[t][ti])[1]/traces[t][ti].length
                Temp_Gij_odd = GHCalc_analytical(panels[i].xa, panels[i].ya, traces[t][ti])[
                    1] / traces[t][ti].length
                Temp_Gij_even = GHCalc_analytical(panels[i].xc, panels[i].yc, traces[t][ti])[
                    1] / traces[t][ti].length
                GET[2 * i + 1, tii] = Temp_Gij_odd
                GET[2 * i + 2, tii] = Temp_Gij_even
                tii = tii + 1
    GET = np.delete(GET, 0, axis=1)
    GET = np.delete(GET, 0, axis=0)

    #6. GTT-BTT
    GTT = np.zeros((Nt + 1, Nt + 1), dtype=float)
    TII = 1
    tii = 1
    for T in range(len(traces)):
        for TI in range(len(traces[T])):
            tii = 1
            for t in range(len(traces)):  # which trace
                for ti in range(len(traces[t])):  # which node on a trace
                    if tii != TII:
                        #Temp_Gij=GHCalc(traces[T][TI].xc,traces[T][TI].yc,traces[t][ti])[6]
                        Temp_Gij = GHCalc_analytical(traces[T][TI].xc, traces[T][TI].yc, traces[t][ti])[
                            1] / traces[t][ti].length
                        GTT[TII, tii] = Temp_Gij
                    #compute the diagonal term
                    if tii == TII:
                        #GTT[TII,tii]=traces[t][ti].length*(np.log(2/traces[t][ti].length)+1)/traces[t][ti].length
                        GTT[TII, tii] = (np.log(2 / traces[t][ti].length) + 1)
                    tii = tii + 1
            TII = TII + 1

    GTT = np.delete(GTT, 0, axis=1)
    GTT = np.delete(GTT, 0, axis=0)

    debug=0
    #-------------Apply boundary conditions
    #2. Reorder the matrix
    # bd_type, 1-neumann, 0-dirichlet, odd-left node, even-right node
    A = np.block([[HEE, -GET],  # {x}={p_e,q_t}'
                  [HTE, -GTT]
                  ])

    B = np.block([[GEE, -HET],   #{b}={q_e,p_t}'
              [GTE, -HTT]
              ])
    debug=0

    #Apply BC for edge elements
    for i, pl in enumerate(panels):
        if(pl.bd_Indicator==0):#Dirichelt
            for j in range(pl.ndof):
                nodeid = mesh.getNodeId(i, j, 'Edge')
                for k in range(mesh.Ndof):
                    temp = A[k, nodeid]
                    A[k, nodeid] = -B[k, nodeid]
                    B[k, nodeid] = -temp
                    if(debug): print("Ele:%s A,Col%s<->B,Col%s" % (i+1,nodeid+1,nodeid+1))
        elif(pl.bd_Indicator == 1):#Neumann or Robin
            if(debug): 
                print('Ele:%s Neumann BC doesn\'t need to interchange col'%(i+1))

    #Apply BC for trace elements
    eleid = 0
    for ti in range(mesh.Num_trace):#Trace
        if(debug): print("--Trace ", ti + 1)
        for i,pl in enumerate(traces[ti]):#Trace ele
            if(pl.bd_Indicator == 1):  # Neumann or Robin
                for j in range(pl.ndof):
                    nodeid = mesh.getNodeId(eleid, j, 'Trace')
                    for k in range(mesh.Ndof):
                        temp = A[k, nodeid]
                        A[k, nodeid] = -B[k, nodeid]
                        B[k, nodeid] = -temp
                        if(debug): print("Ele:%s A,Col%s<->B,Col%s" % (eleid+1+mesh.Ne_edge,nodeid+1,nodeid+1))
            elif(pl.bd_Indicator==0):#Dirichelt
                if(debug):
                    print('Ele:%s Dirichlet BC doesn\'t need to interchange col'%(eleid+1+mesh.Ne_edge))
            eleid = eleid + 1

    #Collecting prescribed BC values
    if(debug): print('Prescribed Value')
    b = np.zeros(mesh.Ndof, dtype=float)
    for i, pl in enumerate(panels):
        bdvals = pl.get_bdvals()
        for j in range(pl.ndof):
            nodeid = mesh.getNodeId(i, j, 'Edge')
            b[nodeid]=bdvals[j]
            if(debug): print("Ele:%s Node:%s BC_Val:%s" % (i+1,nodeid+1,b[nodeid]))
    
    eleid = 0
    for ti in range(mesh.Num_trace):  # Trace
        for i, pl in enumerate(traces[ti]):  # Trace ele
            bdvals = pl.get_bdvals()
            for j in range(pl.ndof):
                nodeid = mesh.getNodeId(eleid, j, 'Trace')
                b[nodeid] = bdvals[j]
                if(debug): print("Ele:%s Node:%s BC_Val:%s" % (eleid+1+mesh.Ne_edge,nodeid+1,b[nodeid]))
            eleid = eleid + 1

    #Get the final RHS matrix
    b=np.dot(B,b)



    return A,b,HEE,HET,HTE,HTT,GEE,GET,GTE,GTT


def solution_allocate_quadratic_Trace(panels, traces, X, mesh, debug=0):
    #Assign the solution and BC into panel and trace element class
    for i, pl in enumerate(panels):
        var1 = pl.get_bdvals()
        var2 = []
        for j in range(pl.ndof):
            nodeid = mesh.getNodeId(i, j, 'Edge')
            var2.append(X[nodeid])
        if(pl.bd_Indicator == 0):  # prescribed BC is Dirichelt
                P = var1
                Q = var2
        else:
                P = var2
                Q = var1
        pl.set_PQ(P, Q)
        pl.eval_UV()
    
    eleid = 0
    for ti in range(mesh.Num_trace):  # Trace
        for i, pl in enumerate(traces[ti]):  # Trace ele
            var1 = pl.get_bdvals()
            var2 = []
            for j in range(pl.ndof):
                nodeid = mesh.getNodeId(eleid, j, 'Trace')
                var2.append(X[nodeid])
            #print('Trace %s ele %s %s' % (ti+1,i+1,nodeid+1), var1, var2)
            if(pl.bd_Indicator == 0):  # prescribed BC is Dirichelt
                P = var1
                Q = var2
            else:
                P = var2
                Q = var1
            pl.set_PQ(P, Q)
            pl.eval_UV()
            eleid = eleid + 1
        

def Field_Solve_quadratic_Trace(xi, yi, panels, traces, elementID=-1):
    #Calculate domain solution using BIE formulation-Page 105 @ BEM Introduction Course-1991
    #elementID=-1 internal point   elementID>=0 boundary point

    PI = 3.141592653
    p, u, v = 0, 0, 0
    if (elementID == -1):  # query point locate on internal domain
        #Edge element
        for i, pl in enumerate(panels):
                                                                    #  0   1   2   3   4   5
            A = GHCalc_quadratic_adapative(xi, yi, pl, 'internal')  # Hij,Gij,DUx,DQx,DUy,DQy
            Hij, Gij = A[0], A[1] 
            DUx, DQx = A[2], A[3]
            DUy, DQy = A[4], A[5]
            P,Q=pl.get_PQ()
            for j in range(1,pl.ndof+1): #Hij index start from 1
                p = p + Gij[j] * Q[j-1] - Hij[j] * P[j-1]
                u = u + DUx[j] * Q[j-1] - DQx[j] * P[j-1]
                v = v + DUy[j] * Q[j-1] - DQy[j] * P[j-1]
        #Trace element
        for ti in range(len(traces)):  # Trace
            for i, pl in enumerate(traces[ti]):  # Trace ele
                A = GHCalc_analytical(xi, yi, pl)
                A= np.array(A)/pl.length #Still not clear, but this is correct
                Gij = A[1]
                DUx = A[2]
                DUy = A[4]
                Gij=np.atleast_1d(Gij)
                DUx = np.atleast_1d(DUx)
                DUy = np.atleast_1d(DUy)
                P, Q = pl.get_PQ()
                #print(P,Q,Gij)
                for j in range(pl.ndof):
                    p = p + Gij[j] * Q[j] 
                    u = u + DUx[j] * Q[j] 
                    v = v + DUy[j] * Q[j]

    if (elementID != -1):  # query point locate on edge element
        Pts = (xi, yi)  # query point
        if(elementID < len(panels)):
            Element = panels[elementID]
        elif(elementID >= len(panels)):
            print('Not Ready Yet')
        #shape function & Node value
        phi = Element.get_ShapeFunc(Pts)
        Pi = Element.get_P()
        ui = Element.get_U()
        vi = Element.get_V()

        for i in range(Element.ndof):
            p += phi[i] * Pi[i]
            u += phi[i] * ui[i]
            v += phi[i] * vi[i]


    p = p /2 /PI
    u = u /2 /PI
    v = v /2 /PI
    return p,-u,-v


                    
                    






