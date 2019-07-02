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
from .Constant_element import *


######################## Solver Module-Matrix assemble and field point solve ########################
def build_matrix_const_Trace(panels, traces,mesh,DDM=0,AB=[]):
    #Reference A BEM solution of steady-state ﬂow problems in discrete fracture networks with minimization of core storage
    #Edge use quad element and trace use constant element

    #All variables start from 1
    debug = 0


    if(DDM==1 and AB!='none'):
        return update_matrix_const_Trace(panels, traces, mesh, AB)

    NE = len(panels)  # number of elements
    N = mesh.Ndof_edge  # number of nodes
    #H IS A SQUARE MATRIX (2*NE,2*NE); G IS RECTANGULAR (2*NE,3*NE)
    #Index from 1

    G = np.empty((N, N), dtype=float)
    H = np.empty((N, N), dtype=float)
    PI = 3.141592653

    #Assemble G,H matrix
    for i, p_i in enumerate(panels):  # nodes
        for j, p_j in enumerate(panels):  # BEs
            if i == j:
                G[i, j] = p_i.length * (np.log(2 / p_i.length) + 1)
                H[i, j] = PI  # Hii=1/2*2PI due to intergral=0
            if i != j:
                xi, yi = p_i.xc, p_i.yc
                TEMP = GHCalc_analytical(xi, yi, p_j)
                H[i, j] = TEMP[0]
                G[i, j] = TEMP[1]

    HEE = H
    GEE = G

    #2. ATE-HTE BTE-GTE -Influence factor from edge to trace
    Ne = mesh.Ndof_edge
    Nt = mesh.Ndof_trace
    HTE = np.zeros((Nt, Ne), dtype=float)
    GTE = np.zeros((Nt, Ne), dtype=float)

    tii = 0
    for t in range(len(traces)):  # which trace
        for ti in range(len(traces[t])):  # which node on a trace

            for j in range(NE):  # store Hij into element
                Temp = GHCalc_analytical(traces[t][ti].xc, traces[t][ti].yc, panels[j])
               
                HTE[tii, j] = Temp[0]
                GTE[tii, j] = Temp[1]
            tii = tii + 1  # node index for trace

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
    GET = np.zeros((Ne, Nt), dtype=float)
    for i in range(NE):
        tii = 0
        for t in range(len(traces)):  # which trace
            for ti in range(len(traces[t])):  # which node on a trace
                Temp = GHCalc_analytical(panels[i].xc, panels[i].yc, traces[t][ti])[1] 
                Temp = Temp / traces[t][ti].length
                GET[i, tii] = Temp
                tii = tii + 1

    #6. GTT-BTT
    GTT = np.zeros((Nt, Nt), dtype=float)
    TII = 0
    for T in range(len(traces)):
        for TI in range(len(traces[T])):
            tii = 0
            for t in range(len(traces)):  # which trace
                for ti in range(len(traces[t])):  # which node on a trace
                    if tii != TII:
                        Temp_Gij = GHCalc_analytical(traces[T][TI].xc, traces[T][TI].yc, traces[t][ti])[1] 
                        Temp_Gij = Temp_Gij / traces[t][ti].length
                        GTT[TII, tii] = Temp_Gij
                    #compute the diagonal term
                    if tii == TII:
                        #GTT[TII,tii]=traces[t][ti].length*(np.log(2/traces[t][ti].length)+1)/traces[t][ti].length
                        GTT[TII, tii] = (np.log(2 / traces[t][ti].length) + 1)
                    tii = tii + 1
            TII = TII + 1


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

    return A,b,B,HEE,HET,HTE,HTT,GEE,GET,GTE,GTT


def solution_allocate_const_Trace(panels, traces, X, mesh, debug=0):
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
        

def update_matrix_const_Trace(panels, traces, mesh, AB=[]):
    debug=0
    A=AB[0]
    B=AB[1]
    #Collecting prescribed BC values
    if(debug):
        print('Prescribed Value')
    b = np.zeros(mesh.Ndof, dtype=float)
    for i, pl in enumerate(panels):
        bdvals = pl.get_bdvals()
        for j in range(pl.ndof):
            nodeid = mesh.getNodeId(i, j, 'Edge')
            b[nodeid] = bdvals[j]
            if(debug):
                print("Ele:%s Node:%s BC_Val:%s" %
                      (i + 1, nodeid + 1, b[nodeid]))

    eleid = 0
    for ti in range(mesh.Num_trace):  # Trace
        for i, pl in enumerate(traces[ti]):  # Trace ele
            bdvals = pl.get_bdvals()
            for j in range(pl.ndof):
                nodeid = mesh.getNodeId(eleid, j, 'Trace')
                b[nodeid] = bdvals[j]
                if(debug):
                    print("Ele:%s Node:%s BC_Val:%s" %
                          (eleid + 1 + mesh.Ne_edge, nodeid + 1, b[nodeid]))
            eleid = eleid + 1

    #Get the final RHS matrix
    b = np.dot(B, b)

    return A,b,B


def Field_Solve_const_Trace(xi, yi, panels, traces, elementID=-1):
    #Calculate domain solution using BIE formulation-Page 105 @ BEM Introduction Course-1991
    #elementID=-1 internal point   elementID>=0 boundary point

    PI = 3.141592653
    p, u, v = 0, 0, 0
    if (elementID == -1):  # query point locate on internal domain
        #Edge element
        for i, pl in enumerate(panels):
                                                                    #  0   1   2   3   4   5
            A = GHCalc_analytical(xi, yi, pl)  # Hij,Gij,DUx,DQx,DUy,DQy
            Hij, Gij = A[0], A[1] 
            DUx, DQx = A[2], A[3]
            DUy, DQy = A[4], A[5]
            Hij = np.atleast_1d(Hij)
            Gij = np.atleast_1d(Gij)
            DUx = np.atleast_1d(DUx)
            DUy = np.atleast_1d(DUy)
            DQx = np.atleast_1d(DQx)
            DQy = np.atleast_1d(DQy)
            P,Q=pl.get_PQ()
            for j in range(pl.ndof): #Hij index start from 1
                p = p + Gij[j] * Q[j] - Hij[j] * P[j]
                u = u + DUx[j] * Q[j] - DQx[j] * P[j]
                v = v + DUy[j] * Q[j] - DQy[j] * P[j]
        #Trace element
        for ti in range(len(traces)):  # Trace
            for i, pl in enumerate(traces[ti]):  # Trace ele
                A = GHCalc_analytical(xi, yi, pl)
                A= np.array(A)/pl.length #Internal source is the unit length strength
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


                    
                    






