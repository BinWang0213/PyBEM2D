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
from ...Tools.Geometry import cosspace,Point2Segment,GaussLib,Global2Iso,Subdivision,point_in_panel

from .Exact_Integration import Analytical_Intergration_cython, Analytical_Intergration_source_cython


######################## Solver Module-Matrix assemble and field point solve ########################
def build_matrix_all(panels, traces, sources, mesh,DDM=0, AB=[]):
    '''Assemble BEM matrix (G,H) for edge and trace

    Arguments
    ---------
    DDM         -- Domain decomposition button
                   1-on -off 
    AB          -- Input A matrix and b vector
    panels      -- List of BEM_Elements
    mesh        -- BEM mesh object
    NE          -- Number of elements

    Author:Bin Wang (binwang.0213@gmail.com)
    Date: July. 2018
    '''
    #All variables start from 1
    debug=0

    if(DDM == 1 and AB != 'none'):
        return update_BCs_trace(panels,traces,sources,mesh,AB)

    #------------Assemble matrix---------

    #1. H,G matrix for edge-edge influence
    NE = len(panels)  # number of elements
    NT = len(traces)  # number of traces
    NS = len(sources) # number of sources
    Ne = mesh.Ndof_edge  # total number of nodes on edges, assuming all of element is discontious elements
    Nt = mesh.Ndof_trace # total number of nodes on traces
    Ns = mesh.Ndof_source # total number of dofs for sources
    
    ##Coefficients for potential problem- miu/h/|k_tensor|
    k_tensor = mesh.BEMobj.k
    k_coeff = mesh.BEMobj.k_coeff
    k_det = k_tensor[0] * k_tensor[2] - k_tensor[1] * k_tensor[1]

    #---------------------Edge Elements---------------------


    #1. H,G matrix for edge-edge influence
    GEE=np.zeros((Ne, Ne), dtype=float) #double node for flux term
    HEE=np.zeros((Ne, Ne), dtype=float)


    #Compute Gij[1,2,3] Hij[1,2,3] for each node and BE
    nodeid_i=0 #local edge node index, field point side
    for i in range(NE):  # Node side
        p_i=panels[i]
        for ni in range(p_i.ndof):
            xi, yi = p_i.get_node(ni)
            
            nodeid_j=0 #local edge node index, source side
            for j in range(NE): #Ele side
                p_j=panels[j]
                G, H, Gx, Gy, Hx, Hy = Analytical_Intergration_cython(xi, yi, p_j,k_tensor)
                for nj in range(p_j.ndof):
                    HEE[nodeid_i,nodeid_j]=H[nj] * k_coeff
                    GEE[nodeid_i, nodeid_j] = G[nj] * k_coeff
                    nodeid_j = nodeid_j+1
            
            nodeid_i = nodeid_i + 1 


    #Compute the diagonal H term
    for i in range(Ne):
        HEE[i,i]=0
        for j in range(Ne):
            if(i!=j):
                HEE[i,i]=HEE[i,i]-HEE[i,j]

    #2. H,G matrix for trace-edge influence
    HTE = np.zeros((Nt, Ne), dtype=float)
    GTE = np.zeros((Nt, Ne), dtype=float)

    nodeid_t = 0
    for t in range(NT):  # Trace
        for ti in range(len(traces[t])):#Trace ele  
            for ni in range(traces[t][ti].ndof):  # Trace ele node
                xi, yi = traces[t][ti].get_node(ni)

                nodeid_e=0
                for j in range(NE):  # store Hij into element
                    G,H,Gx,Gy,Hx,Hy=Analytical_Intergration_cython(xi, yi, panels[j],k_tensor)
                    for nj in range(panels[j].ndof):
                        HTE[nodeid_t, nodeid_e] = H[nj] * k_coeff
                        GTE[nodeid_t, nodeid_e] = G[nj] * k_coeff
                        nodeid_e = nodeid_e+1
                
                nodeid_t = nodeid_t+1

    #3. H,G matrix for source-edge influence
    HSE = np.zeros((Ns, Ne), dtype=float)
    GSE = np.zeros((Ns, Ne), dtype=float)

    nodeid_s = 0
    for s in range(NS):  # Source
        xi, yi = sources[s].get_node()
        nodeid_e=0
        for j in range(NE):  # store Hij into element
            G,H,Gx,Gy,Hx,Hy=Analytical_Intergration_cython(xi, yi, panels[j],k_tensor)
            for nj in range(panels[j].ndof):
                HSE[nodeid_s, nodeid_e] = H[nj] * k_coeff
                GSE[nodeid_s, nodeid_e] = G[nj] * k_coeff
                nodeid_e = nodeid_e+1
        nodeid_s = nodeid_s+1
    
    #print('HSE',HSE)
    #print('GSE',GSE)

    #---------------------Trace Elements---------------------
    #4. H,G matrix for trace-trace influence
    HTT = np.zeros((Nt, Nt), dtype=float)
    #compute the diagonal term
    for i in range(Nt):
        HTT[i, i] = 0
        for j in range(Ne):
            HTT[i, i] = HTT[i, i] - HTE[i, j] 
    
    #GET,GTT,GST
    GET = np.zeros((Ne, Nt), dtype=float)
    GTT = np.zeros((Nt, Nt), dtype=float)
    GST = np.zeros((Ns, Nt), dtype=float)

    nodeid_t = 0
    for t in range(NT):  # Trace
        for ti in range(len(traces[t])):  # Trace ele
            for nj in range(traces[t][ti].ndof):  # Trace ele node
                
                #GET
                nodeid_e = 0
                for i in range(NE):
                    for ni in range(panels[i].ndof):
                        xi, yi = panels[i].get_node(ni)
                        G,H,Gx,Gy,Hx,Hy=Analytical_Intergration_cython(xi, yi, traces[t][ti],k_tensor)
                        GET[nodeid_e, nodeid_t] = G[nj] * k_coeff / traces[t][ti].length #Unit strength for trace
                        nodeid_e += 1
                
                #GTT
                nodeid_T = 0
                for T in range(NT):  # Trace
                    for TI in range(len(traces[T])):#Trace ele  
                        for ni in range(traces[T][TI].ndof):
                            xi, yi = traces[T][TI].get_node(ni)
                            G,H,Gx,Gy,Hx,Hy=Analytical_Intergration_cython(xi, yi, traces[t][ti],k_tensor)
                            GTT[nodeid_T,nodeid_t] = G[nj] * k_coeff / traces[t][ti].length #Unit strength for trace
                            #print(G[nj]/traces[t][ti].length)
                            nodeid_T+=1
                #GST
                nodeid_s=0
                for i in range(NS):
                    xi, yi = sources[i].get_node()
                    G,H,Gx,Gy,Hx,Hy=Analytical_Intergration_cython(xi, yi, traces[t][ti],k_tensor)
                    GST[nodeid_s, nodeid_t] = G[nj] * k_coeff / traces[t][ti].length #Unit strength for trace
                    nodeid_s+=1

                nodeid_t+=1
    
    #print('GST',GST)
    #print('GTT',GTT)

    #---------------------Source Elements---------------------
    #5. HSS - compute constant c
    HSS = np.zeros((Ns, Ns), dtype=float)
    #compute the diagonal term
    for i in range(Ns):
        HSS[i, i] = 0
        for j in range(Ne):
            HSS[i, i] = HSS[i, i] - HSE[i, j]
    
    #print('HSS',HSS)

    #GES,GTS,GSS
    GES = np.zeros((Ne, Ns), dtype=float)
    GTS = np.zeros((Nt, Ns), dtype=float)
    GSS = np.zeros((Ns, Ns), dtype=float)

    nodeid_s = 0
    for s in range(NS):  
        #GES
        nodeid_e = 0
        for i in range(NE):
            for ni in range(panels[i].ndof):
                xi, yi = panels[i].get_node(ni)
                G,Gx,Gy=Analytical_Intergration_source_cython(xi, yi, sources[s], k_tensor)
                GES[nodeid_e, nodeid_s] = G * k_coeff 
                nodeid_e += 1
        
        #GTS
        nodeid_T = 0
        for T in range(NT):  # Trace
            for TI in range(len(traces[T])):#Trace ele  
                for ni in range(traces[T][TI].ndof):
                    xi, yi = traces[T][TI].get_node(ni)
                    G,Gx,Gy=Analytical_Intergration_source_cython(xi, yi, sources[s], k_tensor)
                    GTS[nodeid_T,nodeid_s] = G * k_coeff 
                    nodeid_T+=1
        #GSS
        nodeid_S=0
        for i in range(NS):
            xi, yi = sources[i].get_node()
            if(i==s): xi+=1e-3 #this is equivalent to wellbore radius
            #if(i==s): G=-6.90327 #G=-6.90327*k_tensor[0]
            G,Gx,Gy=Analytical_Intergration_source_cython(xi, yi, sources[s], k_tensor)
            #print((i,s),k_coeff,G)
            #if(i==s): G=-6.90327
            GSS[nodeid_S, nodeid_s] = G * k_coeff 
            nodeid_S+=1

        nodeid_s+=1

    #print('GES',GES)
    #print('GTS',GTS)
    #print('GSS',GSS)

    
    A = np.block([[HEE, np.zeros((Ne, Nt+Ns))],  # {x}={p_e,p_t,p_s}'
                  [HTE, HTT,np.zeros((Nt, Ns))],
                  [HSE,np.zeros((Ns, Nt)),HSS] ])
    B = np.block([[GEE, GET, GES],   #{b}={q_e,q_t,q_s}'
                   [GTE, GTT, GTS],
                   [GSE, GST, GSS]])
    
    #print('H\n',A)
    #print('G\n',B,np.sum(B[0:-1,-1]))

    #------------Applying BCs---------
    #2. Reorder the matrix
    # bd_type, 1-neumann, 0-dirichlet, odd-left node, even-right node
    debug=0

    #Apply BC for edge elements
    for i, pl in enumerate(panels):
        if(pl.bd_Indicator==0):#Dirichelt
            for j in range(pl.ndof):
                nodeid = mesh.getNodeId(i, j, 'Edge')
                #print('Ele-localdof-globaldof',i,j,nodeid)
                if(debug): print("Edge Ele:%s A,Col%s<->B,Col%s" % (i+1,nodeid+1,nodeid+1))
                for k in range(mesh.Ndof):
                    temp = A[k, nodeid]
                    A[k, nodeid] = -B[k, nodeid]
                    B[k, nodeid] = -temp
        elif(pl.bd_Indicator == 1):#Neumann or Robin
            if(debug): 
                print('Edge Ele:%s Neumann BC doesn\'t need to interchange col'%(i+1))                
            if(pl.Robin_alpha > 0): #Robin
                for j in range(pl.ndof):
                    nodeid = mesh.getNodeId(i, j, 'Edge')
                    for k in range(mesh.Ndof):
                        A[k, nodeid] += pl.Robin_alpha * B[k, nodeid]
                    if(debug): print('Robin-Ele:%s, A,Col%s+=B,Col%s'% (i+1,nodeid+1,nodeid+1))

    #Apply BC for trace elements
    eleid = 0
    for ti in range(NT):  # Trace
        if(debug): print("--Trace ", ti + 1)
        for i,pl in enumerate(traces[ti]):#Trace ele
            if(pl.bd_Indicator == 0):  # Dirichlet
                for j in range(pl.ndof):
                    nodeid = mesh.getNodeId(eleid, j, 'Trace')
                    if(debug): print("Trace Ele:%s A,Col%s<->B,Col%s" % (eleid+1+mesh.Ne_edge,nodeid+1,nodeid+1))
                    for k in range(mesh.Ndof):
                        temp = A[k, nodeid]
                        A[k, nodeid] = -B[k, nodeid]
                        B[k, nodeid] = -temp

            elif(pl.bd_Indicator==1):#Neumann or Robin
                if(debug):
                    print('Trace Ele:%s Neumann BC doesn\'t need to interchange col'%(eleid+1+mesh.Ne_edge))
                if(pl.Robin_alpha > 0):  # Robin
                    for j in range(pl.ndof):
                        nodeid = mesh.getNodeId(eleid, j, 'Trace')
                        for k in range(mesh.Ndof):
                            A[k, nodeid] += pl.Robin_alpha * B[k, nodeid]
                        if(debug): print('Trace Robin-Ele:%s, A,Col%s+=B,Col%s'% (eleid+1+mesh.Ne_edge,nodeid+1,nodeid+1))

            eleid = eleid + 1    

    #Apply BC for source nodes
    for i, pl in enumerate(sources):
        if(pl.bd_Indicator==0):#Dirichelt
            nodeid = mesh.getNodeId(i,0,'Source')
            if(debug): print("Source Ele:%s A,Col%s<->B,Col%s" % (eleid+mesh.Ne_edge+i+1,nodeid+1,nodeid+1))
            for k in range(mesh.Ndof): #replace column
                temp = A[k, nodeid]
                A[k, nodeid] = -B[k, nodeid]
                B[k, nodeid] = -temp
        elif(pl.bd_Indicator == 1):#Neumann or Robin
            if(debug): 
                print('Source Ele:%s Neumann BC doesn\'t need to interchange col'%(i+1))
    

    #Collecting prescribed BC values, RHS 
    if(debug): print('Prescribed Value')
    b = np.zeros(mesh.Ndof, dtype=float)

    #Apply for elements
    for i, pl in enumerate(panels):
        bdvals = pl.get_bdvals()
        for j in range(pl.ndof):
            nodeid = mesh.getNodeId(i, j, 'Edge')
            b[nodeid]=bdvals[j]
            if(debug): print("[Edge] Ele:%s Node:%s BC_Val:%s" % (i+1,nodeid+1,b[nodeid]))
    
    #Apply for traces
    eleid = 0
    for ti in range(mesh.Num_trace):  # Trace
        for i, pl in enumerate(traces[ti]):  # Trace ele
            bdvals = pl.get_bdvals()
            for j in range(pl.ndof):
                nodeid = mesh.getNodeId(eleid, j, 'Trace')
                b[nodeid] = bdvals[j]
                if(debug): print("[Trace] Ele:%s Node:%s BC_Val:%s" % (eleid+1+mesh.Ne_edge,nodeid+1,b[nodeid]))
            eleid = eleid + 1

    #Apply for sources
    for i, pl in enumerate(sources):
        nodeid = mesh.getNodeId(i,0,'Source')
        b[nodeid] = pl.get_bdvals()[0] #only 1 dof for source
        if(debug): print("[Source] Ele:%s Node:%s BC_Val:%s" % (eleid+1+mesh.Ne_edge+i,nodeid+1,b[nodeid]))
    
    #Get the final RHS matrix
    b=np.dot(B,b)

    #print(A)
    #print(b)
    #print(B)

    return A,b,B#,HEE, HET, HTE, HTT, GEE, GET, GTE, GTT


def solution_allocate_all(panels, traces, sources, mesh, X, debug=0):
    '''Assign the solution (P,Q,R) into the BEM_Elements

    Arguments
    ---------
    panels      -- array of BEM_elements
    X           -- solution vector for each node
    mesh        -- BEM mesh class
    bd_vals     -- prescribed boundary conditons
    sol_vals    -- BEM solution 

    Author:Bin Wang (binwang.0213@gmail.com)
    Date: July. 2018
    '''
    debug=0
    #Solution allocation for Edge
    for i, pl in enumerate(panels):
        var1 = pl.get_bdvals()
        var2 = []
        for j in range(pl.ndof):
            nodeid = mesh.getNodeId(i, j, 'Edge')
            var2.append(X[nodeid])
        if(pl.bd_Indicator == 1):#Neumann BC
            P = var2
            Q = var1
            if(pl.Robin_alpha > 0):#Robin BC
                for j in range(pl.ndof):
                    Q[j] = Q[j] - pl.Robin_alpha * P[j]
        elif(pl.bd_Indicator == 0):  #Dirichelt BC
            P = var1
            Q = var2
        if(debug): print('Ele',i,'P',P,'Q',Q)
        pl.set_PQ(P, Q)
        #pl.eval_UV()

    #Solution allocation for Traces
    eleid = 0
    for ti in range(mesh.Num_trace):  # Trace
        for i, pl in enumerate(traces[ti]):  # Trace ele
            var1 = pl.get_bdvals()
            var2 = []
            for j in range(pl.ndof):
                nodeid = mesh.getNodeId(eleid, j, 'Trace')
                var2.append(X[nodeid])
            #print('Trace %s ele %s %s' % (ti+1,i+1,nodeid+1), var1, var2)
            if(pl.bd_Indicator == 1):#Neumann BC
                P = var2
                Q = var1
                if(pl.Robin_alpha > 0):#Robin BC
                    for j in range(pl.ndof):
                        Q[j] = Q[j] - pl.Robin_alpha * P[j]
            elif(pl.bd_Indicator == 0):  #Dirichelt BC
                P = var1
                Q = var2
            if(debug): print('Ele',i,'P',P,'Q',Q)
            pl.set_PQ(P, Q)
            #pl.eval_UV()
            eleid = eleid + 1
    
    #Solution allocation for sources
    for i, pl in enumerate(sources):
        nodeid = mesh.getNodeId(i,0,'Source')
        if(pl.bd_Indicator == 0):  # prescribed BC is Dirichelt
            pl.set_PQ(pl.get_bdvals(),[X[nodeid]])
        else: # Neumann BC 
            pl.set_PQ([X[nodeid]],pl.get_bdvals())

    # [Additional for boundary solution interp] Evaluate the pressure derivate on nodes
    for i in range(len(panels)):
        p_i = panels[i]
        Nodes = p_i.get_nodes()
        U = [0] * p_i.ndof
        V = [0] * p_i.ndof
        for ni in range(p_i.ndof):
            x, y = Nodes[ni][0], Nodes[ni][1]
            P, U[ni], V[ni] = Field_Solve_all(x, y, panels,traces,sources, mesh, elementID=-2)
        p_i.set_UV(U, V)
    
    for ti in range(mesh.Num_trace):  # Trace
        for i, pl in enumerate(traces[ti]):  # Trace ele
            pl.get_nodes()
            U = [0] * p_i.ndof
            V = [0] * p_i.ndof
            for ni in range(p_i.ndof):
                x, y = Nodes[ni][0], Nodes[ni][1]
                P, U[ni], V[ni] = Field_Solve_all(x, y, panels,traces,sources, mesh, elementID=-2)
            pl.set_UV(U,V)

def Field_Solve_all(xi,yi,panels,traces,sources,mesh,elementLoc='Interior',elementID=-1):
    #Calculate domain solution using BIE formulation-Page 105 @ BEM Introduction Course-1991
    #elementID=-1 internal point   elementID>=0 boundary point

    NE = len(panels)
    NT = len(traces)
    NS = len(sources)
    p,u,v=0,0,0

    pi=3.141592653

    ##Coefficients for potential problem- miu/h/|k_tensor|
    k_tensor = mesh.BEMobj.k
    k_coeff = mesh.BEMobj.k_coeff
    miu = mesh.BEMobj.miu
    
    #if (elementID == -1 or elementID == -2):  # query point locate on the internal domain
    if(elementLoc=='Interior'):   
        #Edge element contribution
        for i in range(NE):
            Element = panels[i]
            G, H, Gx, Gy, Hx, Hy = Analytical_Intergration_cython(xi, yi, Element,k_tensor)
            P, Q = Element.get_PQ()
            for nj in range(Element.ndof):
                p += G[nj]*k_coeff*Q[nj] - H[nj]*k_coeff*P[nj]
                u += Gx[nj]*k_coeff*Q[nj] - Hx[nj]*k_coeff*P[nj]
                v += Gy[nj]*k_coeff*Q[nj] - Hy[nj]*k_coeff*P[nj]

        #Trace element contribution
        for t in range(NT):  # Trace
            for ti in range(len(traces[t])):  # Trace ele
                G, H, Gx, Gy, Hx, Hy = Analytical_Intergration_cython(xi, yi, traces[t][ti],k_tensor)
                Lj = traces[t][ti].length
                Q = traces[t][ti].get_Q()
                # Internal source is the unit length strength
                #print(P,Q,Gij)
                for j in range(traces[t][ti].ndof):
                    p += G[j]*k_coeff/Lj * Q[j]
                    u += Gx[j]*k_coeff/Lj * Q[j]
                    v += Gy[j]*k_coeff/Lj * Q[j]

        #Source element contribution
        for s in range(NS):
            if(xi==sources[s].x0 and yi==sources[s].y0): xi+=1e-3 #avoid singularity
            G,Gx,Gy=Analytical_Intergration_source_cython(xi, yi, sources[s], k_tensor)
            Q = sources[s].get_Q()
            p +=  G*k_coeff * Q[0]
            u +=  Gx*k_coeff * Q[0]
            v +=  Gy*k_coeff * Q[0]


        P = p
        U = (u * k_tensor[0] + v * k_tensor[1])
        V = (u * k_tensor[1] + v * k_tensor[2])

        if(elementID == -2):  # Node on boundary c=1/2
            P = P * 2.0
            U = -U * 2.0
            V = -V * 2.0

    #if (elementID != -1 and elementID != -2):  # query point locate on the element
    else:  # query point locate on the element   
        Pts=(xi,yi) #query point

        if(elementLoc=='Trace'):#Trace element elementID[0]=[TraceID,EleID]
            #print("Query ponit is on trace",Pts,elementID)
            #Element = traces[elementID[0][0]][elementID[0][1]]
            #Trace is not accurate, get the solution next to it
            return Field_Solve_all(xi+1e-5,yi-1e-3,panels,traces,sources,mesh,elementID=-1)

        if(elementLoc=='Edge'):
            Element = panels[elementID[0]]
            #print('Pts on the %d Edge element'%(elementID[0]))
            #shape function & Node value
            phi= Element.get_ShapeFunc(Pts)
            Pi = Element.get_P()
            ui = Element.get_U()
            vi = Element.get_V()

            P = np.dot(phi, Pi)
            U = np.dot(phi, ui)
            V = np.dot(phi, vi)
        if(elementLoc=='Source'):
            Element = sources[elementID]

            P=Element.get_P()[0]
            U=0.0 #velocity is singularity at the source point
            V=0.0
        
        
        #debug
        #print('Point',Pts)
        #print(phi)
        #print('Element',elementID[0]+1,Element.nx,Element.ny)
        #print(Pi,ui,vi)

    #darcy flow -k/u*dp/dx
    return P, -U, -V


def update_BCs_trace(panels,traces,sources,mesh,AB=[]):
    
    debug = 0
    A = AB[0]
    B = AB[1]

    #Collecting prescribed BC values
    if(debug):
        print('Prescribed Value')
    
    b = np.zeros(mesh.Ndof, dtype=float)
    # Edge BC
    for i, pl in enumerate(panels):
        bdvals = pl.get_bdvals()
        for j in range(pl.ndof):
            nodeid = mesh.getNodeId(i, j, 'Edge')
            b[nodeid] = bdvals[j]
            if(debug):
                print("Ele:%s Node:%s BC_Val:%s" %
                      (i + 1, nodeid + 1, b[nodeid]))

    # Trace BC
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

    # Source BC
    for i, pl in enumerate(sources):
        nodeid = mesh.getNodeId(i,0,'Source')
        b[nodeid] = pl.get_bdvals()[0] #only 1 dof for source
        if(debug): print("[Source] Ele:%s Node:%s BC_Val:%s" % (eleid+1+mesh.Ne_edge+i,nodeid+1,b[nodeid]))


    #Get the final RHS matrix
    b = np.dot(B, b)
    
    return A,b,B


    

