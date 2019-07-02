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
cimport numpy as np
import numpy as np
import math
import time
from Lib.Tools.Geometry import cosspace,Point2Segment,GaussLib,Global2Iso,Subdivision,point_in_panel

from .Exact_Integration import Analytical_Intergration_cython


######################## Solver Module-Matrix assemble and field point solve ########################
def build_matrix(panels, mesh,DDM=0, AB=[]):
    '''Assemble BEM matrix (G,H)

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
        return update_BCs(panels,mesh,AB)

    NE=len(panels) #number of elements
    N = mesh.Ndof  # number of nodes, assuming all of element is discontious elements
    
    ##Coefficients for potential problem- miu/h/|k_tensor|
    k_tensor = mesh.BEMobj.k
    k_coeff = mesh.BEMobj.k_coeff

    G_mat=np.zeros((N, N), dtype=float) #double node for flux term
    H_mat=np.zeros((N, N), dtype=float)

    #1. Compute Gij[1,2,3] Hij[1,2,3] for each node and BE
    for i in range(NE):  # Node side
        p_i=panels[i]
        for ni in range(p_i.ndof):
            xi, yi = p_i.get_node(ni)
            nodeid_i = mesh.getNodeId(i, ni, 'Edge')
            for j in range(NE): #Ele side
                p_j=panels[j]
                #print('Source Ele',j+1,'Target Node',xi,yi)
                #print(p_j)
                G, H, Gx, Gy, Hx, Hy = Analytical_Intergration_cython(xi, yi, p_j,k_tensor)
                for nj in range(p_j.ndof):
                    nodeid_j = mesh.getNodeId(j, nj, 'Edge')
                    H_mat[nodeid_i,nodeid_j]=H[nj]*k_coeff
                    G_mat[nodeid_i,nodeid_j]=G[nj]*k_coeff
    
    #Compute the diagonal term
    for i in range(N):
        H_mat[i,i]=0
        for j in range(N):
            if(i!=j):
                H_mat[i,i]=H_mat[i,i]-H_mat[i,j]
    
    #2. Applying boundary conditions
    #Assemble matrix A and reorder matrix H and G (Switching column)
    for j in range(NE):  # Node side
        # If boundary condition is Dirichlet then interchange G-H G*U=H*P
        if (panels[j].bd_Indicator == 0):
            p_j=panels[j]
            for nj in range(p_j.ndof):
                nodeid_j = mesh.getNodeId(j, nj, 'Edge')
                for nodeid_i in range(N):  # All rows
                    temp = H_mat[nodeid_i, nodeid_j]
                    H_mat[nodeid_i, nodeid_j] = -G_mat[nodeid_i, nodeid_j]
                    G_mat[nodeid_i, nodeid_j] = -temp

    #Robin Boundary Condition
    #Reference https://www.researchgate.net/project/Extending-
    #the-boundary-element-method-to-the-generalised-Robin-boundary-condition
    for j in range(NE):  # Node side
        # Robin modification H-robin*G
        if (panels[j].bd_Indicator == 1 and panels[j].Robin_alpha > 0):
            p_j = panels[j]
            for nj in range(p_j.ndof):
                nodeid_j = mesh.getNodeId(j, nj, 'Edge')
                for nodeid_i in range(N):  # All rows
                    H_mat[nodeid_i, nodeid_j] += panels[j].Robin_alpha * G_mat[nodeid_i, nodeid_j]  # beta*G

    #3.Assemble RHS vector b
    A = H_mat
    
    b = np.zeros(N, dtype=float)
    for nodeid_i in range(N):  # All rows
        b[nodeid_i]=0
        for j in range(NE):  # Node side
            p_j = panels[j]
            for nj in range(p_j.ndof):
                nodeid_j = mesh.getNodeId(j, nj, 'Edge')
                bdvals = p_j.get_bdvals()
                b[nodeid_i] += G_mat[nodeid_i, nodeid_j] * bdvals[nj]  # beta*G

    return A, b, G_mat


def solution_allocate(panels, mesh, X, debug=1):
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
    NE=len(panels) #BE number

    #Pressure and flux on element nodes
    for i in range(NE):
        p_i=panels[i]
        bd_vals = p_i.get_bdvals()
        sol_vals = []
        for ni in range(p_i.ndof):
            nodeid_i = mesh.getNodeId(i, ni, 'Edge')
            sol_vals.append(X[nodeid_i])
        # Neumann Boundary
        if p_i.bd_Indicator == 1:  
            Q=bd_vals
            P=sol_vals
        # Robin Boundary
            if(p_i.Robin_alpha > 0):  # This is a Robin boundary condition
                for ni in range(p_i.ndof):
                    nodeid_i = p_i.ndof * i + ni
                    Q[ni] = Q[ni] - p_i.Robin_alpha * P[ni]
        # Dirichelt Boundary
        elif p_i.bd_Indicator == 0:
            P = bd_vals
            Q = sol_vals

        if(debug): print('Ele',i,'P',P,'Q',Q)
        p_i.set_PQ(P,Q)
        p_i.eval_UV(mesh.BEMobj.k)  # Evaluate the pressure derivate on nodes


cpdef Field_Solve(double xi,double yi,object panels,object mesh,elementID=-1):
    #Calculate domain solution using BIE formulation-Page 105 @ BEM Introduction Course-1991
    #elementID=-1 internal point   elementID>=0 boundary point

    cdef int i,NE
    cdef double p,u,v
    cdef double k_coeff
    #cdef double[:] k_tensor,G,H,Gx,Gy,Hx,Hy,P,Q
    cdef double[:] k_tensor,P,Q
    cdef double[:] Pi,ui,vi,phi

    NE=len(panels)
    p,u,v=0.0,0.0,0.0
    
    ##Coefficients for potential problem- miu/h/|k_tensor|
    k_tensor = mesh.BEMobj.k
    k_coeff = mesh.BEMobj.k_coeff

    if (elementID==-1): #query point locate on the internal domain

        for i in range(NE):
            Element = panels[i]
            G, H, Gx, Gy, Hx, Hy = Analytical_Intergration_cython(xi, yi, Element,k_tensor)
            P, Q = Element.get_PQ()

            #p += np.dot(G, Q)  - np.dot(H,P)
            #u += np.dot(Gx, Q) - np.dot(Hx,P)
            #v += np.dot(Gy, Q) - np.dot(Hy,P)
            for j in range(Element.ndof):
                p = p + G[j] * Q[j] - H[j] * P[j]
                u = u + Gx[j] * Q[j] - Hx[j] * P[j]
                v = v + Gy[j] * Q[j] - Hy[j] * P[j]

        p = p * k_coeff
        u = u * k_coeff
        v = v * k_coeff
    
    if (elementID!=-1): #query point locate on the element

        Pts=(xi,yi) #query point
        Element=panels[elementID[0]]
        #print("Ele",elementID)
        #print(Element)
        #shape function & Node value
        phi= Element.get_ShapeFunc(Pts)
        Pi = Element.get_P()
        ui = Element.get_U()
        vi = Element.get_V()

        
        #p = np.dot(phi,Pi)
        #u = np.dot(phi, ui)
        #v = np.dot(phi, vi)

        for i in range(Element.ndof):
            p+=phi[i]*Pi[i]
            u+=phi[i]*ui[i]
            v+=phi[i]*vi[i]
        

        #debug
        #print('Point',Pts)
        #print(phi)
        #print('Element',elementID[0]+1,Element.nx,Element.ny)
        #print(Pi,ui,vi)



    #darcy flow -k/u*dp/dx
    return p,-u,-v


def update_BCs(panels,mesh,AB=[]):
    
    NE=len(panels) #number of elements
    N = mesh.Ndof  # number of nodes
    
    #Collecting prescribed BC values
    debug = 0
    A = AB[0]
    G_mat = AB[1]

    #3.Assemble vector b
    b = np.zeros(N, dtype=float)
    for nodeid_i in range(N):  # All rows
        b[nodeid_i] = 0
        for j in range(NE):  # Node side
            p_j = panels[j]
            for nj in range(p_j.ndof):
                nodeid_j = mesh.getNodeId(j, nj, 'Edge')
                bdvals = p_j.get_bdvals()
                b[nodeid_i] += G_mat[nodeid_i, nodeid_j] * bdvals[nj]  # beta*G
    
    return A,b,G_mat


    

