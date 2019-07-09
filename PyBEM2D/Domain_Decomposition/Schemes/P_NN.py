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

#######################################
#
#  Parallel Neumann-Neumann Method
#
#######################################

def PNN(obj,alpha,TOL,max_iter,opt):
        """Neumann-Neumann iterative loop
           Reference: Section 3.2 in the reference paper
        ------------------------
        |  Current | Connected |
        |   Domain |   Domain  |
        ------------------------
             Intersection
        
        Non-conforming mesh are supported
        Intersection may have different nodes on two domain
        
        Update flux(q) in k+1 steps:
            q_k+1=q_k-alpha*(h_left_k-h_right_k)
            q_left=-q_right=q_k+1
        
        Arguments
        ---------
        Num_shared_edge   -- Number of intersection in a BEMobj domain. e.g. 1 domain may have 2 intersections
        ConnectObjID      -- Index of connected domain in the list of obj.BEMobjs
        IntersectID       -- Index of intersection in the list of obj.Intersects
        Intersect         -- Two end coords of a intersection edge
        bdID              -- Boundary index of the current domain
        bdID_connect      -- Boundary index of the connected domain
        CurrentNodes      -- Intersection nodes in the current domain
        ConnectNodes      -- Intersection nodes in the connected domain
        P(Q)_current         -- Solution variables in the nodes of current domain
        P(Q)_connect         -- Interpolated solution variables from connected domain
                             at the nodes of current domain
        Q_new(old)        -- Updated(old) Neumann BC on the intersection
        MaxIter           -- Maximum iteration number
        
        Author:Bin Wang(binwang.0213@gmail.com)
        Date: July. 2017
        """

        debug1=0
        debug2=0
        
        #for optimal relxation parameters
        NumInt=len(obj.Intersects)
        Q_old_old = [[] for i in range(NumInt)]  # q^k-1 for current side
        Q_conn_old_old= [[] for i in range(NumInt)]  # q^k-1 for current side
        P_cur_old = [[] for i in range(NumInt)]  # h^k-1 for current side
        P_con_old = [[] for i in range(NumInt)]  # h^k-1 for connect side
        AB_mat = []  # BEM Matrix for each domain

        MaxIter=max_iter
        for it in range(MaxIter):
            if(debug2): print('----Loop:',it+1)
            error_final=0.0
            error=[]

            if(it>2 and opt==1):
                alpha_opt=PNN_OPT(obj,Q_old_old,Q_conn_old_old,P_cur_old,P_con_old,alpha)
                alpha=alpha_opt
            
            #Step1. Prepare and update BCs for all domains
            for IntID in range(NumInt):#For each intersection
                DomainID, DomainID_connect = obj.Intersects[IntID][0], obj.Intersects[IntID][1]
                EdgeID, EdgeID_connect = obj.Intersects[IntID][2], obj.Intersects[IntID][3]
                BDType=obj.BEMobjs[DomainID].Mesh.getBDType(EdgeID)

                if(debug1): 
                    print('Intersection',IntID,'Domain(%s->%s)'%(DomainID,DomainID_connect),'BD id(%s->%s)'%(EdgeID,EdgeID_connect))
                

                #Init iteration
                if(it==0):
                    EdgeDof = obj.BEMobjs[DomainID].Mesh.getBDDof(EdgeID)
                    Q_old=np.zeros(EdgeDof)
                    Q_old_connect=Q_old
                    Q_new = Q_old
                    Q_new_connect=Q_old
                    P_current=Q_old
                    P_connect=Q_old

                #Normal iterations
                elif(it>0):
                    PQ = obj.BEMobjs[DomainID].PostProcess.get_BDSolution(EdgeID)
                    PQ_connect = obj.BEMobjs[DomainID_connect].PostProcess.get_BDSolution(EdgeID_connect)

                    Q_old=PQ[1]
                    Q_old_connect= PQ_connect[1]
                    P_current = PQ[0]
                    P_connect = PQ_connect[0]                
                    if(debug2): print('P_Current',P_current,'P_Connect',P_connect)
                    
                    #the dof on the other side is reversed
                    if(BDType=='Edge'):  Q_new=Q_old-alpha*(P_current-np.flip(P_connect)) 
                    else: Q_new=Q_old-alpha*(P_current-P_connect)
                    
                    if(BDType=='Edge'): Q_new_connect=Q_old_connect-alpha*(-np.flip(P_current)+P_connect)
                    else: Q_new_connect=Q_old_connect-alpha*(-P_current+P_connect)                    
                    if(debug2): print('q_new',Q_new,'q_old',Q_old)
                    
                    error.append(max(abs(Q_new-Q_old))/max(abs(Q_new)))
                    #print(abs(Q_new-Q_old),abs(Q_new))
                    
                #Update new Neumann BC into system
                bc_Neumann = [(EdgeID, Q_new)]
                obj.BEMobjs[DomainID].set_BoundaryCondition(NeumannBC=bc_Neumann,update=1,mode=1,debug=0)
                bc_Neumann = [(EdgeID_connect, Q_new_connect)]
                obj.BEMobjs[DomainID_connect].set_BoundaryCondition(NeumannBC=bc_Neumann,update=1,mode=1,debug=0)
                
                Q_old_old[IntID]=Q_old  #q_k-1 for current side
                Q_conn_old_old[IntID]=Q_old_connect #q_k-1 for connect side
                P_cur_old[IntID]=P_current#h_k-1 for current side
                P_con_old[IntID]=P_connect#h_k-1 for connect side
            
            #Collect error for plot convergence
            if(it>0):
                error_final=max(error)
                if(it%(MaxIter/50)==0):
                    print('%s\t%s\t\talpha:\t%s'%(it,error_final,alpha))
                obj.error_abs.append(error_final)
            
            #Step2. Update the solution for all fractures
            for i in range(obj.NumObj):#For each subdomain
                if(it == 0):  # Store the intial BEM Matrix
                    AB_mat.append(obj.BEMobjs[i].Solve())
                else:  # Update solution by only update the boundary condition, Fast
                    AB_mat[i] = obj.BEMobjs[i].Solve(DDM=1, AB=[AB_mat[i][0], AB_mat[i][2]],debug=0)
            
            if(it>5 and error_final<TOL):
                print('Converged at',it,'Steps! TOL=',TOL)
                print("Dirichelt",P_current)
                print("Neumann",Q_new)
                break
        obj.plot_Convergence()
        
def PNN_OPT(obj, Q_old_old,Q_conn_old_old,P_cur_old,P_con_old,alpha_old):
        #Calculate the optimal relxation paramters based on error function J
        #Equation 16 in the Reference Paper

        nom=0.0
        denom=0.0

        NumInt = len(obj.Intersects)
        for IntID in range(NumInt):  # For each subdomain
            DomainID, DomainID_connect = obj.Intersects[IntID][0], obj.Intersects[IntID][1]
            EdgeID, EdgeID_connect = obj.Intersects[IntID][2], obj.Intersects[IntID][3]
            BDType=obj.BEMobjs[DomainID].Mesh.getBDType(EdgeID)

            #Local bdID is determined using intersection coordinates
            PQ = obj.BEMobjs[DomainID].PostProcess.get_BDSolution(EdgeID)
            PQ_connect = obj.BEMobjs[DomainID_connect].PostProcess.get_BDSolution(EdgeID_connect)
                    
            Q_old=PQ[1]
            Q_old_conn=PQ_connect[1]
            P_current = PQ[0]                    
            P_connect = PQ_connect[0] #obj.Interp_intersection(i,ConnectObjID,Intersect)#(Current,Connect,Intersect)

            #for optimal relxation parameters

            #alpha current side
            q_dif=Q_old-Q_old_old[IntID]
            h_cur_dif=P_current-P_cur_old[IntID]
            h_con_dif=P_connect-P_con_old[IntID]
            if(BDType=='Edge'): h_ba=h_cur_dif - np.flip(h_con_dif)
            else: h_ba=h_cur_dif - h_con_dif
            #print("q_dif2",q_dif,h_ba)
            #print('nom2',np.inner(q_dif,h_ba))
            #print('dnom2',np.linalg.norm(h_ba)**2)
            nom+=np.inner(q_dif,h_ba)
            denom+=np.linalg.norm(h_ba)**2
            #print(nom,denom)

            #alpha connect side
            if(BDType=='Edge'): h_ab=h_con_dif - np.flip(h_cur_dif)
            else: h_ab=h_con_dif - h_cur_dif
            q_con_dif=Q_old_conn - Q_conn_old_old[IntID]
            nom += np.inner(q_con_dif, h_ab)
            denom += np.linalg.norm(h_ab)**2
                    
        alpha_opt=nom/denom
        #Test of bounded case
        if(alpha_opt<0.0):#Special case: P-DD may have negative alpha
            alpha_opt=alpha_old#Use the alpha from the last step
            print("Warning! Negative alpha!")
        #print('!!!',-nom,denom,alpha_opt)
        return alpha_opt


