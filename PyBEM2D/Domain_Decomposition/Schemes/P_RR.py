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
#  Parallel Robin-Robin Method
#
#######################################
    
def PRR(obj,alpha,robin_a,TOL,max_iter,opt):
        """Robin-Robin iterative loop
           Reference: Section 3.3 in the reference paper
        ------------------------
        |  Current | Connected |
        |   Domain |   Domain  |
        ------------------------
             Intersection
        
        Non-conforming mesh are supported
        Intersection may have different nodes on two domain
        
        r=q+a*h
        Update flux(q) in k+1 steps:
            r_k+1=r_k+alpha*(h_left_k-h_right_k+a*(q_left_k+q_right_k))
            r_left=r_right=r_k+1
        
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
        NumInt = len(obj.Intersects)
        R_old_old = [[] for i in range(NumInt)]  # r^k-1 for current side
        R_conn_old_old = [[] for i in range(NumInt)]  # r^k-1 for connect side
        P_cur_old = [[] for i in range(NumInt)]  # h^k-1 for current side
        P_con_old = [[] for i in range(NumInt)]  # h^k-1 for connect side
        Q_cur_old = [[] for i in range(NumInt)]  # h^k-1 for current side
        Q_con_old = [[] for i in range(NumInt)]  # h^k-1 for connect side
        AB_mat = []  # BEM Matrix for each domain


        MaxIter=max_iter
        for it in range(MaxIter):
            if(debug2): print('----Loop:',it+1)
            error_final=0.0
            error=[]

            if(it>2 and opt==1):
                alpha_opt = PRR_OPT(obj, R_old_old, R_conn_old_old, P_cur_old,P_con_old, Q_cur_old, Q_con_old, alpha, robin_a)
                alpha=alpha_opt
                #print(alpha_opt)
            #alpha=0.1
            
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
                    R_old=np.zeros(EdgeDof)
                    R_old_connect=np.zeros(EdgeDof)
                    R_new = np.zeros(EdgeDof)
                    R_new_connect=np.zeros(EdgeDof)
                    P_current = np.zeros(EdgeDof)
                    P_connect = np.zeros(EdgeDof)
                    Q_current = np.zeros(EdgeDof)
                    Q_connect = np.zeros(EdgeDof)

                #Normal iterations   
                elif(it>0):
                    PQ = obj.BEMobjs[DomainID].PostProcess.get_BDSolution(EdgeID)
                    PQ_connect = obj.BEMobjs[DomainID_connect].PostProcess.get_BDSolution(EdgeID_connect)

                    P_current,Q_current = PQ[0],PQ[1]
                    P_connect, Q_connect = PQ_connect[0], PQ_connect[1]
                    
                    #the dof on the other side is reversed
                    R_old = P_current + Q_current
                    R_old_connect=P_connect + Q_connect
                    if(debug2): print('P_Current',P_current,'P_Connect',P_connect)
                    if(debug2): print('Q_Current',Q_current,'Q_Connect',Q_connect)
                    if(debug2): print('R_Current',R_old,'R_Connect',R_old_connect)
                
                    #print(it+1,'alpha',alpha)

                    #the dof on the other side is reversed
                    if(BDType=='Edge'):
                        R_new=R_old-alpha*((P_current-np.flip(P_connect))+robin_a*(Q_current+np.flip(Q_connect)))
                        R_new_connect=R_old_connect-alpha*((-np.flip(P_current)+P_connect)+robin_a*(np.flip(Q_current)+Q_connect))
                    else:
                        R_new=R_old-alpha*((P_current-P_connect)+robin_a*(Q_current+Q_connect))
                        R_new_connect=R_old_connect-alpha*((-P_current+P_connect)+robin_a*(Q_current+Q_connect)) #oppsite dP for connect domain
                    if(debug2): print('R_new',R_new,'R_old',R_old,(P_current-P_connect)+robin_a*(Q_current+Q_connect))

                    if max(abs(R_new)) > 0:
                        error.append(max(abs(R_new - R_old)) / max(abs(R_new)))
                    if max(abs(R_new_connect)) > 0:
                        error.append(max(abs(R_new_connect - R_old_connect)) / max(abs(R_new_connect)))
                    else:
                        error.append(1)

                
                #Update new Dirichlet into system
                bc_Robin = [(EdgeID, R_new)]
                obj.BEMobjs[DomainID].set_BoundaryCondition(RobinBC=bc_Robin,update=1,mode=1,Robin_a=robin_a,debug=0)
                bc_Robin = [(EdgeID_connect, R_new_connect)]
                obj.BEMobjs[DomainID_connect].set_BoundaryCondition(RobinBC=bc_Robin,update=1,mode=1,Robin_a=robin_a,debug=0)
            
                
                #Save last time iteration info
                R_old_old[IntID] = R_old  #q_k-1 for current side 
                R_conn_old_old[IntID]=R_old_connect #q_k-1 for connect side 
                P_cur_old[IntID] = P_current  # h_k-1 for current side
                P_con_old[IntID] = P_connect  # h_k-1 for current side
                Q_cur_old[IntID] = Q_current  # h_k-1 for current side
                Q_con_old[IntID] = Q_connect  # h_k-1 for current side

                    
            #Collect error for plot convergence
            if(it>0):
                error_final=max(error)
                if(it%(MaxIter/50)==0):
                    print('%s\t%s\t\talpha:\t%s'%(it,error_final,alpha))
                obj.error_abs.append(error_final)
            
            #Step2. Update the solution for all fractures
            for i in range(obj.NumObj):#For each subdomain
                if(it == 0):  # Store the intial BEM Matrix
                    AB_mat.append(obj.BEMobjs[i].Solve(debug=0))
                else:  # Update solution by only update the boundary condition, Fast
                    AB_mat[i] = obj.BEMobjs[i].Solve(DDM=1, AB=[AB_mat[i][0], AB_mat[i][2]],debug=0)
            
            if(it>5 and error_final<TOL):
                print('Converged at',it,'Steps! TOL=',TOL)
                print("Dirichelt",P_current)
                print("Robin",R_new)
                break
        obj.plot_Convergence()


def PRR_OPT(obj, R_old_old, R_conn_old_old,P_cur_old, P_con_old, Q_cur_old, Q_con_old, alpha_old, robin_a):
        #Calculate the optimal relxation paramters based on error function J
        #Equation 17 in the Reference Paper
        debug1=0
        nom = 0.0
        denom = 0.0

        NumInt = len(obj.Intersects)
        for IntID in range(NumInt):  # For each subdomain
            DomainID, DomainID_connect = obj.Intersects[IntID][0], obj.Intersects[IntID][1]
            EdgeID, EdgeID_connect = obj.Intersects[IntID][2], obj.Intersects[IntID][3]
            BDType=obj.BEMobjs[DomainID].Mesh.getBDType(EdgeID)

            if(debug1):
                print('Intersection',IntID,'Domain(%s->%s)'%(DomainID,DomainID_connect),'BD id(%s->%s)'%(EdgeID,EdgeID_connect))
            
            PQ = obj.BEMobjs[DomainID].PostProcess.get_BDSolution(EdgeID)
            PQ_connect = obj.BEMobjs[DomainID_connect].PostProcess.get_BDSolution(EdgeID_connect)

            P_current, Q_current = PQ[0], PQ[1]
            P_connect, Q_connect = PQ_connect[0], PQ_connect[1]
            R_old = P_current + robin_a*Q_current
            R_old_connect = P_connect + robin_a * Q_connect

            #print(Q_current, Q_cur_old[IntID])
            #print(P_current, P_cur_old[IntID])
            #print(P_connect, P_con_old[IntID])
            #print(Q_connect, Q_con_old[IntID])
            #print(P_current,P_connect)

            #alpha current side
            h_cur_dif = P_current - P_cur_old[IntID]
            h_con_dif = P_connect - P_con_old[IntID]
            if(BDType=='Edge'): h_ba = h_cur_dif - np.flip(h_con_dif)
            else: h_ba = h_cur_dif - h_con_dif
            q_cur_dif = Q_current - Q_cur_old[IntID]
            q_con_dif = Q_connect - Q_con_old[IntID]
            if(BDType=='Edge'): q_ba = np.flip(q_con_dif) + q_cur_dif
            else: q_ba = q_con_dif + q_cur_dif

            hq_ba = h_ba + robin_a * q_ba
            r_dif = R_old - R_old_old[IntID]

            #print("q_dif", r_dif,q_ba, h_ba, hq_ba,np.sum(r_dif+q_ba+h_ba+hq_ba))

            nom = nom + np.inner(r_dif, hq_ba)
            denom = denom + np.linalg.norm(hq_ba)**2
            #print(np.inner(r_dif, hq_ba), np.linalg.norm(hq_ba)**2)

            
            #alpha connect side
            if(BDType=='Edge'): h_ab = h_con_dif - np.flip(h_cur_dif)
            else: h_ab = h_con_dif - h_cur_dif
            if(BDType=='Edge'): q_ab= np.flip(q_ba)
            else: q_ab=q_ba
            hq_ab = h_ab + robin_a * q_ab
            r_con_dif = R_old_connect-R_conn_old_old[IntID]

            #print("q_dif2", r_con_dif, q_ba, h_ab, hq_ab,np.sum(r_con_dif+q_ba+h_ab+hq_ab))
            nom = nom + np.inner(r_con_dif, hq_ab)
            denom = denom + np.linalg.norm(hq_ab)**2
            #print(np.inner(r_con_dif, hq_ab),np.linalg.norm(hq_ab)**2)
            
            

        alpha_opt = nom / denom
        if(alpha_opt < 0.0):  # Special case: P-DD may have negative alpha
            #alpha_opt=alpha_old#Use the alpha from the last step
            print("Warning! Negative alpha!")

        #Test of bounded case
        #if(alpha_opt > 1.0):
        #    alpha_opt = 1.0
        #print('!!!',-nom,denom,alpha_opt)
        return alpha_opt
