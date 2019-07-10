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
#  Parallel Dirichlet-Dirichlet Method
#
#######################################

def PDD(obj,alpha,TOL,max_iter,opt):
        """Dirichlet-Dirichlet iterative loop
           Boundary Conditions are updated by looping through intersections

           Reference: Section 3.1 in the reference paper
        ------------------------
        |  Current | Connected |
        |   Domain |   Domain  |
        ------------------------
             Intersection
        
        Non-conforming mesh are supported
        Intersection may have different nodes on two domain
        
        Update flux(q) in k+1 steps:
            h_k+1=h_k+alpha*(q_left_k+q_right_k)
            h_left=h_right=h_k+1
        
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
        P_old_old = [[] for i in range(NumInt)]  # q^k-1 for current side
        Q_cur_old = [[] for i in range(NumInt)]  # h^k-1 for current side
        Q_con_old = [[] for i in range(NumInt)]  # h^k-1 for connect side
        AB_mat = []  # BEM Matrix for each domain


        MaxIter = max_iter
        for it in range(MaxIter):
            if(debug2): print('----Loop:',it+1)
            error_final=0.0
            error=[]

            if(it>2 and opt==1):
                alpha_opt=PDD_OPT(obj,P_old_old,Q_cur_old,Q_con_old,alpha)
                alpha=alpha_opt
                #print(alpha_opt)
            
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
                    P_old=np.zeros(EdgeDof)
                    P_new = P_old
                    P_new_connect=P_old
                    Q_current=P_old
                    Q_connect=P_old

                #Normal iterations
                elif(it>0):
                    PQ = obj.BEMobjs[DomainID].PostProcess.get_BDSolution(EdgeID)
                    PQ_connect = obj.BEMobjs[DomainID_connect].PostProcess.get_BDSolution(EdgeID_connect)
                    h_current=obj.BEMobjs[DomainID].h
                    h_connect=obj.BEMobjs[DomainID_connect].h

                    #print('Orig',PQ_connect[0],PQ_connect[1])
                    #print('New',np.flip(PQ_connect[0]),np.flip(PQ_connect[1]))
                    #PQ_connect=[np.flip(PQ_connect[0]),np.flip(PQ_connect[1])]

                    P_old=PQ[0]
                    Q_current = PQ[1]                   
                    if(BDType=='Edge'):  Q_connect=np.flip(PQ_connect[1]) #the dof on the other side is reversed
                    else: Q_connect = PQ_connect[1]
                    if(debug2): print('Q_Current',Q_current,'Q_Connect',Q_connect)
                    
                    #* Consider thickness variation
                    Q_current*=h_current 
                    Q_connect*=h_connect

                    #print(it+1,'alpha',alpha)
                    #* Key iteration equation
                    P_new=P_old-alpha*(Q_current+Q_connect)
                    if(debug2): print('p_new',P_new,'p_old',P_old)

                    if(BDType=='Edge'): P_new_connect=np.flip(P_new)
                    else: P_new_connect=P_new
                    if(debug2): print('p_new_connect',P_new_connect,'p_old',P_old)

                    if max(abs(P_new)) > 0:
                        error.append(max(abs(P_new - P_old)) / max(abs(P_new)))
                    else:
                        error.append(1)
                        #print(abs(Q_new-Q_old),abs(Q_new))
                
                #Update new Dirichlet into system
                bc_Dirichlet = [(EdgeID, P_new)]
                obj.BEMobjs[DomainID].set_BoundaryCondition(DirichletBC=bc_Dirichlet,update=1,mode=1,debug=0)
                bc_Dirichlet = [(EdgeID_connect, P_new_connect)]
                obj.BEMobjs[DomainID_connect].set_BoundaryCondition(DirichletBC=bc_Dirichlet,update=1,mode=1,debug=0)
                
                #Save last time iteration info
                P_old_old[IntID] = P_old  #q_k-1 for current side
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
                    AB_mat.append(obj.BEMobjs[i].Solve())
                else:  # Update solution by only update the boundary condition, Fast
                    AB_mat[i] = obj.BEMobjs[i].Solve(DDM=1, AB=[AB_mat[i][0], AB_mat[i][2]],debug=0)
            
            if(it>5 and error_final<TOL):
                print('Converged at',it,'Steps! TOL=',TOL)
                print("Dirichelt",P_new)
                print("Neumann",Q_current)
                break
        obj.plot_Convergence()


def PDD_OPT(obj, P_old_old,Q_cur_old,Q_con_old,alpha_old):
        #Calculate the optimal relxation paramters based on error function J
        #Equation 15 in the Reference Paper
        
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
            h_current=obj.BEMobjs[DomainID].h
            h_connect=obj.BEMobjs[DomainID_connect].h

            #PQ_connect=[np.flip(PQ_connect[0]),np.flip(PQ_connect[1])]

            P_old = PQ[0]
            Q_current = PQ[1]
            #Q_connect = PQ_connect[1]
            #Q_connect=np.flip(PQ_connect[1])
            if(BDType=='Edge'):  Q_connect=np.flip(PQ_connect[1]) #the dof on the other side is reversed
            else: Q_connect = PQ_connect[1]

            #* Consider thickness variation
            Q_current*=h_current
            Q_connect*=h_connect

            #for optimal relxation parameters
            h_dif = P_old - P_old_old[IntID]
            q_cur_dif = Q_current - Q_cur_old[IntID]
            q_con_dif = Q_connect - Q_con_old[IntID]
            q_ba=q_con_dif+q_cur_dif
            #print("q_dif2",h_dif,q_ba)
            #print('nom22',np.inner(h_dif,q_ba))
            #print('dnom2',np.linalg.norm(-h_ba)**2)
            nom+=np.inner(h_dif,q_ba)
            denom+=np.linalg.norm(q_ba)**2
                    
                    
        alpha_opt=nom/denom
        #alpha_opt = alpha_old*nom / denom
        if(alpha_opt<0.0):#Special case: P-DD may have negative alpha
            alpha_opt=alpha_old#Use the alpha from the last step
            print("Warning! Negative alpha!")
        
        #Test of bounded case
        #if(alpha_opt>1.0):
        #    alpha_opt=1.0
            #alpha_opt=5.0
        #print('!!!',-nom,denom,alpha_opt)
        return alpha_opt
