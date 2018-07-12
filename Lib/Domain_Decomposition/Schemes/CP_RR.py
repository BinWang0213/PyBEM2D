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
    
def CPRR(obj,alpha,TOL,opt):
        """Classic Robin-Robin iterative loop
           Reference: Section 3.3 in the reference paper
        ------------------------
        |  Current | Connected |
        |   Domain |   Domain  |
        ------------------------
             Intersection
        
        Non-conforming mesh are supported
        Intersection may have different nodes on two domain
        
        r=q+alpha*h
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
        R_old_old=obj.new_var()#r^k-1 for current side
        P_cur_old=obj.new_var()#h^k-1 for current side
        P_con_old=obj.new_var()#h^k-1 for connect side
        Q_cur_old=obj.new_var()#h^k-1 for current side
        Q_con_old=obj.new_var()#h^k-1 for connect side

        MaxIter=15
        for it in range(MaxIter):
            if(debug2): print('----Loop:',it+1)
            error_final=0.0
            error=[]

            if(it>2):
                if(opt==1):
                    alpha_opt=CPRR_OPT(obj,R_old_old,P_cur_old,P_con_old,Q_cur_old,Q_con_old,alpha)
                    alpha=alpha_opt
                    #print(alpha_opt)
            alpha=1.9
            #Step1. Prepare and update BCs for all domains
            for i in range(obj.NumObj):#For each subdomain
                Num_shared_edge=len(obj.Connect[i])
                #CurrentBEM=obj.BEMobjs[i]
                for j in range(Num_shared_edge):#For each connected edge in this domain
                    ConnectObjID=obj.Connect[i][j][0]
                    IntersectID=obj.Connect[i][j][1]
                    Intersect=obj.Intersects[IntersectID]
                    if(debug1): print('Subdomain:',i,'ConnectDomain:',ConnectObjID,'Intersection Coords:',Intersect)
                    
                    #Non-conforming mesh- Interpolating the current nodes on connected domain
                    #Local bdID is determined using intersection coordinates
                    bdID=obj.BEMobjs[i].Mesh.EndPoint2bdmarker(Intersect[0],Intersect[1])
                    PQ=obj.BEMobjs[i].PostProcess.get_BDSolution(bdID)

                    if(it==0):
                        R_old=np.zeros(len(obj.BEMobjs[i].Mesh.mesh_nodes[bdID])) #initial guess must be 0
                        if(obj.BEMobjs[ConnectObjID].TypeE_edge=="Const"):
                            R_old=np.zeros(len(obj.BEMobjs[i].Mesh.mesh_nodes[bdID])-1) #Const element number = nodes number -1
                        #Update new Neumann BC into system
                        bc_Robin=[(bdID,R_old)]
                        obj.BEMobjs[i].set_BoundaryCondition(RobinBC=bc_Robin,update=1,mode=1)
                    if(it>0):
                        R_old=PQ[1]+PQ[0]

                        P_current=PQ[0]
                        P_connect=obj.Interp_intersection(i,ConnectObjID,Intersect)#(Current,Connect,Intersect)
                        Q_current=PQ[1]
                        Q_connect=obj.Interp_intersection(i,ConnectObjID,Intersect,varID=1)
                        
                        if(debug2): print('PCurrent',P_current,'PConnect',P_connect)
                        if(debug2): print('QCurrent',Q_current,'QConnect',Q_connect)

                    
                        R_new=-Q_connect+alpha*P_connect
                        if(debug2): print('r_new',R_new,'r_old',R_old)

                        #Update new Neumann BC into system
                        bc_Robin=[(bdID,R_new)]
                        obj.BEMobjs[i].set_BoundaryCondition(RobinBC=bc_Robin,update=1,mode=1)
                    
                        error.append(max(abs(R_new-R_old))/max(abs(R_new)))
                    #print(abs(R_new-R_old),abs(R_new))
                    
                        R_old_old[i][j]=R_old  #q_k-1 for current side 
                        P_cur_old[i][j]=P_current#h_k-1 for current side
                        P_con_old[i][j]=P_connect#h_k-1 for current side
                        Q_cur_old[i][j]=Q_current#h_k-1 for current side
                        Q_con_old[i][j]=Q_connect#h_k-1 for current side
                    
            #Collect error for plot convergence
            if(it>0):
                error_final=max(error)
                print('%s\t%s\t\talpha:\t%s'%(it,error_final,alpha))
                obj.error_abs.append(error_final)
            
            #Step2. Update the solution for all fractures
            for i in range(obj.NumObj):#For each subdomain
                obj.BEMobjs[i].DFN=0
                #obj.BEMobjs[i].print_debug()
                obj.BEMobjs[i].Solve()
            
            if(it>5 and error_final<TOL):
                print('Converged at',it,'Steps! TOL=',TOL)
                print("Dirichelt",P_current)
                print("Robin",R_new)
                break
        obj.plot_Convergence()


def CPRR_OPT(obj,R_old_old,P_cur_old,P_con_old,Q_cur_old,Q_con_old,alpha_old):
        #Calculate the optimal relxation paramters based on error function J
        #Equation 17 in the Reference Paper

        nom=0.0
        denom=0.0

        for i in range(obj.NumObj):#For each subdomain
                Num_shared_edge=len(obj.Connect[i])
                #CurrentBEM=obj.BEMobjs[i]
                for j in range(Num_shared_edge):#For each connected edge in this domain
                    ConnectObjID=obj.Connect[i][j][0]
                    IntersectID=obj.Connect[i][j][1]
                    Intersect=obj.Intersects[IntersectID]
                    
                    #Non-conforming mesh- Interpolating the current nodes on connected domain
                    #Local bdID is determined using intersection coordinates
                    bdID=obj.BEMobjs[i].Mesh.EndPoint2bdmarker(Intersect[0],Intersect[1])
                    PQ=obj.BEMobjs[i].PostProcess.get_BDSolution(bdID)

                    R_old=PQ[1]+PQ[0]

                    P_current=PQ[0]
                    P_connect=obj.Interp_intersection(i,ConnectObjID,Intersect)#(Current,Connect,Intersect)
                    Q_current=PQ[1]
                    Q_connect=obj.Interp_intersection(i,ConnectObjID,Intersect,varID=1)
                    
                    #for optimal relxation parameters
                    #r_dif=R_old-R_old_old[i][j]

                    h_cur_dif=P_current-P_cur_old[i][j]
                    h_con_dif=P_connect-P_con_old[i][j]
                    #h_ba=h_cur_dif-h_con_dif
                    q_cur_dif=Q_current-Q_cur_old[i][j]
                    q_con_dif=Q_connect-Q_con_old[i][j]
                    #q_ba=q_con_dif+q_cur_dif

                    #hq_ba=h_ba+q_ba
                    #print("q_dif2",q_dif,h_ba)
                    #print('nom2',np.inner(q_dif,h_ba))
                    #print('dnom2',np.linalg.norm(h_ba)**2)
                    nom+=np.inner(q_cur_dif,h_cur_dif)
                    denom+=np.linalg.norm(h_cur_dif)**2
                    #print(nom,denom)
                    
                    
        alpha_opt=nom/denom
        if(alpha_opt<0.0):#Special case: P-DD may have negative alpha
            #alpha_opt=alpha_old#Use the alpha from the last step
            print("Warning! Negative alpha!")

        #print('!!!',-nom,denom,alpha_opt)
        return alpha_opt
