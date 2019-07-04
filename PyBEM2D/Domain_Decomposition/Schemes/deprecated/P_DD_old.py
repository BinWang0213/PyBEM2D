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

def PDDo(obj,alpha,TOL,opt):
        """Dirichlet-Dirichlet iterative loop
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
        P_old_old=obj.new_var()#q^k-1 for current side
        Q_cur_old=obj.new_var()#h^k-1 for current side
        Q_con_old=obj.new_var()#h^k-1 for connect side
        AB_mat = []  # BEM Matrix for each domain


        MaxIter=100
        for it in range(MaxIter):
            if(debug2): print('----Loop:',it+1)
            error_final=0.0
            error=[]

            if(it>2):
                if(opt==1):
                    alpha_opt=PDD_OPT(obj,P_old_old,Q_cur_old,Q_con_old,alpha)
                    alpha=alpha_opt
                    #print(alpha_opt)
            
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
                        P_old=np.zeros(len(obj.BEMobjs[i].Mesh.mesh_nodes[bdID])) #initial guess zero vector
                        if(obj.BEMobjs[ConnectObjID].TypeE_edge=="Const"):
                            P_old=np.zeros(len(obj.BEMobjs[i].Mesh.mesh_nodes[bdID])-1) #Const element number = nodes number -1
                    if(it>0):
                        P_old=PQ[0]
                    Q_current=PQ[1]
                    
                    Q_connect=obj.Interp_intersection(i,ConnectObjID,Intersect,varID=1)#(Current,Connect,Intersect)
                    if(debug2): print('Q_Current',Q_current,'Q_Connect',Q_connect)
                    
                    #print(it+1,'alpha',alpha)
                    P_new=P_old-alpha*(Q_current+Q_connect)
                    if(debug2): print('p_new',P_new,'p_old',P_old)
                    
                    #Update new Dirichlet into system
                    bc_Dirichlet=[(bdID,P_new)]
                    obj.BEMobjs[i].set_BoundaryCondition(DirichletBC=bc_Dirichlet,update=1,mode=1,debug=0)
                    
                    if(it>0): error.append(max(abs(P_new-P_old))/max(abs(P_new)))
                    #print(abs(Q_new-Q_old),abs(Q_new))
                    
                    P_old_old[i][j]=P_old  #q_k-1 for current side
                    Q_cur_old[i][j]=Q_current#h_k-1 for current side
                    Q_con_old[i][j]=Q_connect#h_k-1 for current side
                    
            
            #Collect error for plot convergence
            if(it>0):
                error_final=max(error)
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

                    P_old=PQ[0]
                    Q_current=PQ[1]
                    
                    Q_connect=obj.Interp_intersection(i,ConnectObjID,Intersect,varID=1)#(Current,Connect,Intersect)

                    #for optimal relxation parameters
                    h_dif=P_old-P_old_old[i][j]
                    q_cur_dif=Q_current-Q_cur_old[i][j]
                    q_con_dif=Q_connect-Q_con_old[i][j]
                    q_ba=q_con_dif+q_cur_dif
                    #print("q_dif2",h_dif,q_ba)
                    #print('nom22',np.inner(h_dif,q_ba))
                    #print('dnom2',np.linalg.norm(-h_ba)**2)
                    nom+=np.inner(h_dif,q_ba)
                    denom+=np.linalg.norm(q_ba)**2
                    
                    
        alpha_opt=nom/denom
        if(alpha_opt<0.0):#Special case: P-DD may have negative alpha
            alpha_opt=alpha_old#Use the alpha from the last step
            print("Warning! Negative alpha!")
        
        #Test of bounded case
        #if(alpha_opt>1.0):
        #    alpha_opt=1.0
            #alpha_opt=5.0
        #print('!!!',-nom,denom,alpha_opt)
        return alpha_opt
