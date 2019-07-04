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
#  Sequential Dirichlet-Neumann Method
#
#######################################
    
def SDN(obj,alpha,TOL,opt):
        """Dirichlet-Neumann iterative loop
           Reference: Section 3.4 in the reference paper
        ------------------------
        |  Current | Connected |
        |   Domain |   Domain  |
        ------------------------
             Intersection
        
        Non-conforming mesh are supported
        Intersection may have different nodes on two domain
        
        Update pressure(h) in k+1 steps:
            left:  h_k+1=h_k-alpha*(h_left_k-h_right_k)
            right: q_right=-q_left->h_right
        
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
        P_cur_old=obj.new_var()#h^k-1 for current side
        P_con_old=obj.new_var()#h^k-1 for connect side
        AB_mat = []  # BEM Matrix for each domain

        MaxIter=50
        for it in range(MaxIter):
            if(debug2): print('----Loop:',it+1)
            error_final=0.0
            error=[]

            if(it>2):
                if(opt==1):
                    alpha_opt=SDN_OPT(obj,P_cur_old,P_con_old)
                    alpha=alpha_opt
            
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
                    #bdID_connect=obj.BEMobjs[ConnectObjID].EndPoint2bdmarker(Intersect[0],Intersect[1])
                    PQ=obj.BEMobjs[i].PostProcess.get_BDSolution(bdID)
                    #PQ_connect=obj.BEMobjs[ConnectObjID].get_BDSolution(bdID_connect)

                    if(it==0):#initial guess must be 0
                        P_new=np.zeros(len(obj.BEMobjs[i].Mesh.mesh_nodes[bdID]))
                        Q_new=np.zeros(len(obj.BEMobjs[i].Mesh.mesh_nodes[bdID]))
                        if(obj.BEMobjs[ConnectObjID].TypeE_edge=="Const"):
                            P_new=np.zeros(len(obj.BEMobjs[i].Mesh.mesh_nodes[bdID])-1)
                            Q_new=np.zeros(len(obj.BEMobjs[i].Mesh.mesh_nodes[bdID])-1)

                    if(it>0): #real iteration loop
                        P_old=PQ[0]
                        P_current=PQ[0]
                    
                        P_connect=obj.Interp_intersection(i,ConnectObjID,Intersect)#(CurrentID,ConnectID,Intersect)
                        if(i<ConnectObjID): 
                            if(debug2): print('P_Current',P_current,'P_Connect',P_connect)
                
                        P_new=P_old-alpha*(P_current-P_connect)
                        if(i<ConnectObjID): 
                            if(debug2): print('P_new',P_new,'alpha',alpha,'P_old',P_old)
                        
                        Q_connect=obj.Interp_intersection(i,ConnectObjID,Intersect,varID=1)
                        if(i>ConnectObjID): 
                            if(debug2): print('Q_Connect',Q_connect)
                        Q_new=-Q_connect
                        if(i>ConnectObjID): 
                            if(debug2): print('Q_new',Q_new)

                    #Update new Neumann BC into system-right
                    if(i>ConnectObjID):#this is right side domain
                        bc_Neumann=[(bdID,Q_new)]
                        obj.BEMobjs[i].set_BoundaryCondition(NeumannBC=bc_Neumann,update=1,mode=1,debug=0)

                        if(it == 0):  # Store the intial BEM Matrix
                            AB_mat.append(obj.BEMobjs[i].Solve())
                        else:  # Update solution by only update the boundary condition, Fast
                            AB_mat[i] = obj.BEMobjs[i].Solve(DDM=1, AB=[AB_mat[i][0], AB_mat[i][2]],debug=0)
                    
                    #Update new Dirichelt BC into system-left
                    if(i<ConnectObjID):#this is left side domain
                        bc_Dirichlet=[(bdID,P_new)]
                        obj.BEMobjs[i].set_BoundaryCondition(DirichletBC=bc_Dirichlet,update=1,mode=1,debug=0)
                        
                        #Sequential Scheme with dynamic alpha (Lin, 1996, An interative finite element-boundary element algorithm)
                        if(it == 0 and len(AB_mat)==0):  # Store the intial BEM Matrix
                            AB_mat.append(obj.BEMobjs[i].Solve())
                        else:  # Update solution by only update the boundary condition, Fast
                            AB_mat[i] = obj.BEMobjs[i].Solve(DDM=1, AB=[AB_mat[i][0], AB_mat[i][2]],debug=0)

                        if(it>0): 
                            error.append(max(abs(P_new-P_old))/max(abs(P_new)))
                            #print(abs(Q_new-Q_old),abs(Q_new))
                    
                            P_cur_old[i][j]=P_current#h_k-1 for current side
                            P_con_old[i][j]=P_connect#h_k-1 for connect side
                    
            
            #Collect error for plot convergence
            if(it>0):
                error_final=max(error)
                print('%s\t%s\t\talpha:\t%s'%(it,error_final,alpha))
                obj.error_abs.append(error_final)
            
            #Step2. Update the solution for all fractures
            #for i in range(obj.NumObj):#For each subdomain
            #    obj.BEMobjs[i].DFN=0
            #    obj.BEMobjs[i].Solve()
            
            if(it>5 and error_final<TOL):
                print('Converged at',it,'Steps! TOL=',TOL)
                print("Dirichelt",P_new)
                print("Neumann",Q_new)
                break
        obj.plot_Convergence()

def SDN_OPT(obj, P_cur_old,P_con_old):
        #Calculate the optimal relxation paramters based on error function J
        #Equation 18 in the Reference Paper

        debug=0
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

                    P_current=PQ[0]
                    
                    P_connect=obj.Interp_intersection(i,ConnectObjID,Intersect)#(Current,Connect,Intersect)
                    
                    if(i<ConnectObjID):#this is left side domain

                    #for optimal relxation parameters
                        h_cur_dif=P_current-P_cur_old[i][j]
                        h_con_dif=P_connect-P_con_old[i][j]
                        h_ba=h_cur_dif-h_con_dif
                        if(debug):
                            print("P_cur,P_cur_k-1",P_current,P_cur_old[i][j])
                            print("P_con,P_con_k-1",P_connect,P_con_old[i][j])
                            #print('h_dif(cur,con)',h_cur_dif,h_con_dif)
                            print('h_cur_dif',h_cur_dif,'h_ba',h_ba)
                        
                            print('nom1',np.inner(h_cur_dif,h_ba))
                            print('dnom1',np.linalg.norm(h_ba)**2)
                        
                        nom+=np.inner(h_cur_dif,h_ba)
                        denom+=np.linalg.norm(h_ba)**2

                        if(debug): print('nom',nom,'denom',denom,'alpha',-nom/denom)
                    
                    
        alpha_opt=nom/denom
        if(alpha_opt>2):
            alpha_opt=1
        if (alpha_opt<0.0):
            alpha_opt=0.1
        #Test of bounded case
        #if(alpha_opt > 1.0):
        #    alpha_opt = 1.0
        #print('!!!',-nom,denom,alpha_opt)
        return alpha_opt
    

