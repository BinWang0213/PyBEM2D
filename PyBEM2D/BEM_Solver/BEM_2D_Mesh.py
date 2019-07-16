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
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import path

#[BEM Mesh]
from .Elements.BEM_Elements import BEM_element,Source_element

#[General Geometry Lib]
from ..Tools.Geometry import *
from .BEM_2D_Postprocessing import *


#Publication quality figure paramters
_params = {'font.family': 'sans-serif',
           'font.serif': ['Times', 'Computer Modern Roman'],
           'font.sans-serif': ['Helvetica', 'Arial',
                               'Computer Modern Sans serif'],
           'font.size': 14,

           'axes.labelsize': 14,
           'axes.linewidth': 1,

            
           'savefig.dpi': 300,
           'savefig.format': 'eps',
           # 'savefig.bbox': 'tight',
           # this will crop white spaces around images that will make
           # width/height no longer the same as the specified one.

           'legend.fontsize': 14,
           'legend.frameon': False,
           'legend.numpoints': 1,
           'legend.handlelength': 2,
           'legend.scatterpoints': 1,
           'legend.labelspacing': 0.5,
           'legend.markerscale': 0.9,
           'legend.handletextpad': 0.5,  # pad between handle and text
           'legend.borderaxespad': 0.5,  # pad between legend and axes
           'legend.borderpad': 0.5,  # pad between legend and legend content
           'legend.columnspacing': 1,  # pad between each legend column

           'xtick.labelsize': 14,
           'ytick.labelsize': 14,
           'xtick.direction':'in',
           'ytick.direction':'in',
           
           'lines.linewidth': 1,
           'lines.markersize': 7,
           # 'lines.markeredgewidth' : 0,
           # 0 will make line-type markers, such as '+', 'x', invisible
           }

###############################
#
#  Core BEM 2D Class
#
###############################

class BEM_2DMesh:
    """Contains information and functions related to a 2D potential problem using BEM."""
        
    def __init__(self,BEMobj):
        """Creates a BEM objecti with some specific paramters
        
        Arguments
        ---------
        [Discretization]
        Pts_e     -- Boundary end nodes for abritary shape Polygon. 
                     e.g. Boundary_vert=[(0.0, 0.0), (0.0, 1.5), (1.0, 1.5), (1.5, 0.0)]
        Pts_t     -- Internal trace end node for abritary intersection 
                     e.g. Trace_vert=[((0.25, 0.5), (1.25, 0.5)),((0.25, 1.2), (1.25, 1.2))]
        Pts_s     -- nodes for point sources
                     e.g. Source_vert=[(0.5,0.2),(0.3,0.1)]
        
        Ne_edge     -- Number of element on all edges
        Ne_trace    -- Number of element on all traces
        Nedof_edge  -- Number of DOF for edge elements (const=1 linear=2 quad=3)
        Nedof_trace -- Number of DOF for trace elements
        Ndof_edge   -- Total number of DOF for all edge elements
        Ndof_trace  -- Total number of DOF for all trace elements
        Ndof        -- Total number of DOF for all elements 

        domain_min      -- minimum coords of a domain 
        domain_max      -- maxmum coords of a domain

        h_edge    -- Length of element for boundary edge [optional]
        h_trace   -- Length of element for trace [optional]
        Num_boundary -- Number of boundary edges
        Num_trace    -- Number of traces
        Num_sources  -- Number of sources

        [BEM Solver]
        TraceOn         -- Enable the internal (geometry) trace
        BEs_edge        -- BEM element collection for boundary edge
        BEs_trace       -- BEM element collection for internal edge
        NumE_bd         -- the number of element on each boundary
        NumE_t          -- the number of element on each trace

        
        Author:Bin Wang(binwang.0213@gmail.com)
        Date: July. 2017
        """
        self.tol=1e-7 #!Geometry calculation tolerance
        self.BEMobj=BEMobj
        self.Pts_e=[]
        self.Pts_t=[]
        self.Pts_s=[]
        
        self.domain_min=[] #Additional Geometory variable
        self.domain_max=[]
        
        self.Ne=0
        self.Ne_edge=0
        self.Ne_trace=0
        self.Nedof_edge=0
        self.Nedof_trace=0
        self.Ndof_edge=0
        self.Ndof_trace=0
        self.Ndof_source=0
        self.Ndof=0
        self.h_edge=0
        self.h_trace=0
        
        self.Num_boundary=1
        self.Num_trace=1
        self.Num_source=1
        self.EdgeStartID=0
        self.TraceStartID=0
        self.SourceStartID=0
        
        self.NumE_bd=[]
        self.NumE_t=[]

        #[BEM Mesh Info]
        self.mesh_nodes=[] #All of bem nodes for each edge
     
    def set_Mesh(self,Pts_e=[],Pts_t=[],Pts_s=[],
                      h_edge=0.1,h_trace=0.1,
                      Ne_edge=None,Ne_trace=None,
                      Type='Quad',mode=0,geo_check=True):
        """Create BEM mesh based on either number of element or length of element
           Support for:
           1. Constant element,linear element and Quadratic element
           2. Abritary closed shape by giving boundary vertices(Pts_e)
           3. Internal line segments by giving internal vertices(Pts_t)
           4. Internal point sources by giving internal vertices(Pts_s)
        
        Arguments
        ---------
        Ne_edge   -- Number of element in all boundary edge [optional]
        Ne_trace  -- Number of element in trace [optional]
        h_edge    -- Length of element for boundary edge [optional]
        h_trace   -- Length of element for trace [optional]
        mode      -- 0-round up connect 1-manully set up 
        geo_check -- check if trace or edges are intersected
        
        Author:Bin Wang(binwang.0213@gmail.com)
        Date: July. 2017

        Updated: Bin Wang @ June. 2019, point source
        """
        #Check intersection segments
        if(geo_check): Pts_e, Pts_t = self.Split_ByIntersections(Pts_e, Pts_t)
        
        #Collect basic geometric info
        self.Pts_e=Pts_e
        self.Pts_t=Pts_t
        self.Pts_s=Pts_s
        self.Num_boundary=len(Pts_e)
        self.Num_trace=len(Pts_t)
        self.Num_source=len(Pts_s)
        self.EdgeStartID=0
        self.TraceStartID=self.Num_boundary
        self.SourceStartID=self.Num_boundary + self.Num_trace


        if(Ne_edge is not None):
            self.h_edge=self.NumEle2LenEle(Ne_edge)
        else:
            self.h_edge = h_edge
        if(Ne_trace is not None):
            self.h_trace = self.NumEle2LenEle(None,Ne_trace)
        else:
            self.h_trace=h_trace
                
        #find the domain min,max for plotting
        self.domain_min=(min(np.asarray(self.Pts_e)[:,0]),min(np.asarray(self.Pts_e)[:,1]))
        self.domain_max=(max(np.asarray(self.Pts_e)[:,0]),max(np.asarray(self.Pts_e)[:,1]))        
        self.tol*=abs(max(self.domain_max)-min(self.domain_min))

        #Continous boundary marker index
        bd_marker_id=0

        #Boundary mesh
        self.BEMobj.TypeE_edge=Type
        for i in range(self.Num_boundary):
            Node=self.Pts_e[i]
            if (i==self.Num_boundary-1):
                if(mode==0): Node_next=self.Pts_e[0] #round connect
                elif(mode==1): break#manully connect
            else: 
                Node_next=self.Pts_e[i+1]
            Ne_edge=int(np.ceil(calcDist(Node,Node_next)/self.h_edge))
            self.NumE_bd.append(Ne_edge)
            
            added_nodes=self.Append_Line(Node,Node_next,Ne_edge,self.BEMobj.BEs_edge,
                                         bd_marker=bd_marker_id,Type=Type)
            self.mesh_nodes.append(added_nodes)
            bd_marker_id+=1
        
        #Additional mesh info
        self.Ne_edge=len(self.BEMobj.BEs_edge)
        if(self.BEMobj.TypeE_edge=="Const"):
            self.Ndof_edge = 1 * self.Ne_edge
            self.Nedof_edge=1
        elif(self.BEMobj.TypeE_edge == "Linear"):
            self.Ndof_edge = 2 * self.Ne_edge
            self.Nedof_edge=2
        else:
            self.Ndof_edge = 3*self.Ne_edge
            self.Nedof_edge=3
        
        #Trace mesh
        #Type="Const"
        self.BEMobj.TypeE_trace = Type
        if (self.h_trace != None and len(Pts_t)!=0):
            self.TraceOn=1 #
            #print('We have trace')
            for i in range(self.Num_trace):
                Node,Node_next=self.Pts_t[i][0],self.Pts_t[i][1]
                Ne_trace=int(np.ceil(calcDist(Node,Node_next)/self.h_trace))
                self.NumE_t.append(Ne_trace)
                
                temp_trace=[]
                added_nodes=self.Append_Line(Node,Node_next,Ne_trace,temp_trace,
                                             bd_marker=bd_marker_id, 
                                             Type=Type)#, refinement="cosspace")  # fracture always 0 flux on edge
                self.BEMobj.BEs_trace.append(temp_trace)
                self.mesh_nodes.append(added_nodes)
            
                bd_marker_id+=1
        
        #Source Point Mesh
        if(len(Pts_s)>0):
            for i in range(self.Num_source):
                Node=self.Pts_s[i]
                Source_Ele=Source_element(Node,bd_marker=bd_marker_id)
                self.BEMobj.BEs_source.append(Source_Ele)

                bd_marker_id+=1
        
        #Additional mesh info
        self.Ne_trace=sum(self.NumE_t)
        if(self.BEMobj.TypeE_trace == "Const"):
            self.Ndof_trace = 1 * self.Ne_trace
            self.Nedof_trace = 1
        elif(self.BEMobj.TypeE_trace == "Linear"):
            self.Ndof_trace = 2 * self.Ne_trace
            self.Nedof_trace = 2
        else:
            self.Ndof_trace = 3 * self.Ne_trace
            self.Nedof_trace = 3
        
        self.Ndof_source+=self.Num_source

        #Total #Elements
        self.Ne=self.Ne_edge+self.Ne_trace+self.Num_source

        #Total DOF
        self.Ndof=self.Ndof_edge+self.Ndof_trace+self.Ndof_source

        if(self.Ndof_trace == 0):
            self.TraceOn = 0
        else:
            self.TraceOn = 1

        #Plot Mesh
        print("[Mesh] Genetrated...")
        print("[Mesh] Discontinous Element used")
        print("[Mesh] Number of boundary elements:%s(Total) %s(Edge) %s(Trace)" % (self.Ne,self.Ne_edge,self.Ne_trace) )
        print("[Mesh] Number of Nodes:%s(Total) %s(Edge) %s(Trace) %s(Source)" % (self.Ndof,self.Ndof_edge,self.Ndof_trace,self.Ndof_source) )
        #self.plot_Mesh()

    def print_debug(self):
        """Check Meshing and B.C for different type of elements
        
        Author:Bin Wang(binwang.0213@gmail.com)
        Date: July. 2017
        """
        #Check Meshing and B.C for different type of elements
        print("[Mesh] Info")
        print("Number of boundary elements:%s E-T(%s,%s)" % (self.Ne_edge + self.Ne_trace,self.Ne_edge,self.Ne_trace) )
        print("Number of Nodes:%s E-T(%s-%s)" % (self.Ndof,self.Ndof_edge,self.Ndof_trace) )
        print("Edge Num.:%s" % (self.Num_boundary))
        print("# Neumann-1   Dirichlet-0")
        print("(E)Pts\tX\tY\tType\tMarker\t\tBC_type\t\tBC_value\tRobin")

        eleid = 0
        for i, pl in enumerate(self.BEMobj.BEs_edge):
            eleid = i + 1
            for j in range(self.Nedof_edge):
                nodeid = self.getNodeId(i, j, 'Edge') + 1
                xi,yi=pl.get_node(j)
                print("(%s)%s\t%5.3f\t%.3f\t%.4s\t%d\t\t%d\t\t%.2f\t\t%d" %
                      (eleid, nodeid, xi, yi, pl.element_type, pl.bd_marker, pl.bd_Indicator, pl.bd_values[0], pl.Robin_alpha))
        
        print("Trace Num.:%s" % (self.Num_trace))
        for i in range(self.Num_trace):
            print("--Trace ", i + 1)
            for j, pl in enumerate(self.BEMobj.BEs_trace[i]):
                eleid = eleid + 1
                for k in range(self.Nedof_trace):
                    nodeid = self.getNodeId(eleid, k, 'Trace') + 1
                    xi,yi=pl.get_node(j)
                    #print("(%s)%s\t%5.3f\t\t%.3f\t\t%.4s\t\t%d" % (index,2*len(self.BEMobj.BEs_edge)+index,pl.xa,pl.ya,pl.element_type,pl.bd_marker))
                    print("(%s)%s\t%5.3f\t%.3f\t%.4s\t%d\t\t%d\t\t%.2f\t\t%d" %
                          (eleid, nodeid, xi, yi, pl.element_type, pl.bd_marker, pl.bd_Indicator, pl.bd_value1, pl.Robin_alpha))

        
        print("Sources Num.:%s" % (self.Num_source))
        for i, pl in enumerate(self.BEMobj.BEs_source):
            eleid+=1
            nodeid = self.getNodeId(i,0,'Source')
            xi, yi = pl.get_node()
            print("(%s)%s\t%5.3f\t%.3f\t%.4s\t%d\t\t%d\t\t%.2f\t\t%d" %
                          (eleid, nodeid, xi, yi, 'Pts', pl.bd_marker, pl.bd_Indicator, pl.bd_values[0], -999))


    ###########Visulation Module################
    def plot_Mesh(self,Annotation=1,legend=1,node_size=5,scale=1.0,img_fname=None):
        """Plot BEM Mesh

        Author:Bin Wang(binwang.0213@gmail.com)
        Date: July. 2017
        """

        #Set paramters all at once
        for i in _params:
            rcParams[i]=_params[i]

        x_min, x_max = self.domain_min[0], self.domain_max[0]
        y_min, y_max = self.domain_min[1], self.domain_max[1]

        space = 0.1 * calcDist((x_min, y_min), (x_max, y_max))

        plt.figure(figsize=(5*scale, 4*scale))
        #plt.axes().set(xlim=[x_min - space, x_max + space],
        #               ylim=[y_min - space, y_max + space], aspect='equal')
        plt.axes().set(aspect='equal')


        #Domain boundary line
        plt.plot(*np.asarray(list(self.Pts_e)+[self.Pts_e[0]]).T,'k-',
                 lw=1.5, label="Domain Edges")
        #Trace boundary line
        for i in range(self.Num_trace):
            trace_line=[list(p) for p in self.Pts_t[i]]
            if(i==0):
                plt.plot(*np.asarray(trace_line).T, '-',color='gray',
                     lw=1.5, label="Trace Edges")
            else:
                plt.plot(*np.asarray(trace_line).T, 'k-',color='gray',
                     lw=1.5)

        #Point source
        #plt.scatter(*np.asarray(self.Pts_w).T,s=20,color='red')
        
        #Boundary elements
        BEs_pts=[]
        BEs_endpts=[]
        for BE in self.BEMobj.BEs_edge:
            BEs_endpts.append((BE.xa,BE.ya))
            for j in range(BE.ndof):
                BEs_pts.append(BE.get_node(j))
        plt.plot(*np.asarray(BEs_pts).T, 'bo', lw=1, markersize=node_size,
                 label=str(self.Ne_edge)+' Boundary Elements ')
        plt.scatter(*np.asarray(BEs_endpts).T, s=80,marker="x", c='b', alpha=0.8)

        #Trace elements
        if (self.TraceOn):
            BEs_pts = []
            BEs_endpts = []
            for t in self.BEMobj.BEs_trace:
                for BE_t in t:
                    BEs_endpts.append((BE_t.xa, BE_t.ya))
                    for j in range(BE.ndof):
                        BEs_pts.append(BE_t.get_node(j))
                BEs_endpts.append((t[-1].xb, t[-1].yb))
            
            plt.plot(*np.asarray(BEs_pts).T, 'go', lw=1, markersize=node_size,
                     label=str(self.Ne_trace) + ' Trace Elements ')
            plt.scatter(*np.asarray(BEs_endpts).T, s=80, marker="x", c='g', alpha=0.8,
            label='Mesh Nodes')

        #Source nodes
        if(len(self.Pts_s)>0):
            plt.plot(*np.asarray(self.Pts_s).T, 'ro', lw=1, markersize=5,
                     label=str(self.Num_source) + ' Source Nodes ')

        if (Annotation):
            #Show marker index-convenient for BC assignment
            index=0
            for i in range(self.Num_boundary):
                Node = self.Pts_e[i]
                if (i == self.Num_boundary - 1):
                    Node_next = self.Pts_e[0]  # round connect
                else:
                    Node_next = self.Pts_e[i + 1]
                rightmiddle = line_leftright(Node, Node_next, space * 0.8)[1]
                plt.text(*rightmiddle.T, "%s" % (index), fontsize=14)
                index+=1

            for i in range(self.Num_trace):
                Node, Node_next = self.Pts_t[i][0], self.Pts_t[i][1]

                rightmiddle = line_leftright(Node, Node_next, space * 0.3)[1]
                plt.text(*rightmiddle.T, "%s" % (index),color='gray', fontsize=14)
                index+=1
            
            for i in range(self.Num_source):
                Node= np.asarray(self.Pts_s[i])*1.03
                plt.text(*Node.T, "%s" % (index), color='r',fontsize=14)
                index+=1


        if(legend): plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title('BEM Mesh')
        plt.xlabel('x(m)')
        plt.ylabel('y(m)')

        #extend the margin
        plot_margin = space
        x0, x1, y0, y1 = plt.axis()
        plt.axis((x0 - plot_margin,
          x1 + plot_margin,
          y0 - plot_margin,
          y1 + plot_margin))

        #plt.tight_layout()
        #Give some margin for velocity field
        if(img_fname is not None): 
            #plt.tight_layout()
            plt.savefig(img_fname,dpi=300,bbox_inches='tight')
        plt.show()



    ###########Auxiliary Modes################
    def Split_ByIntersections(self,Pts_e,Pts_t):
        """Split the edge or trace by their intersections

        Arguments
        ---------
        xa, ya -- Cartesian coordinates of the first start-point.

        Author:Bin Wang(binwang.0213@gmail.com)
        Date: July. 2017
        """
        Edge_lines=Polygon2NodePair(Pts_e)
        Trace_lines=Pts_t

        #Split edge first
        New_Pts_e=[]
        for ei,edge in enumerate(Edge_lines):
            New_Pts_e.append(edge[0])
            for ti, trace in enumerate(Trace_lines):                
                if(LineSegIntersect2(edge, trace)):#Found Intersection Line
                    Pts_isect = LineIntersect(edge,trace)
                    print("Found Intersection-Edge", ei,
                          "Trace", ti, '@', Pts_isect)
                    New_Pts_e.append(Pts_isect)

        

        #Split edge and trace by intersections
        New_Pts_t = Split_IntersectLines(Trace_lines)

        return New_Pts_e,New_Pts_t

    def Append_Line(self, Pts_a=(0, 0), Pts_b=(0, 0), Nbd=1, panels=[], bd_marker=0, Type="Quad", refinement="linspace"):
        """Creates a BE along a line boundary. Anticlock wise, it decides the outward normal direction
           This can be used to create abritary polygon shape domain

        Arguments
        ---------
        xa, ya -- Cartesian coordinates of the first start-point.

        Author:Bin Wang(binwang.0213@gmail.com)
        Date: July. 2017
        """
        
        if (Type=="Quad"): 
            #Quadratic element of additional point
            Nbd=Nbd*2
        
        Pts=EndPointOnLine(Pts_a,Pts_b,Nbd,refinement)
        
        if (Type=="Const" or Type=="Linear"):
             for i in range(Nbd):
                #[0,1]  [1,2]
                Node1=Pts[i]
                Node2=Pts[i+1]
                panels.append(BEM_element(Node1,[],Node2,Type,bd_marker))#Neumann, Dirchlet    
        
        if (Type=="Quad"):
            #To get coordinates of additional one node for a Quadratic BE
            Nbd=int(Nbd/2)
            for i in range(Nbd):
                #[0,1,2]  [2,3,4]
                Node1=Pts[2*i]
                Node2=Pts[2*i+1]
                Node3=Pts[2*i+2]
                panels.append(BEM_element(Node1,Node2,Node3,Type,bd_marker))#Neumann, Dirchlet
        return Pts

    def getNodeId(self, eleid, local_id=0, EdgeorTrace='Edge'):
        """get the global node index based on the element id and local id
        # Ele1    Ele2    Ele3
        #[1,2,3] [3,4,5] [5,6,7]

        Author:Bin Wang(binwang.0213@gmail.com)
        Date: July. 2017
        """
        global_id = 0

        if(EdgeorTrace == 'Edge'):  # Edge
            #Special Case of Round end connect for continous element
            #if(eleid == self.Ne_edge - 1 and local_id == self.Nedof_edge-1 and self.Nedof_edge>1): #This is the last node at the last element
            #    global_id = 0
            #    return global_id

            if (self.BEMobj.TypeE_edge == "Quad"):
                global_id = 3 * eleid + local_id
            elif (self.BEMobj.TypeE_edge == "Linear"):
                global_id = 2 * eleid + local_id
            else:
                global_id = eleid
        elif(EdgeorTrace == 'Trace'): #Trace
            start_id=self.Ndof_edge
            if(EdgeorTrace == 'Trace'):  # Trace
                if (self.BEMobj.TypeE_trace == "Quad"):
                    global_id = start_id + 3 * eleid + local_id
                elif (self.BEMobj.TypeE_trace == "Linear"):
                    global_id = start_id + 2 * eleid + local_id
                else:
                    global_id = start_id + eleid
        else: #Source nodes
            start_id=self.Ndof_edge + self.Ndof_trace
            global_id = start_id + eleid

        return global_id

    def NumEle2LenEle(self,Ne_edge=None,Ne_trace=None):
        '''Determine the length of element based on total number of elements
        
        Author:Bin Wang(binwang.0213@gmail.com)
        Date: June. 2018
        '''
        TotalLength_edge=0.0
        TotalLength_trace=0.0

        if(Ne_edge is not None):
            if(Ne_edge < len(self.Pts_e)):  # Minimum element number is required
                Ne_edge = 2 * len(self.Pts_e)

            Pts_e=list(self.Pts_e)+[self.Pts_e[0]] #Round end connect
            for i in range(self.Num_boundary):
                TotalLength_edge=TotalLength_edge+calcDist(Pts_e[i],Pts_e[i+1])
            return TotalLength_edge / Ne_edge
        
        elif(Ne_trace is not None):
            if(Ne_trace < len(self.Pts_t)):  # Minimum element number is required
                Ne_trace = len(self.Pts_t)

            for i in range(self.Num_trace):
                TotalLength_trace=TotalLength_trace+calcDist(self.Pts_t[i][0],self.Pts_t[i][1])
            return TotalLength_trace / Ne_trace

    def IsBDIntersection(self,bd_markerID):
        '''Check a bd is a Trace or Not
        '''
        if (bd_markerID > self.Num_boundary - 1):  # this is a trace
            return True
        else:
            return False

    def getBDDof(self, bd_markerID):
        '''Get the number of Dof on a specific edge
        '''
        elementID = self.bdmarker2element(
            bd_markerID)  # find the element idx on this edge
        Ndof = 0
        if (bd_markerID > self.Num_boundary - 1):  # this is a trace
            TracerID = elementID[0][0]
            for ei, pl in enumerate(self.BEMobj.BEs_trace[TracerID]):
                Ndof = Ndof + pl.ndof

        else:  # This is boundary edge
            for i in range(len(elementID)):  # loop for all elements on this edge
                Ndof = Ndof + self.BEMobj.BEs_edge[elementID[i]].ndof

        return Ndof

    def getBDType(self,bd_markerID):
        '''Get the type of a specific edge
        '''
        if(bd_markerID>=self.Num_boundary+self.Num_trace):
            return 'Source'
        elif(bd_markerID>=self.Num_boundary):
            return 'Trace'
        else:
            return 'Edge'

    def getTraceID(self,bd_markerID):
        '''The the local trace id (0,1,2) from global bd id
        '''
        if (bd_markerID > self.Num_boundary - 1):  # this is a trace
            return bd_markerID-len(self.Pts_e)
        else:
            print("[BEM_2D_Mesh.py->getTraceID]This is not a Trace, but a Edge!")
            return -1
    
    def point_on_element(self, Pts):
        '''check and determine the element which a point is located on edge or trace

        Return
        --------
        -1              Interior node
        [ID1,ID2,...]   Pts on a Boundary element
        [(TraceID1,EleID1),(TraceID2,EleID2)] Pts on a Trace element
        0,1,2...        Pts on a source element

        '''
        Location='Interior'
        element=[]

        #Boundary edge
        for i in range(self.Num_boundary):#edge search
            Node=self.Pts_e[i]
            if (i==self.Num_boundary-1): 
                Node_next=self.Pts_e[0] #round connect
            else: 
                Node_next=self.Pts_e[i+1]
            #print('Checking...',Node,Node_next,self.tol,point_on_line(Pts,Node,Node_next,self.tol))
            if (point_on_line(Pts,Node,Node_next,self.tol)):# Found! point on a edge
                elementID=self.bdmarker2element(i)#element index on this edge
                for j in range(len(elementID)):
                    ID = elementID[j]
                    Pts_a = (self.BEMobj.BEs_edge[ID].xa, self.BEMobj.BEs_edge[ID].ya)
                    Pts_b = (self.BEMobj.BEs_edge[ID].xb, self.BEMobj.BEs_edge[ID].yb)
                    if(point_on_line(Pts, Pts_a, Pts_b,self.tol)):
                        Location='Edge'
                        element.append(ID)
                        break #element belonging is enough
        
        if(len(element)>0): return Location,element
        
        #Internal trace
        for ti in range(self.Num_trace):
            markerID=ti+self.Num_boundary
            Node=self.Pts_t[ti][0]
            Node_next = self.Pts_t[ti][1]

            if (point_on_line(Pts, Node, Node_next,self.tol)):  # Found! point on a edge
                elementID = self.bdmarker2element(markerID)  # element index on this edge
                for j in range(len(elementID)):
                    TracerID = elementID[j][0]
                    ID = elementID[j][1]
                    Pts_a = (self.BEMobj.BEs_trace[ti][ID].xa, self.BEMobj.BEs_trace[ti][ID].ya)
                    Pts_b = (self.BEMobj.BEs_trace[ti][ID].xb, self.BEMobj.BEs_trace[ti][ID].yb)
                    if(point_on_line(Pts, Pts_a, Pts_b,self.tol)):
                        Location='Trace'
                        element.append([TracerID,ID])
                        break  # element belonging is enough

        if(len(element)>0): return Location,element

        #Internal source
        for si in range(self.Num_source):
            Node=self.Pts_s[si]
            dist=calcDist(Node,Pts)

            if(dist<self.tol):
                Location='Source'
                return Location,si
        
        return Location,-1

    def element2edge(self,idx_element):
        #find the edge index form a elemetn index
        #Currently only support for edge element

        pts_c=[self.BEMobj.BEs_edge[idx_element].xc,self.BEMobj.BEs_edge[idx_element].yc] #central point of this element

        for i in range(self.Num_boundary):#edge search
            Node=self.Pts_e[i]
            if (i==self.Num_boundary-1): 
                Node_next=self.Pts_e[0] #round connect
            else:
                Node_next=self.Pts_e[i+1]
            if (point_on_line(pts_c,Node,Node_next,self.tol)):#edge found
                return i

        print('Error!! Func-element2edge')   


    def bdmarker2element(self, markerID):
        #find the element index based on bd markerID(boundary index)
        # example： markerID=3  Element index=[0 1]

        index=[]

        if(markerID<self.TraceStartID):#this is a boundary edge
            elementID_start=0
            for i in range(markerID):
                elementID_start+=self.NumE_bd[i]
            for i in range(self.NumE_bd[markerID]):
                index.append(elementID_start+i)

        if (markerID>=self.TraceStartID and markerID<self.SourceStartID):#this is a trace
            tracerID=markerID-self.Num_boundary
            for i in range(len(self.BEMobj.BEs_trace[tracerID])):
                index.append([tracerID,i])
        
        if(markerID>self.SourceStartID-1):#This is a source  
            index.append(markerID-self.Num_boundary-self.Num_trace)           
        
        return np.array(index)

    def bd2element(self,element_type="Const",eleid=0,node_values=[]):
        #! Assuming the discontinous element is used here
        #extract the node_values of a element from a sets of values along a edge
        #eleid is the local index, e.g 3 element on a edge, eleid=0,1,2
        if(element_type=="Const"):#[0] [1]
            return [node_values[eleid]]
        elif(element_type=="Linear"):#[0,1] [1,2]
            return [node_values[eleid*2],node_values[eleid*2+1]]
        elif(element_type=="Quad"):#[0,1,2] [2,3,4]
            return [node_values[eleid*3],node_values[eleid*3+1],node_values[eleid*3+2]]

    def getEdgeEleNodeCoords(self,eleid):
        #Get the node coordinates of a edge element
        return self.BEMobj.BEs_edge[eleid].get_nodes()


    def EndPoint2bdmarker(self,Pts0,Pts1):
        #find the bd_markerID based on two end points[Pts0,Pts1]
        #currently only boundary edge are support

        pts_c=[(Pts0[0]+Pts1[0])*0.5,(Pts0[1]+Pts1[1])*0.5] #central point of this element

        for i in range(self.Num_boundary):#edge search
            Node=self.Pts_e[i]
            if (i==self.Num_boundary-1): 
                Node_next=self.Pts_e[0] #round connect
            else: 
                Node_next=self.Pts_e[i+1]
            if (point_on_line(pts_c,Node,Node_next,self.tol)):#edge found
                return i
        print("Can not find the bd_markerID",Pts0,Pts1)
    
