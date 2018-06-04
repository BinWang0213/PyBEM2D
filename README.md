PyBEM2D: A Python-based Boundary Element Library
==============================================================================================
Bin Wang (binwang.0213@gmail.com), Craft and Hawkins Department of Petroleum Engineering, Louisiana State University, Baton Rouge, US

Yin Feng, Department of Petroleum Engineering, Univeristy of Louisiana at Lafayette, Lafayette, US

<p align="center">
  <img src = "https://github.com/BinWang0213/PyBEM2D/blob/master/img/Multidomain.png" height="250">
</p>

`PyBEM2D` is a open source boundary element library for two dimensional geometries. It is designed for academic users to quickly learn and test various BEM's techniques under the framework of Python. With the modular code structure and easy-to-use interfaces to `PyBEM2D`, it is easy to get started and added more features for academic users. 

Currently available features include:
* Easy-to-use python interface for BEM meshing and post-processing
* Potential problem
* Classic point collocation method for BEM discretization
* Support for constant,linear and quadratic element types
* Support internal sources (traces)
* Corner treatment with double node algorihtm [Brebbia, 1991]
* Near singular integration with adaptive element subdivision algorithm [Gao et al. 2014]
* Multi-domain problem and parallel computing with domain decomposition method [Wang et al, 2018]

`PyBEM2D` is currently support for `64-bit` `Windows` and `Linux` platforms, several `Jupyter-Notebook` examples are provided 


# Solving A potential porblem in PyBEM2D

<p align="center">
  <img src = "https://github.com/BinWang0213/PyBEM2D/blob/master/img/LSU.png">
</p>

After downloading and unzipping the current <a href="https://github.com/BinWang0213/PyBEM2D/archive/master.zip">repository</a>, navigate to the library directory and run a simple example contained in `Example/LSU_Traces.ipynb`:
```python
from Lib.BEM_Solver.BEM_2D import *
from Lib.Domain_Decomposition.Coupling_Main import *

BEM_Case1=BEM2D()

#1.Build Mesh
Boundary_vert=[(0.0, 0.0), (1.0, 0.0), (1.0, 0.75), (0.0, 0.75)] #Anti-clock wise for internal domain

Trace_vert=[]
L=[((0.1, 0.15), (0.1, 0.6)),((0.1, 0.15), (0.3, 0.15))]
S=[((0.6,0.6),(0.38,0.6)),((0.38,0.6),(0.38,0.38)),((0.38,0.38),(0.6,0.38)),((0.6,0.38),(0.6,0.14)),((0.6,0.14),(0.38,0.14))]
U= [((0.72,0.6),(0.72,0.14)),((0.72,0.14),(0.92,0.14)),((0.92,0.14),(0.92,0.6))]
Points= [((0.195,0.35),(0.205,0.35)),
      ((0.795,0.35),(0.805,0.35))]
Trace_vert=L+S+U+Points
element_esize=0.2 #Edge mesh is important to overall mass balance
element_tszie=0.1 #Trace mesh size

BEM_Case1.set_Mesh(Boundary_vert,Trace_vert,element_esize,element_tszie,Type="Const")

#2.Set Boundary condition
bc0=[(4,50),(5,50),(11,50),(12,50),(13,50),(14,10),(15,10)]
bc1=[(6,-100),(7,-100),(8,-100),(9,-100),(10,-100)]
BEM_Case1.set_BoundaryCondition(DirichletBC=bc0,NeumannBC=bc1)

#3. Solve and plot
Mat=BEM_Case1.Solve()
PUV=BEM_Case1.plot_Solution(v_range=(-150,250),p_range=(30,50),resolution=30)
```

# Reference

If you make use of `PyBEM2D` please reference appropriately. The algorithmic developments behind `PyBEM2D` have been the subject of a number of publications, beginning with my graduate research at the University of Louisiana at Lafayette:

`[1]` - Wang, B., Feng, Y., Du, J., et al. (2017) An Embedded Grid-Free Approach for Near Wellbore Streamline Simulation. doi:10.2118/SPE-182614-MS
https://www.researchgate.net/publication/313665682_An_Embedded_Grid-Free_Approach_for_Near_Wellbore_Streamline_Simulation

# License

This code is released under the terms of the BSD license, and thus free for commercial and research use. Feel free to use the code into your own project with a PROPER REFERENCE.  




