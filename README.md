PyBEM2D: A Python-based Boundary Element Library
==============================================================================================
Bin Wang (binwang.0213@gmail.com), Yin Feng
Department of Petroleum Engineering, Univeristy of Louisiana at Lafayette, Lafayette, US, 70506

<p align="center">
  <img src = "https://github.com/BinWang0213/PyBEM2D/blob/master/img/Multidomain.png" height="250">
</p>

`PyBEM2D` is a open source boundary element library for two dimensional geometries. It is designed for academic users to quickly learn and test various BEM's techniques under the framework of Python. With the modular code structure and easy-to-use interfaces to `PyBEM2D`, it is easy to get started and added more features for academic users. 

Currently available features include:
* Easy-to-use python interface for BEM meshing and post-processing
* Potential problem
* Classic point collocation method for BEM discretization
* Support for constant,linear and quadratic element types
* Corner treatment with double node algorihtm [Brebbia, 1991]
* Near singular integration with adaptive element subdivision algorithm [Gao et al. 2014]
* Multi-domain problem and parallel computing with domain decomposition method [Wang et al, 2018]

`PyBEM2D` is currently support for `64-bit` `Windows` and `Linux` platforms, several `Jupyter-Notebook` examples are provided 


# Solving A potential porblem in PyBEM2D

<p align="center">
  <img src = "https://github.com/BinWang0213/PyBEM2D/blob/master/img/LSU.png">
</p>

After downloading and unzipping the current <a href="https://github.com/BinWang0213/PyBEM2D/archive/master.zip">repository</a>, navigate to the library directory and run a simple example contained in `Example/KingDomain2D.ipynb`:
```python
from Lib.BEM_Solver.BEM_2D import *
from Lib.Domain_Decomposition.Coupling_Main import *

KingDomain=BEM2D()

#1.Build Mesh-#Anti-clock wise for internal domain
Boundary_vert=[(0.0, 0.0), (5.0, 0.0), (5.0, 1.0), (3.0, 1.0),(3.0,2.0), #bottom
               (4.5,2.0),(4.5,3.0),(3.0,3.0),(3.0,4.0),                  #middle
               (4.0,4.0),(4.0,5.0),(1.0,5.0),(1.0,4.0),(2.0,4.0),        #top
               (2.0,3.0),(0.5,3.0),(0.5,2.0),(2.0,2.0),                  #middle
               (2.0,1.0),(0.0,1.0)]  #bottom
element_esize=1.0

KingDomain.set_Mesh(Boundary_vert,[],element_esize,[],Type="Quad")

#2.Set Boundary condition
bc0=[(10,100),(0,10)]
bc1=[(5,-10),(15,-10)]

KingDomain.set_BoundaryCondition(DirichletBC=bc0,NeumannBC=bc1)

#3. Solve and Show Solution
puv=KingDomain.Solve()

puv=KingDomain.plot_Solution(v_range=(0,40),p_range=(10,100))
```

# Reference

If you make use of `PyBEM2D` please reference appropriately. The algorithmic developments behind `PyBEM2D` have been the subject of a number of publications, beginning with my graduate research at the University of Louisiana at Lafayette:

`[1]` - Wang, B., Feng, Y., Du, J., et al. (2017) An Embedded Grid-Free Approach for Near Wellbore Streamline Simulation. doi:10.2118/SPE-182614-MS
https://www.researchgate.net/publication/313665682_An_Embedded_Grid-Free_Approach_for_Near_Wellbore_Streamline_Simulation

# License

This code is released under the terms of the BSD license, and thus free for commercial and research use. Feel free to use the code into your own project with a PROPER REFERENCE.  




