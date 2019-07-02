from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [Extension('Unified_Element',
                         ['Unified_Element.pyx'],
                         include_dirs=[numpy.get_include()]
                         )]
setup(
    name='Unified_Element',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
    include_dirs=[numpy.get_include()]
)
#python setup3.py build_ext -i clean
