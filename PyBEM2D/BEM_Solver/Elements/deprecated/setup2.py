from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
ext_modules = [Extension('Constant_element',
                         ['Constant_element.pyx'],
                         )]
setup(
    name='Constant_element',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules
)
#python setup2.py build_ext -i clean
