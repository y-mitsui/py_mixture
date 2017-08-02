from distutils.core import setup, Extension
from Cython.Build import cythonize
import os

#os.environ["CC"] = "gcc-5"
#os.environ["CXX"] = "g++"

ext1 = Extension("many_mixture_wrap",
                 extra_compile_args=["-g"],
                sources=["many_mixture_wrap.pyx", "many_mixture.c"],
                extra_link_args=["-lgsl", "-lgslcblas"]
                )
                
setup(name = 'many_mixture', ext_modules = cythonize([ext1]))


