from distutils.core import setup, Extension
from Cython.Build import cythonize
import os

#os.environ["CC"] = "gcc-5"
#os.environ["CXX"] = "g++"

ext1 = Extension("bernoulli_mixture_wrap",
                 extra_compile_args=["-g"],
                sources=["bernoulli_mixture_wrap.pyx", "bernoulli_mixture.c", "lib/logsumexp.c"],
                )
                
setup(name = 'bernoulli_mixture', ext_modules = cythonize([ext1]))


