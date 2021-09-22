# Copyright 2021 CRS4
# 
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from setuptools import setup
from distutils.core import Extension
import pybind11
from glob import glob

EXTRA_COMPILE_ARGS =['-fvisibility=hidden', '-g0', '-Wall', '-Wextra', '-pedantic', '-std=c++17', '-O2']

cpp_handler = Extension("BPH",
        sorted(glob("cassandradl/cpp/*.cpp")),
        include_dirs=[
            '/usr/include/opencv4',
            '/usr/local/include/eigen3/',
            pybind11.get_include(user=True),
        ],
        language='c++',
        libraries=['cassandra', 'opencv_core', 'opencv_imgcodecs', 'eddl', 'ecvl_core', 'ecvl_eddl'],
        extra_compile_args=EXTRA_COMPILE_ARGS,  
    )

ext_mods = [ cpp_handler ]

setup(
    name="cassandradl",
    version="0.1",
    author="Francesco Versaci, Giovanni Busonera",
    author_email="francesco.versaci@gmail.com, giovanni.busonera@crs4.it",
    description="Cassandra data loader for ML pipelines",
    packages=['cassandradl'],
    url="https://github.com/bla",
    ext_modules=ext_mods,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
    ],
    python_requires='>=3.6',
)


