import glob
import os
import os.path as osp

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# source files
extensions_dir = "csrc"
sources = glob.glob(osp.join(extensions_dir, "*.cpp")) + glob.glob(osp.join(extensions_dir, "*.cu"))
sources = [path for path in sources if "hip" not in path] # remove generated 'hip' files, in case of rebuilds

# flags
extra_compile_args = {'cxx': ['-O3', '-Wno-sign-compare'], 'nvcc': ['-O3']}
extra_link_args = ["-s"]

ext_modules = [
    CUDAExtension("nerfngp_cuda",sources, # The example_kernel should match pybind11
                  include_dirs=[osp.join(extensions_dir, "include")],
                  extra_compile_args=extra_compile_args,extra_link_args=extra_link_args)
]


setup(
    name="instant-ngp",
    # version=__version__,
    description="Custom C++ and CUDA kernels for instant NGP",
    author="Chaitanya",
    author_email="cgaddipati@wpi.edu",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    packages=find_packages(),
)