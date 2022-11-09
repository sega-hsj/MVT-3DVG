import os
import subprocess

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def make_cuda_ext(name, module, sources):
    cuda_ext = CUDAExtension(
        name='%s.%s' % (module, name),
        sources=[os.path.join(*module.split('.'), src) for src in sources]
    )
    return cuda_ext

setup(name='referit3d',
      version='0.1',
      description='Comprehension of localizing 3D objects in scenes.',
      url='http://github.com/referit3d/referit3d',
      author='referit3d_team',
      author_email='optas@cs.stanford.edu',
      license='MIT',
      install_requires=['scikit-learn',
                        'matplotlib',
                        'six',
                        'tqdm',
                        'pandas',
                        'plyfile',
                        'requests',
                        'symspellpy',
                        'termcolor',
                        'tensorboardX',
                        'shapely',
                        'pyyaml',
                        'easydict'
                        ],
      packages=['referit3d'],
      zip_safe=False)
