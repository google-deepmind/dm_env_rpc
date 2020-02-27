# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Install script for setuptools."""

import importlib.machinery
import os

from distutils.cmd import Command
import pkg_resources
from setuptools import find_packages
from setuptools import setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
GOOGLE_COMMON_PROTOS_ROOT_DIR = os.path.join(ROOT_DIR,
                                             'third_party/api-common-protos')


class _GenerateProtoFiles(Command):
  """Command to generate protobuf bindings for dm_env_rpc.proto."""

  descriptions = 'Generates Python protobuf bindings for dm_env_rpc.proto.'
  user_options = []

  def initialize_options(self):
    pass

  def finalize_options(self):
    pass

  def run(self):
    # Import grpc_tools here, after setuptools has installed setup_requires
    # dependencies.
    from grpc_tools import protoc  # pylint: disable=g-import-not-at-top

    if not os.path.exists(
        os.path.join(GOOGLE_COMMON_PROTOS_ROOT_DIR, 'google/rpc/status.proto')):
      raise RuntimeError(
          'Cannot find third_party/api-common-protos. '
          'Please run `git submodule init && git submodule update` to install '
          'the api-common-protos submodule.'
      )
    dm_env_rpc_proto = os.path.join(ROOT_DIR, 'dm_env_rpc/v1/dm_env_rpc.proto')
    grpc_protos_include = pkg_resources.resource_filename(
        'grpc_tools', '_proto')
    proto_args = [
        'grpc_tools.protoc',
        '--proto_path={}'.format(GOOGLE_COMMON_PROTOS_ROOT_DIR),
        '--proto_path={}'.format(grpc_protos_include),
        '--proto_path={}'.format(ROOT_DIR),
        '--python_out={}'.format(ROOT_DIR),
        '--grpc_python_out={}'.format(ROOT_DIR),
        dm_env_rpc_proto,
    ]
    if protoc.main(proto_args) != 0:
      raise RuntimeError('ERROR: {}'.format(proto_args))


class _BuildExt(build_ext):
  """Generate protobuf bindings in build_ext stage."""

  def run(self):
    self.run_command('generate_protos')
    build_ext.run(self)


class _BuildPy(build_py):
  """Generate protobuf bindings in build_py stage."""

  def run(self):
    self.run_command('generate_protos')
    build_py.run(self)


setup(
    name='dm-env-rpc',
    version=importlib.machinery.SourceFileLoader(
        '_version', 'dm_env_rpc/_version.py').load_module().__version__,
    description='A networking protocol for agent-environment communication.',
    author='DeepMind',
    license='Apache License, Version 2.0',
    keywords='reinforcement-learning python machine learning',
    packages=find_packages(exclude=['examples']),
    install_requires=[
        'dm-env>=1.2',
        'googleapis-common-protos',
        'grpcio',
        'numpy',
        'protobuf>=3.8',
    ],
    tests_require=[
        'absl-py',
        'nose',
        'mock',
        'portpicker',
    ],
    python_requires='>=3.5',
    setup_requires=['grpcio-tools'],
    extras_require={
        'examples': ['pygame', 'portpicker'],
    },
    cmdclass={
        'build_ext': _BuildExt,
        'build_py': _BuildPy,
        'generate_protos': _GenerateProtoFiles,
    },
    test_suite='nose.collector',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
