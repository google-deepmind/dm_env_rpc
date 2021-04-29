# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
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
"""Micro-benchmark for tensor_utils.pack_tensor."""

import abc
import timeit

from absl import app
from absl import flags
import numpy as np

from dm_env_rpc.v1 import tensor_utils

flags.DEFINE_integer('repeats', 10000,
                     'Number of times each benchmark will run.')
FLAGS = flags.FLAGS


class _AbstractBenchmark(metaclass=abc.ABCMeta):
  """Base class for benchmarks using timeit."""

  def run(self):
    time = timeit.timeit(self.statement, setup=self.setup, number=FLAGS.repeats)
    print(f'{self.name} -- overall: {time:0.2f} s, '
          f'per call: {time/FLAGS.repeats:0.1e} s')

  def setup(self):
    pass

  @abc.abstractmethod
  def statement(self):
    pass

  @abc.abstractproperty
  def name(self):
    pass


class _PackBenchmark(_AbstractBenchmark):
  """Benchmark for packing a numpy array to a Tensor proto."""

  def __init__(self, dtype, shape):
    self._name = f'pack {np.dtype(dtype).name}'
    self._dtype = dtype
    self._shape = shape

  @property
  def name(self):
    return self._name

  def setup(self):
    # Use non-zero values in case there's something special about zero arrays.
    self._unpacked = np.arange(
        np.prod(self._shape), dtype=self._dtype).reshape(self._shape)

  def statement(self):
    self._unpacked.flat[0] += 1  # prevent caching of the result
    tensor_utils.pack_tensor(self._unpacked, self._dtype)


class _UnpackBenchmark(_AbstractBenchmark):
  """Benchmark for unpacking a Tensor proto to a numpy array."""

  def __init__(self, dtype, shape):
    self._name = f'unpack {np.dtype(dtype).name}'
    self._shape = shape
    self._dtype = dtype

  @property
  def name(self):
    return self._name

  def setup(self):
    # Use non-zero values in case there's something special about zero arrays.
    tensor = np.arange(
        np.prod(self._shape), dtype=self._dtype).reshape(self._shape)
    self._packed = tensor_utils.pack_tensor(tensor, self._dtype)

  def statement(self):
    tensor_utils.unpack_tensor(self._packed)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  # Pick `shape` such that number of bytes is consistent between benchmarks.
  benchmarks = (
      _PackBenchmark(dtype=np.uint8, shape=(128, 128, 3)),
      _PackBenchmark(dtype=np.int32, shape=(64, 64, 3)),
      _PackBenchmark(dtype=np.int64, shape=(32, 64, 3)),
      _PackBenchmark(dtype=np.uint32, shape=(64, 64, 3)),
      _PackBenchmark(dtype=np.uint64, shape=(32, 64, 3)),
      _PackBenchmark(dtype=np.float32, shape=(64, 64, 3)),
      _PackBenchmark(dtype=np.float64, shape=(32, 64, 3)),
      _UnpackBenchmark(dtype=np.uint8, shape=(128, 128, 3)),
      _UnpackBenchmark(dtype=np.int32, shape=(64, 64, 3)),
      _UnpackBenchmark(dtype=np.int64, shape=(32, 64, 3)),
      _UnpackBenchmark(dtype=np.uint32, shape=(64, 64, 3)),
      _UnpackBenchmark(dtype=np.uint64, shape=(32, 64, 3)),
      _UnpackBenchmark(dtype=np.float32, shape=(64, 64, 3)),
      _UnpackBenchmark(dtype=np.float64, shape=(32, 64, 3)),
  )
  for benchmark in benchmarks:
    benchmark.run()


if __name__ == '__main__':
  app.run(main)
