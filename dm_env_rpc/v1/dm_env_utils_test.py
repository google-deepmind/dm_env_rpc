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
"""Tests for dm_env_rpc/dm_env utilities."""

import typing

from absl.testing import absltest
from dm_env import specs
import numpy as np

from dm_env_rpc.v1 import dm_env_rpc_pb2
from dm_env_rpc.v1 import dm_env_utils
from dm_env_rpc.v1 import spec_manager


class TensorSpecToDmEnvSpecTests(absltest.TestCase):

  def test_no_bounds_gives_arrayspec(self):
    tensor_spec = dm_env_rpc_pb2.TensorSpec()
    tensor_spec.dtype = dm_env_rpc_pb2.DataType.UINT32
    tensor_spec.shape[:] = [3]
    tensor_spec.name = 'foo'
    actual = dm_env_utils.tensor_spec_to_dm_env_spec(tensor_spec)
    self.assertEqual(specs.Array(shape=[3], dtype=np.uint32), actual)
    self.assertEqual('foo', actual.name)

  def test_string_give_string_array(self):
    tensor_spec = dm_env_rpc_pb2.TensorSpec()
    tensor_spec.dtype = dm_env_rpc_pb2.DataType.STRING
    tensor_spec.shape[:] = [1, 2, 3]
    tensor_spec.name = 'string_spec'
    actual = dm_env_utils.tensor_spec_to_dm_env_spec(tensor_spec)
    self.assertEqual(specs.StringArray(shape=[1, 2, 3]), actual)
    self.assertEqual('string_spec', actual.name)

  def test_scalar_with_0_n_bounds_gives_discrete_array(self):
    tensor_spec = dm_env_rpc_pb2.TensorSpec()
    tensor_spec.dtype = dm_env_rpc_pb2.DataType.UINT32
    tensor_spec.name = 'foo'

    max_value = 9
    tensor_spec.min.uint32s.array[:] = [0]
    tensor_spec.max.uint32s.array[:] = [max_value]
    actual = typing.cast(specs.DiscreteArray,
                         dm_env_utils.tensor_spec_to_dm_env_spec(tensor_spec))
    expected = specs.DiscreteArray(
        num_values=max_value + 1, dtype=np.uint32, name='foo')
    self.assertEqual(expected, actual)
    self.assertEqual(0, actual.minimum)
    self.assertEqual(max_value, actual.maximum)
    self.assertEqual('foo', actual.name)

  def test_scalar_with_1_n_bounds_gives_bounded_array(self):
    tensor_spec = dm_env_rpc_pb2.TensorSpec()
    tensor_spec.dtype = dm_env_rpc_pb2.DataType.UINT32
    tensor_spec.name = 'foo'
    tensor_spec.min.uint32s.array[:] = [1]
    tensor_spec.max.uint32s.array[:] = [10]
    actual = dm_env_utils.tensor_spec_to_dm_env_spec(tensor_spec)
    expected = specs.BoundedArray(
        shape=(), dtype=np.uint32, minimum=1, maximum=10, name='foo')
    self.assertEqual(expected, actual)
    self.assertEqual('foo', actual.name)

  def test_scalar_with_0_min_and_no_max_bounds_gives_bounded_array(self):
    tensor_spec = dm_env_rpc_pb2.TensorSpec()
    tensor_spec.dtype = dm_env_rpc_pb2.DataType.UINT32
    tensor_spec.name = 'foo'
    tensor_spec.min.uint32s.array[:] = [0]
    actual = dm_env_utils.tensor_spec_to_dm_env_spec(tensor_spec)
    expected = specs.BoundedArray(
        shape=(), dtype=np.uint32, minimum=0, maximum=2**32 - 1, name='foo')
    self.assertEqual(expected, actual)
    self.assertEqual('foo', actual.name)

  def test_only_min_bounds(self):
    tensor_spec = dm_env_rpc_pb2.TensorSpec()
    tensor_spec.dtype = dm_env_rpc_pb2.DataType.UINT32
    tensor_spec.shape[:] = [3]
    tensor_spec.name = 'foo'
    tensor_spec.min.uint32s.array[:] = [1]
    actual = dm_env_utils.tensor_spec_to_dm_env_spec(tensor_spec)
    expected = specs.BoundedArray(
        shape=[3], dtype=np.uint32, minimum=1, maximum=2**32 - 1)
    self.assertEqual(expected, actual)
    self.assertEqual('foo', actual.name)

  def test_only_max_bounds(self):
    tensor_spec = dm_env_rpc_pb2.TensorSpec()
    tensor_spec.dtype = dm_env_rpc_pb2.DataType.UINT32
    tensor_spec.shape[:] = [3]
    tensor_spec.name = 'foo'
    tensor_spec.max.uint32s.array[:] = [10]
    actual = dm_env_utils.tensor_spec_to_dm_env_spec(tensor_spec)
    expected = specs.BoundedArray(
        shape=[3], dtype=np.uint32, minimum=0, maximum=10)
    self.assertEqual(expected, actual)
    self.assertEqual('foo', actual.name)

  def test_both_bounds(self):
    tensor_spec = dm_env_rpc_pb2.TensorSpec()
    tensor_spec.dtype = dm_env_rpc_pb2.DataType.UINT32
    tensor_spec.shape[:] = [3]
    tensor_spec.name = 'foo'
    tensor_spec.min.uint32s.array[:] = [1]
    tensor_spec.max.uint32s.array[:] = [10]
    actual = dm_env_utils.tensor_spec_to_dm_env_spec(tensor_spec)
    expected = specs.BoundedArray(
        shape=[3], dtype=np.uint32, minimum=1, maximum=10)
    self.assertEqual(expected, actual)
    self.assertEqual('foo', actual.name)

  def test_bounds_oneof_not_set_gives_dtype_bounds(self):
    tensor_spec = dm_env_rpc_pb2.TensorSpec()
    tensor_spec.dtype = dm_env_rpc_pb2.DataType.UINT32
    tensor_spec.shape[:] = [3]
    tensor_spec.name = 'foo'

    # Just to force the message to get created.
    tensor_spec.min.floats.array[:] = [3.0]
    tensor_spec.min.ClearField('floats')

    actual = dm_env_utils.tensor_spec_to_dm_env_spec(tensor_spec)
    expected = specs.BoundedArray(
        shape=[3], dtype=np.uint32, minimum=0, maximum=2**32 - 1)
    self.assertEqual(expected, actual)
    self.assertEqual('foo', actual.name)

  def test_bounds_wrong_type_gives_error(self):
    tensor_spec = dm_env_rpc_pb2.TensorSpec()
    tensor_spec.dtype = dm_env_rpc_pb2.DataType.UINT32
    tensor_spec.shape[:] = [3]
    tensor_spec.name = 'foo'
    tensor_spec.min.floats.array[:] = [1.9]
    with self.assertRaisesRegex(ValueError, 'uint32'):
      dm_env_utils.tensor_spec_to_dm_env_spec(tensor_spec)

  def test_bounds_on_string_gives_error(self):
    tensor_spec = dm_env_rpc_pb2.TensorSpec()
    tensor_spec.dtype = dm_env_rpc_pb2.DataType.STRING
    tensor_spec.shape[:] = [2]
    tensor_spec.name = 'named'
    tensor_spec.min.floats.array[:] = [1.9]
    tensor_spec.max.floats.array[:] = [10.0]
    with self.assertRaisesRegex(ValueError, 'string'):
      dm_env_utils.tensor_spec_to_dm_env_spec(tensor_spec)


class DmEnvSpecTests(absltest.TestCase):

  def test_spec(self):
    dm_env_rpc_specs = {
        54:
            dm_env_rpc_pb2.TensorSpec(
                name='fuzz', shape=[3], dtype=dm_env_rpc_pb2.DataType.FLOAT),
        55:
            dm_env_rpc_pb2.TensorSpec(
                name='foo', shape=[2], dtype=dm_env_rpc_pb2.DataType.INT32),
    }
    manager = spec_manager.SpecManager(dm_env_rpc_specs)

    expected = {
        'foo': specs.Array(shape=[2], dtype=np.int32),
        'fuzz': specs.Array(shape=[3], dtype=np.float32)
    }

    self.assertDictEqual(expected, dm_env_utils.dm_env_spec(manager))

  def test_empty_spec(self):
    self.assertDictEqual({},
                         dm_env_utils.dm_env_spec(spec_manager.SpecManager({})))

  def test_spec_generate_and_validate_scalars(self):
    dm_env_rpc_specs = []
    for name, dtype in dm_env_rpc_pb2.DataType.items():
      if dtype != dm_env_rpc_pb2.DataType.INVALID_DATA_TYPE:
        dm_env_rpc_specs.append(
            dm_env_rpc_pb2.TensorSpec(name=name, shape=(), dtype=dtype))

    for dm_env_rpc_spec in dm_env_rpc_specs:
      spec = dm_env_utils.tensor_spec_to_dm_env_spec(dm_env_rpc_spec)
      value = spec.generate_value()
      spec.validate(value)

  def test_spec_generate_and_validate_tensors(self):
    example_shape = (10, 10, 3)

    dm_env_rpc_specs = []
    for name, dtype in dm_env_rpc_pb2.DataType.items():
      if dtype != dm_env_rpc_pb2.DataType.INVALID_DATA_TYPE:
        dm_env_rpc_specs.append(
            dm_env_rpc_pb2.TensorSpec(
                name=name, shape=example_shape, dtype=dtype))

    for dm_env_rpc_spec in dm_env_rpc_specs:
      spec = dm_env_utils.tensor_spec_to_dm_env_spec(dm_env_rpc_spec)
      value = spec.generate_value()
      spec.validate(value)

if __name__ == '__main__':
  absltest.main()
