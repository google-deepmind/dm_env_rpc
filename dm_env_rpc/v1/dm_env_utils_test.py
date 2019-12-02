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

from absl.testing import absltest
from absl.testing import parameterized
from dm_env import specs
import numpy as np

from dm_env_rpc.v1 import dm_env_rpc_pb2
from dm_env_rpc.v1 import dm_env_utils
from dm_env_rpc.v1 import spec_manager


class NpRangeInfoTests(parameterized.TestCase):

  def test_floating(self):
    expected_min = np.finfo(np.float32).min
    actual_min = dm_env_utils._np_range_info(np.float32).min
    self.assertEqual(expected_min, actual_min)

  def test_integer(self):
    actual_min = dm_env_utils._np_range_info(np.uint32).min
    self.assertEqual(0, actual_min)

  def test_string_gives_error(self):
    with self.assertRaisesRegex(ValueError, 'numpy.str_'):
      _ = dm_env_utils._np_range_info(np.str_).min


class FindExtremeTests(parameterized.TestCase):

  def test_min_from_type(self):
    tensor_spec = dm_env_rpc_pb2.TensorSpec()
    tensor_spec.dtype = dm_env_rpc_pb2.DataType.UINT32
    tensor_spec.shape[:] = [3]
    tensor_type = np.uint32
    self.assertEqual(
        0, dm_env_utils._find_extreme(tensor_spec, tensor_type, 'min'))

  def test_explicit_min(self):
    tensor_spec = dm_env_rpc_pb2.TensorSpec()
    tensor_spec.dtype = dm_env_rpc_pb2.DataType.UINT32
    tensor_spec.shape[:] = [3]
    tensor_spec.min.uint32 = 1
    tensor_type = np.uint32
    self.assertEqual(
        1, dm_env_utils._find_extreme(tensor_spec, tensor_type, 'min'))


class TensorSpecToDmEnvSpecTests(parameterized.TestCase):

  def test_no_bounds_gives_arrayspec(self):
    tensor_spec = dm_env_rpc_pb2.TensorSpec()
    tensor_spec.dtype = dm_env_rpc_pb2.DataType.UINT32
    tensor_spec.shape[:] = [3]
    tensor_spec.name = 'foo'
    actual = dm_env_utils.tensor_spec_to_dm_env_spec(tensor_spec)
    self.assertEqual(specs.Array(shape=[3], dtype=np.uint32), actual)
    self.assertEqual('foo', actual.name)

  def test_only_min_bounds(self):
    tensor_spec = dm_env_rpc_pb2.TensorSpec()
    tensor_spec.dtype = dm_env_rpc_pb2.DataType.UINT32
    tensor_spec.shape[:] = [3]
    tensor_spec.name = 'foo'
    tensor_spec.min.uint32 = 1
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
    tensor_spec.max.uint32 = 10
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
    tensor_spec.min.uint32 = 1
    tensor_spec.max.uint32 = 10
    actual = dm_env_utils.tensor_spec_to_dm_env_spec(tensor_spec)
    expected = specs.BoundedArray(
        shape=[3], dtype=np.uint32, minimum=1, maximum=10)
    self.assertEqual(expected, actual)
    self.assertEqual('foo', actual.name)

  def test_bounds_oneof_not_set_gives_error(self):
    tensor_spec = dm_env_rpc_pb2.TensorSpec()
    tensor_spec.dtype = dm_env_rpc_pb2.DataType.UINT32
    tensor_spec.shape[:] = [3]
    tensor_spec.name = 'foo'

    # Just to force the message to get created.
    tensor_spec.min.float = 3
    tensor_spec.min.ClearField('float')

    with self.assertRaisesRegex(ValueError, 'min'):
      dm_env_utils.tensor_spec_to_dm_env_spec(tensor_spec)

  def test_bounds_wrong_type_gives_error(self):
    tensor_spec = dm_env_rpc_pb2.TensorSpec()
    tensor_spec.dtype = dm_env_rpc_pb2.DataType.UINT32
    tensor_spec.shape[:] = [3]
    tensor_spec.name = 'foo'
    tensor_spec.min.float = 1.9
    with self.assertRaisesRegex(ValueError, 'numpy.uint32'):
      dm_env_utils.tensor_spec_to_dm_env_spec(tensor_spec)

  def test_bounds_on_string_gives_error(self):
    tensor_spec = dm_env_rpc_pb2.TensorSpec()
    tensor_spec.dtype = dm_env_rpc_pb2.DataType.STRING
    tensor_spec.shape[:] = [2]
    tensor_spec.name = 'named'
    tensor_spec.min.float = 1.9
    tensor_spec.max.float = 10.0
    with self.assertRaisesRegex(ValueError, 'numpy.str_'):
      dm_env_utils.tensor_spec_to_dm_env_spec(tensor_spec)


class DmEnvSpecTests(parameterized.TestCase):

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


if __name__ == '__main__':
  absltest.main()
