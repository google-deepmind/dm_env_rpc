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
"""Tests for dm_env_rpc helper functions."""

from absl.testing import absltest

import numpy as np

from dm_env_rpc.v1 import dm_env_rpc_pb2
from dm_env_rpc.v1 import tensor_spec_utils
from dm_env_rpc.v1 import tensor_utils


class NpRangeInfoTests(absltest.TestCase):

  def test_floating(self):
    expected_min = np.finfo(np.float32).min
    actual_min = tensor_spec_utils._np_range_info(np.float32).min
    self.assertEqual(expected_min, actual_min)

  def test_integer(self):
    actual_min = tensor_spec_utils._np_range_info(np.uint32).min
    self.assertEqual(0, actual_min)

  def test_string_gives_error(self):
    with self.assertRaisesRegex(ValueError, 'numpy.str_'):
      _ = tensor_spec_utils._np_range_info(np.str_).min


class BoundsTests(absltest.TestCase):

  def test_unbounded_unsigned(self):
    tensor_spec = dm_env_rpc_pb2.TensorSpec()
    tensor_spec.dtype = dm_env_rpc_pb2.DataType.UINT32
    bounds = tensor_spec_utils.bounds(tensor_spec)
    self.assertEqual((0, 2**32-1), bounds)

  def test_unbounded_signed(self):
    tensor_spec = dm_env_rpc_pb2.TensorSpec()
    tensor_spec.dtype = dm_env_rpc_pb2.DataType.INT32
    bounds = tensor_spec_utils.bounds(tensor_spec)
    self.assertEqual((-2**31, 2**31-1), bounds)

  def test_min_n_shape(self):
    minimum = np.array([[1, 2], [3, 4]])

    tensor_spec = dm_env_rpc_pb2.TensorSpec()
    tensor_spec.dtype = dm_env_rpc_pb2.DataType.UINT32
    tensor_spec.min.uint32s.array[:] = minimum.flatten().data.tolist()
    tensor_spec.shape[:] = minimum.shape
    bounds = tensor_spec_utils.bounds(tensor_spec)
    np.testing.assert_array_equal(minimum, bounds.min)
    np.testing.assert_array_equal(np.full(minimum.shape, 2**32 - 1), bounds.max)

  def test_max_n_shape(self):
    maximum = np.array([[1, 2], [3, 4]])

    tensor_spec = dm_env_rpc_pb2.TensorSpec()
    tensor_spec.dtype = dm_env_rpc_pb2.DataType.UINT32
    tensor_spec.max.uint32s.array[:] = maximum.flatten().data.tolist()
    tensor_spec.shape[:] = maximum.shape
    bounds = tensor_spec_utils.bounds(tensor_spec)
    np.testing.assert_array_equal(np.full(maximum.shape, 0), bounds.min)
    np.testing.assert_array_equal(maximum, bounds.max)

  def test_invalid_min_shape(self):
    tensor_spec = dm_env_rpc_pb2.TensorSpec()
    tensor_spec.dtype = dm_env_rpc_pb2.DataType.UINT32
    tensor_spec.min.uint32s.array[:] = [1, 2]
    with self.assertRaisesRegex(ValueError,
                                'Scalar tensors must have exactly 1 element.*'):
      tensor_spec_utils.bounds(tensor_spec)

  def test_invalid_max_shape(self):
    tensor_spec = dm_env_rpc_pb2.TensorSpec()
    tensor_spec.dtype = dm_env_rpc_pb2.DataType.UINT32
    tensor_spec.max.uint32s.array[:] = [1, 2]
    tensor_spec.shape[:] = (2, 2)
    with self.assertRaisesRegex(ValueError,
                                'cannot reshape array of size .* into shape.*'):
      tensor_spec_utils.bounds(tensor_spec)

  def test_min(self):
    tensor_spec = dm_env_rpc_pb2.TensorSpec()
    tensor_spec.dtype = dm_env_rpc_pb2.DataType.UINT32
    tensor_spec.min.uint32s.array[:] = [1]
    bounds = tensor_spec_utils.bounds(tensor_spec)
    self.assertEqual((1, 2**32-1), bounds)

  def test_max(self):
    tensor_spec = dm_env_rpc_pb2.TensorSpec()
    tensor_spec.dtype = dm_env_rpc_pb2.DataType.UINT32
    tensor_spec.max.uint32s.array[:] = [1]
    bounds = tensor_spec_utils.bounds(tensor_spec)
    self.assertEqual((0, 1), bounds)

  def test_min_and_max(self):
    tensor_spec = dm_env_rpc_pb2.TensorSpec()
    tensor_spec.dtype = dm_env_rpc_pb2.DataType.INT32
    tensor_spec.min.int32s.array[:] = [-1]
    tensor_spec.max.int32s.array[:] = [1]
    bounds = tensor_spec_utils.bounds(tensor_spec)
    self.assertEqual((-1, 1), bounds)

  def test_broadcast_var_shape(self):
    tensor_spec = dm_env_rpc_pb2.TensorSpec()
    tensor_spec.dtype = dm_env_rpc_pb2.DataType.INT32
    tensor_spec.min.int32s.array[:] = [-1]
    tensor_spec.max.int32s.array[:] = [1]
    tensor_spec.shape[:] = (-1,)
    bounds = tensor_spec_utils.bounds(tensor_spec)
    self.assertEqual((-1, 1), bounds)

  def test_invalid_min_var_shape(self):
    tensor_spec = dm_env_rpc_pb2.TensorSpec()
    tensor_spec.dtype = dm_env_rpc_pb2.DataType.INT32
    tensor_spec.min.int32s.array[:] = [-1, -1]
    tensor_spec.max.int32s.array[:] = [1]
    tensor_spec.shape[:] = (-1,)
    with self.assertRaisesRegex(
        ValueError, "TensorSpec's with variable length shapes "
        'can only have scalar ranges.'):
      tensor_spec_utils.bounds(tensor_spec)

  def test_min_broadcast(self):
    tensor_spec = dm_env_rpc_pb2.TensorSpec()
    tensor_spec.dtype = dm_env_rpc_pb2.DataType.UINT32
    tensor_spec.min.uint32s.array[:] = [1]
    tensor_spec.shape[:] = (2, 2)
    bounds = tensor_spec_utils.bounds(tensor_spec)
    np.testing.assert_array_equal(np.full(tensor_spec.shape, 1), bounds.min)
    np.testing.assert_array_equal(
        np.full(tensor_spec.shape, 2**32 - 1), bounds.max)

  def test_max_broadcast(self):
    tensor_spec = dm_env_rpc_pb2.TensorSpec()
    tensor_spec.dtype = dm_env_rpc_pb2.DataType.UINT32
    tensor_spec.max.uint32s.array[:] = [1]
    tensor_spec.shape[:] = (2, 2)
    bounds = tensor_spec_utils.bounds(tensor_spec)
    np.testing.assert_array_equal(np.full(tensor_spec.shape, 0), bounds.min)
    np.testing.assert_array_equal(np.full(tensor_spec.shape, 1), bounds.max)

  def test_min_max_broadcast(self):
    tensor_spec = dm_env_rpc_pb2.TensorSpec()
    tensor_spec.dtype = dm_env_rpc_pb2.DataType.UINT32
    tensor_spec.min.uint32s.array[:] = [1]
    tensor_spec.max.uint32s.array[:] = [2]
    tensor_spec.shape[:] = (4,)
    bounds = tensor_spec_utils.bounds(tensor_spec)
    np.testing.assert_array_equal(np.full(tensor_spec.shape, 1), bounds.min)
    np.testing.assert_array_equal(np.full(tensor_spec.shape, 2), bounds.max)

  def test_min_mismatches_type_raises_error(self):
    tensor_spec = dm_env_rpc_pb2.TensorSpec()
    tensor_spec.dtype = dm_env_rpc_pb2.DataType.UINT32
    tensor_spec.min.int32s.array[:] = [1]
    tensor_spec.name = 'foo'
    with self.assertRaisesRegex(ValueError, 'foo.*uint32.*min.*int32'):
      tensor_spec_utils.bounds(tensor_spec)

  def test_max_mismatches_type_raises_error(self):
    tensor_spec = dm_env_rpc_pb2.TensorSpec()
    tensor_spec.dtype = dm_env_rpc_pb2.DataType.UINT32
    tensor_spec.max.int32s.array[:] = [1]
    tensor_spec.name = 'foo'
    with self.assertRaisesRegex(ValueError, 'foo.*uint32.*max.*int32'):
      tensor_spec_utils.bounds(tensor_spec)

  def test_nonnumeric_type_raises_error(self):
    tensor_spec = dm_env_rpc_pb2.TensorSpec()
    tensor_spec.dtype = dm_env_rpc_pb2.DataType.STRING
    tensor_spec.max.int32s.array[:] = [1]
    tensor_spec.name = 'foo'
    with self.assertRaisesRegex(ValueError, 'foo.*non-numeric.*string'):
      tensor_spec_utils.bounds(tensor_spec)

  def test_max_0_stays_0(self):
    tensor_spec = dm_env_rpc_pb2.TensorSpec()
    tensor_spec.dtype = dm_env_rpc_pb2.DataType.INT8
    tensor_utils._BytesWrapper(tensor_spec.max.int8s, signed=False)[:] = [0]
    tensor_spec.name = 'foo'
    self.assertEqual((-128, 0), tensor_spec_utils.bounds(tensor_spec))

  def test_min_0_stays_0(self):
    tensor_spec = dm_env_rpc_pb2.TensorSpec()
    tensor_spec.dtype = dm_env_rpc_pb2.DataType.INT8
    tensor_utils._BytesWrapper(tensor_spec.min.int8s, signed=False)[:] = [0]
    tensor_spec.name = 'foo'
    self.assertEqual((0, 127), tensor_spec_utils.bounds(tensor_spec))

  def test_max_less_than_min_raises_error(self):
    tensor_spec = dm_env_rpc_pb2.TensorSpec()
    tensor_spec.dtype = dm_env_rpc_pb2.DataType.INT32
    tensor_spec.max.int32s.array[:] = [-1]
    tensor_spec.min.int32s.array[:] = [1]
    tensor_spec.name = 'foo'
    with self.assertRaisesRegex(ValueError, 'foo.*min 1.*max -1'):
      tensor_spec_utils.bounds(tensor_spec)

  def test_min_legacy(self):
    tensor_spec = dm_env_rpc_pb2.TensorSpec()
    tensor_spec.dtype = dm_env_rpc_pb2.DataType.UINT32
    tensor_spec.min.uint32 = 1
    bounds = tensor_spec_utils.bounds(tensor_spec)
    self.assertEqual((1, 2**32 - 1), bounds)

  def test_max_legacy(self):
    tensor_spec = dm_env_rpc_pb2.TensorSpec()
    tensor_spec.dtype = dm_env_rpc_pb2.DataType.UINT32
    tensor_spec.max.uint32 = 1
    bounds = tensor_spec_utils.bounds(tensor_spec)
    self.assertEqual((0, 1), bounds)

  def test_min_and_max_legacy(self):
    tensor_spec = dm_env_rpc_pb2.TensorSpec()
    tensor_spec.dtype = dm_env_rpc_pb2.DataType.INT32
    tensor_spec.min.int32 = -1
    tensor_spec.max.int32 = 1
    bounds = tensor_spec_utils.bounds(tensor_spec)
    self.assertEqual((-1, 1), bounds)

  def test_max_larger_than_dtype_can_support_raises_error_legacy(self):
    tensor_spec = dm_env_rpc_pb2.TensorSpec()
    tensor_spec.dtype = dm_env_rpc_pb2.DataType.INT8
    tensor_spec.max.int8 = 500
    tensor_spec.name = 'foo'
    with self.assertRaisesRegex(ValueError, 'foo.*-128, 500.*int8.*-128, 127'):
      tensor_spec_utils.bounds(tensor_spec)

  def test_min_larger_than_dtype_can_support_raises_error(self):
    tensor_spec = dm_env_rpc_pb2.TensorSpec()
    tensor_spec.dtype = dm_env_rpc_pb2.DataType.INT8
    tensor_spec.min.int8 = -500
    tensor_spec.name = 'foo'
    with self.assertRaisesRegex(ValueError, 'foo.*-500, 127.*int8.*-128, 127'):
      tensor_spec_utils.bounds(tensor_spec)


if __name__ == '__main__':
  absltest.main()
