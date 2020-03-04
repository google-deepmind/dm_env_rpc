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

import struct

from absl.testing import absltest
from absl.testing import parameterized
import mock
import numpy as np

from dm_env_rpc.v1 import dm_env_rpc_pb2
from dm_env_rpc.v1 import tensor_utils


class PackTensorTests(parameterized.TestCase):

  @parameterized.parameters(
      (np.float32(2.5), 'floats'),
      (2.5, 'doubles'),
      (np.int32(-25), 'int32s'),
      (np.int64(-25), 'int64s'),
      (np.frombuffer(b'\xF0\xF1\xF2\xF3', np.uint32)[0], 'uint32s'),
      (np.frombuffer(b'\xF0\xF1\xF2\xF3\xF4\xF5\xF6\xF7',
                     np.uint64)[0], 'uint64s'),
      (True, 'bools'),
      (False, 'bools'),
      ('foo', 'strings'),
  )
  def test_pack_scalars(self, scalar, expected_payload):
    tensor = tensor_utils.pack_tensor(scalar)
    self.assertEqual([], tensor.shape)
    self.assertEqual([scalar], getattr(tensor, expected_payload).array)

  @parameterized.parameters(
      (np.int8(-25), 'b', 'int8s'),
      (np.uint8(250), 'B', 'uint8s'),
  )
  def test_pack_scalar_bytes(self, scalar, fmt, expected_payload):
    tensor = tensor_utils.pack_tensor(scalar)
    self.assertEqual([], tensor.shape)
    actual = struct.unpack(fmt, getattr(tensor, expected_payload).array)
    self.assertEqual(scalar, actual)

  @parameterized.parameters(
      (25, np.float32, 'floats'),
      (25, np.float64, 'doubles'),
      (25, np.int32, 'int32s'),
      (25, np.int64, 'int64s'),
      (25, np.uint32, 'uint32s'),
      (25, np.uint64, 'uint64s'),
      (True, np.bool, 'bools'),
      (False, np.bool, 'bools'),
      ('foo', np.str, 'strings'),
  )
  def test_pack_scalars_specific_dtype(self, scalar, dtype, expected_payload):
    tensor = tensor_utils.pack_tensor(scalar, dtype)
    self.assertEqual([], tensor.shape)
    self.assertEqual([scalar], getattr(tensor, expected_payload).array)

  def test_pack_with_dm_env_rpc_data_type(self):
    tensor = tensor_utils.pack_tensor([5], dm_env_rpc_pb2.DataType.FLOAT)
    self.assertEqual([5], tensor.floats.array)

  @parameterized.parameters(
      ([np.int8(-25), np.int8(-23)], '2b', 'int8s'),
      ([np.uint8(249), np.uint8(250)], '2B', 'uint8s'),
  )
  def test_pack_bytes_array(self, scalar, fmt, expected_payload):
    tensor = tensor_utils.pack_tensor(scalar)
    self.assertEqual([2], tensor.shape)
    actual = struct.unpack(fmt, getattr(tensor, expected_payload).array)
    np.testing.assert_array_equal(scalar, actual)

  @parameterized.parameters(
      (np.array([1.0, 2.0], dtype=np.float32), 'floats'),
      (np.array([1.0, 2.0], dtype=np.float64), 'doubles'),
      ([1.0, 2.0], 'doubles'),
      (np.array([1, 2], dtype=np.int32), 'int32s'),
      (np.array([1, 2], dtype=np.int64), 'int64s'),
      (np.array([1, 2], dtype=np.uint32), 'uint32s'),
      (np.array([1, 2], dtype=np.uint64), 'uint64s'),
      ([True, False], 'bools'),
      (np.array([True, False]), 'bools'),
      (['foo', 'bar'], 'strings'),
  )
  def test_pack_arrays(self, array, expected_payload):
    tensor = tensor_utils.pack_tensor(array)
    self.assertEqual([2], tensor.shape)
    packed_array = getattr(tensor, expected_payload).array
    np.testing.assert_array_equal(array, packed_array)

  def test_packed_rowmajor(self):
    array2d = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    tensor = tensor_utils.pack_tensor(array2d)
    self.assertEqual([3, 2], tensor.shape)
    np.testing.assert_array_equal([1, 2, 3, 4, 5, 6], tensor.int32s.array)

  def test_mixed_scalar_types_raises_exception(self):
    with self.assertRaises(TypeError):
      tensor_utils.pack_tensor(['hello!', 75], dtype=np.float32)

  def test_jagged_arrays_throw_exceptions(self):
    with self.assertRaises(ValueError):
      tensor_utils.pack_tensor([[1, 2], [3, 4, 5]])

  def test_class_instance_throw_exception(self):

    class Foo(object):
      pass

    with self.assertRaises(ValueError):
      tensor_utils.pack_tensor(Foo())

  def test_compress_integers_to_1_element_when_all_same(self):
    array = np.array([1, 1, 1, 1, 1, 1], dtype=np.uint32)
    packed = tensor_utils.pack_tensor(array, try_compress=True)
    self.assertEqual([6], packed.shape)
    self.assertEqual([1], packed.uint32s.array)

  def test_compress_floats_to_1_element_when_all_same(self):
    array = np.array([1.5, 1.5, 1.5, 1.5, 1.5, 1.5], dtype=np.float32)
    packed = tensor_utils.pack_tensor(array, try_compress=True)
    self.assertEqual([6], packed.shape)
    self.assertEqual([1.5], packed.floats.array)

  def test_compress_strings_to_1_element_when_all_same(self):
    array = np.array(['foo', 'foo', 'foo', 'foo'], dtype=np.str_)
    packed = tensor_utils.pack_tensor(array, try_compress=True)
    self.assertEqual([4], packed.shape)
    self.assertEqual(['foo'], packed.strings.array)

  def test_compress_multidimensional_arrays_to_1_element_when_all_same(self):
    array = np.array([[4, 4], [4, 4]], dtype=np.int32)
    packed = tensor_utils.pack_tensor(array, try_compress=True)
    self.assertEqual([2, 2], packed.shape)
    self.assertEqual([4], packed.int32s.array)

  def test_doesnt_compress_if_not_asked_to(self):
    array = np.array([1, 1, 1, 1, 1, 1], dtype=np.uint32)
    packed = tensor_utils.pack_tensor(array)
    self.assertEqual([6], packed.shape)
    self.assertEqual([1, 1, 1, 1, 1, 1], packed.uint32s.array)

  def test_ask_to_compress_but_cant(self):
    array = np.array([1, 1, 2, 1, 1, 1], dtype=np.uint32)
    packed = tensor_utils.pack_tensor(array, try_compress=True)
    self.assertEqual([6], packed.shape)
    self.assertEqual([1, 1, 2, 1, 1, 1], packed.uint32s.array)


class UnpackTensorTests(parameterized.TestCase):

  @parameterized.parameters(
      np.float32(2.5),
      np.float64(2.5),
      np.int8(-25),
      np.int32(-25),
      np.int64(-25),
      np.uint8(250),
      np.frombuffer(b'\xF0\xF1\xF2\xF3', np.uint32)[0],
      np.frombuffer(b'\xF0\xF1\xF2\xF3\xF4\xF5\xF6\xF7', np.uint64)[0],
      True,
      False,
      'foo',
  )
  def test_unpack_scalars(self, scalar):
    tensor = tensor_utils.pack_tensor(scalar)
    round_trip = tensor_utils.unpack_tensor(tensor)
    self.assertEqual(scalar, round_trip)

  @parameterized.parameters(
      ([np.float32(2.5), np.float32(3.5)],),
      ([np.float64(2.5), np.float64(3.5)],),
      ([np.int8(-25), np.int8(-23)],),
      ([np.int32(-25), np.int32(-23)],),
      ([np.int64(-25), np.int64(-23)],),
      ([np.uint8(250), np.uint8(249)],),
      ([np.uint32(1), np.uint32(2)],),
      ([np.uint64(1), np.uint64(2)],),
      ([True, False],),
      (['foo', 'bar'],),
  )
  def test_unpack_arrays(self, array):
    tensor = tensor_utils.pack_tensor(array)
    round_trip = tensor_utils.unpack_tensor(tensor)
    np.testing.assert_array_equal(array, round_trip)

  def test_unpack_multidimensional_arrays(self):
    tensor = dm_env_rpc_pb2.Tensor()
    tensor.floats.array[:] = [1, 2, 3, 4, 5, 6, 7, 8]
    tensor.shape[:] = [2, 4]
    round_trip = tensor_utils.unpack_tensor(tensor)
    expected = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    np.testing.assert_array_equal(expected, round_trip)

  def test_too_few_elements(self):
    tensor = dm_env_rpc_pb2.Tensor()
    tensor.floats.array[:] = [1, 2, 3, 4]
    tensor.shape[:] = [2, 4]
    with self.assertRaisesRegexp(ValueError, 'cannot reshape array'):
      tensor_utils.unpack_tensor(tensor)

  def test_too_many_elements(self):
    tensor = dm_env_rpc_pb2.Tensor()
    tensor.floats.array[:] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    tensor.shape[:] = [2, 4]
    with self.assertRaisesRegexp(ValueError, 'cannot reshape array'):
      tensor_utils.unpack_tensor(tensor)

  def test_float_broadcasts_1_element_to_all_elements(self):
    tensor = dm_env_rpc_pb2.Tensor()
    tensor.floats.array[:] = [1]
    tensor.shape[:] = [4]
    unpacked = tensor_utils.unpack_tensor(tensor)
    expected = np.array([1, 1, 1, 1], dtype=np.float32)
    np.testing.assert_array_equal(expected, unpacked)

  def test_integer_broadcasts_1_element_to_all_elements(self):
    tensor = dm_env_rpc_pb2.Tensor()
    tensor.int32s.array[:] = [1]
    tensor.shape[:] = [4]
    unpacked = tensor_utils.unpack_tensor(tensor)
    expected = np.array([1, 1, 1, 1], dtype=np.int32)
    np.testing.assert_array_equal(expected, unpacked)

  def test_unsigned_integer_broadcasts_1_element_to_all_elements(self):
    tensor = dm_env_rpc_pb2.Tensor()
    tensor_utils._BytesWrapper(tensor.uint8s, signed=False)[:] = [1]
    tensor.shape[:] = [4]
    unpacked = tensor_utils.unpack_tensor(tensor)
    expected = np.array([1, 1, 1, 1], dtype=np.uint8)
    np.testing.assert_array_equal(expected, unpacked)

  def test_string_broadcasts_1_element_to_all_elements(self):
    tensor = dm_env_rpc_pb2.Tensor()
    tensor.strings.array[:] = ['foo']
    tensor.shape[:] = [4]
    unpacked = tensor_utils.unpack_tensor(tensor)
    expected = np.array(['foo', 'foo', 'foo', 'foo'], dtype=np.str_)
    np.testing.assert_array_equal(expected, unpacked)

  def test_broadcasts_to_multidimensional_arrays(self):
    tensor = dm_env_rpc_pb2.Tensor()
    tensor.int32s.array[:] = [4]
    tensor.shape[:] = [2, 2]
    unpacked = tensor_utils.unpack_tensor(tensor)
    expected = np.array([[4, 4], [4, 4]], dtype=np.int32)
    np.testing.assert_array_equal(expected, unpacked)

  def test_negative_dimension(self):
    tensor = dm_env_rpc_pb2.Tensor()
    tensor.int32s.array[:] = [1, 2, 3, 4]
    tensor.shape[:] = [-1]
    unpacked = tensor_utils.unpack_tensor(tensor)
    expected = np.array([1, 2, 3, 4], dtype=np.int32)
    np.testing.assert_array_equal(expected, unpacked)

  def test_negative_dimension_in_matrix(self):
    tensor = dm_env_rpc_pb2.Tensor()
    tensor.int32s.array[:] = [1, 2, 3, 4, 5, 6]
    tensor.shape[:] = [2, -1]
    unpacked = tensor_utils.unpack_tensor(tensor)
    expected = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    np.testing.assert_array_equal(expected, unpacked)

  def test_two_negative_dimensions_in_matrix(self):
    tensor = dm_env_rpc_pb2.Tensor()
    tensor.int32s.array[:] = [1, 2, 3, 4, 5, 6]
    tensor.shape[:] = [-1, -2]
    with self.assertRaisesRegexp(ValueError, 'one unknown dimension'):
      tensor_utils.unpack_tensor(tensor)

  def test_negative_dimension_single_element(self):
    tensor = dm_env_rpc_pb2.Tensor()
    tensor.int32s.array[:] = [1]
    tensor.shape[:] = [-1]
    unpacked = tensor_utils.unpack_tensor(tensor)
    expected = np.array([1], dtype=np.int32)
    np.testing.assert_array_equal(expected, unpacked)

  def test_unknown_type_raises_error(self):
    tensor = mock.MagicMock()
    tensor.WhichOneof.return_value = 'foo'
    with self.assertRaisesRegexp(TypeError, 'type foo'):
      tensor_utils.unpack_tensor(tensor)

  def test_scalar_with_too_many_elements_raises_error(self):
    tensor = dm_env_rpc_pb2.Tensor()
    tensor.int32s.array[:] = [1, 2, 3]
    with self.assertRaisesRegexp(ValueError, '3 element'):
      tensor_utils.unpack_tensor(tensor)


class GetTensorTypeTests(absltest.TestCase):

  def test_float(self):
    tensor = tensor_utils.pack_tensor(1.25)
    self.assertEqual(np.float64, tensor_utils.get_tensor_type(tensor))

  def test_unknown_tensor_type(self):
    mock_tensor = mock.MagicMock()
    mock_tensor.WhichOneof.return_value = 'foo'
    with self.assertRaisesRegexp(TypeError, 'foo'):
      tensor_utils.get_tensor_type(mock_tensor)


class GetTensorSpecTypeTests(absltest.TestCase):

  def test_float(self):
    self.assertEqual(
        np.float32,
        tensor_utils.data_type_to_np_type(dm_env_rpc_pb2.DataType.FLOAT))

  def test_proto_type(self):
    with self.assertRaises(TypeError):
      tensor_utils.data_type_to_np_type(dm_env_rpc_pb2.DataType.PROTO)

  def test_unknown_type(self):
    with self.assertRaises(TypeError):
      tensor_utils.data_type_to_np_type(30)


class BytesWrapperTests(absltest.TestCase):

  def test_unsupported_indexing_on_write_raises_error(self):
    tensor = dm_env_rpc_pb2.Tensor()
    wrapper = tensor_utils._BytesWrapper(tensor.uint8s, signed=False)
    with self.assertRaisesRegexp(ValueError, 'index'):
      wrapper[0] = 0


if __name__ == '__main__':
  absltest.main()
