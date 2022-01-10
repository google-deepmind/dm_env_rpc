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

from google.protobuf import any_pb2
from google.protobuf import struct_pb2
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

  def test_pack_scalar_protos(self):
    scalar = struct_pb2.Value(string_value='my message')
    tensor = tensor_utils.pack_tensor(scalar)
    self.assertEqual([], tensor.shape)
    self.assertLen(tensor.protos.array, 1)
    unpacked = struct_pb2.Value()
    self.assertTrue(tensor.protos.array[0].Unpack(unpacked))
    self.assertEqual(scalar, unpacked)

  def test_pack_scalar_any_proto(self):
    scalar = struct_pb2.Value(string_value='my message')
    scalar_any = any_pb2.Any()
    scalar_any.Pack(scalar)
    tensor = tensor_utils.pack_tensor(scalar_any)
    self.assertEqual([], tensor.shape)
    self.assertLen(tensor.protos.array, 1)
    unpacked = struct_pb2.Value()
    self.assertTrue(tensor.protos.array[0].Unpack(unpacked))
    self.assertEqual(scalar, unpacked)

  @parameterized.parameters(
      (25, np.float32, 'floats'),
      (25, np.float64, 'doubles'),
      (25, np.int32, 'int32s'),
      (25, np.int64, 'int64s'),
      (25, np.uint32, 'uint32s'),
      (25, np.uint64, 'uint64s'),
      (2**64-1, np.uint64, 'uint64s'),
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

  @parameterized.parameters(
      ([], None, 'doubles'),
      ([], np.int64, 'int64s'),
      ([1, 2, 3], None, 'int64s'),
      ([1, 2, 3], np.int32, 'int32s'),
  )
  def test_pack_override_dtype(self, value, dtype, expected_payload):
    tensor = tensor_utils.pack_tensor(value, dtype=dtype)
    array = np.asarray(value, dtype)
    self.assertEqual(expected_payload, tensor.WhichOneof('payload'))
    packed_array = getattr(tensor, expected_payload).array
    np.testing.assert_array_equal(array, packed_array)

  def test_pack_proto_arrays(self):
    array = np.array([
        struct_pb2.Value(string_value=message)
        for message in ['foo', 'bar']
    ])
    tensor = tensor_utils.pack_tensor(array)
    self.assertEqual([2], tensor.shape)
    unpacked = struct_pb2.Value()
    tensor.protos.array[0].Unpack(unpacked)
    self.assertEqual(array[0], unpacked)
    tensor.protos.array[1].Unpack(unpacked)
    self.assertEqual(array[1], unpacked)

  def test_pack_mixed_proto_array_fails(self):
    with self.assertRaisesRegex(ValueError, 'not recognized'):
      tensor_utils.pack_tensor(np.array([struct_pb2.Value(), 1, 2, 3]))

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

  @parameterized.parameters(
      (['foo', 'bar'], np.str_),
      ('baz', dm_env_rpc_pb2.DataType.STRING),
      (['foobar'], np.array(['foobar']).dtype),
  )
  def test_np_object_strings(self, value, dtype):
    object_array = np.array(value, dtype=np.object)
    tensor = tensor_utils.pack_tensor(object_array, dtype=dtype)
    self.assertEqual(list(object_array.shape), tensor.shape)
    self.assertTrue(tensor.HasField('strings'))

  def test_np_object_strings_no_dtype_raises_exception(self):
    with self.assertRaises(ValueError):
      tensor_utils.pack_tensor(np.array(['foo'], dtype=np.object))

  @parameterized.parameters(
      (['foo', 42, 'bar'],),
      ([1, 2, 3],),
  )
  def test_np_object_to_strings_fail(self, bad_element):
    with self.assertRaisesRegex(TypeError,
                                'not all elements are Python string types'):
      tensor_utils.pack_tensor(
          np.array(bad_element, dtype=np.object), dtype=np.str_)

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

  def test_unpack_scalar_proto(self):
    scalar = struct_pb2.Value(string_value='my message')
    tensor = tensor_utils.pack_tensor(scalar)

    unpacked = struct_pb2.Value()
    tensor_utils.unpack_tensor(tensor).Unpack(unpacked)
    self.assertEqual(scalar, unpacked)

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

  def test_unpack_proto_arrays(self):
    array = np.array([
        struct_pb2.Value(string_value=message)
        for message in ['foo', 'bar']
    ])
    tensor = tensor_utils.pack_tensor(array)
    round_trip = tensor_utils.unpack_tensor(tensor)

    unpacked = struct_pb2.Value()
    round_trip[0].Unpack(unpacked)
    self.assertEqual(array[0], unpacked)
    round_trip[1].Unpack(unpacked)
    self.assertEqual(array[1], unpacked)

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
    with self.assertRaisesRegex(ValueError, 'cannot reshape array'):
      tensor_utils.unpack_tensor(tensor)

  def test_too_many_elements(self):
    tensor = dm_env_rpc_pb2.Tensor()
    tensor.floats.array[:] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    tensor.shape[:] = [2, 4]
    with self.assertRaisesRegex(ValueError, 'cannot reshape array'):
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
    tensor.uint8s.array = b'\x01'
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
    with self.assertRaisesRegex(ValueError, 'one unknown dimension'):
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
    with self.assertRaisesRegex(TypeError, 'type foo'):
      tensor_utils.unpack_tensor(tensor)

  def test_scalar_with_too_many_elements_raises_error(self):
    tensor = dm_env_rpc_pb2.Tensor()
    tensor.int32s.array[:] = [1, 2, 3]
    with self.assertRaisesRegex(ValueError, '3 element'):
      tensor_utils.unpack_tensor(tensor)


class GetTensorTypeTests(absltest.TestCase):

  def test_float(self):
    tensor = tensor_utils.pack_tensor(1.25)
    self.assertEqual(np.float64, tensor_utils.get_tensor_type(tensor))

  def test_unknown_tensor_type(self):
    mock_tensor = mock.MagicMock()
    mock_tensor.WhichOneof.return_value = 'foo'
    with self.assertRaisesRegex(TypeError, 'foo'):
      tensor_utils.get_tensor_type(mock_tensor)


class DataTypeToNpTypeTests(absltest.TestCase):

  def test_float(self):
    self.assertEqual(
        np.float32,
        tensor_utils.data_type_to_np_type(dm_env_rpc_pb2.DataType.FLOAT))

  def test_empty_object_list(self):
    tensor = tensor_utils.pack_tensor(np.array([], dtype=np.object))
    self.assertEqual([0], tensor.shape)

  def test_unknown_type(self):
    with self.assertRaises(TypeError):
      tensor_utils.data_type_to_np_type(30)  # pytype: disable=wrong-arg-types


class NpTypeToDataTypeTests(absltest.TestCase):

  def test_float32(self):
    self.assertEqual(
        dm_env_rpc_pb2.DataType.FLOAT,
        tensor_utils.np_type_to_data_type(np.float32))

  def test_int32(self):
    self.assertEqual(
        dm_env_rpc_pb2.DataType.INT32,
        tensor_utils.np_type_to_data_type(np.int32))

  def test_dtype(self):
    self.assertEqual(
        dm_env_rpc_pb2.DataType.INT32,
        tensor_utils.np_type_to_data_type(np.dtype(np.int32)))

  def test_unknown_type(self):
    with self.assertRaisesRegex(TypeError, 'dm_env_rpc DataType.*complex64'):
      tensor_utils.np_type_to_data_type(np.complex64)


class GetPackerTests(absltest.TestCase):

  def test_cannot_get_packer_for_invalid_type(self):
    with self.assertRaisesRegex(TypeError, 'complex64'):
      tensor_utils.get_packer(np.complex64)

  def test_can_pack(self):
    packer = tensor_utils.get_packer(np.int32)
    tensor = dm_env_rpc_pb2.Tensor()
    packer.pack(tensor, np.asarray([1, 2, 3], dtype=np.int32))
    self.assertEqual([1, 2, 3], tensor.int32s.array)

  def test_can_unpack(self):
    packer = tensor_utils.get_packer(np.int32)
    tensor = dm_env_rpc_pb2.Tensor()
    tensor.int32s.array[:] = [1, 2, 3]
    np.testing.assert_array_equal([1, 2, 3], packer.unpack(tensor))


if __name__ == '__main__':
  absltest.main()
