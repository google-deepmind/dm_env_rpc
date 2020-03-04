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
"""Tests for SpecManager class."""

from absl.testing import absltest
import numpy as np

from dm_env_rpc.v1 import dm_env_rpc_pb2
from dm_env_rpc.v1 import spec_manager
from dm_env_rpc.v1 import tensor_utils


class SpecManagerTests(absltest.TestCase):

  def setUp(self):
    super(SpecManagerTests, self).setUp()
    specs = {
        54:
            dm_env_rpc_pb2.TensorSpec(
                name='fuzz', shape=[2], dtype=dm_env_rpc_pb2.DataType.FLOAT),
        55:
            dm_env_rpc_pb2.TensorSpec(
                name='foo', shape=[3], dtype=dm_env_rpc_pb2.DataType.INT32),
    }
    self._spec_manager = spec_manager.SpecManager(specs)

  def test_name_to_uid(self):
    self.assertEqual(55, self._spec_manager.name_to_uid('foo'))

  def test_name_to_uid_no_such_name(self):
    with self.assertRaisesRegex(KeyError, 'bar'):
      self._spec_manager.name_to_uid('bar')

  def test_name_to_spec(self):
    spec = self._spec_manager.name_to_spec('foo')
    self.assertEqual([3], spec.shape)

  def test_name_to_spec_no_such_name(self):
    with self.assertRaisesRegex(KeyError, 'bar'):
      self._spec_manager.name_to_spec('bar')

  def test_uid_to_name(self):
    self.assertEqual('foo', self._spec_manager.uid_to_name(55))

  def test_uid_to_name_no_such_uid(self):
    with self.assertRaisesRegex(KeyError, '56'):
      self._spec_manager.uid_to_name(56)

  def test_names(self):
    self.assertEqual(set(['foo', 'fuzz']), self._spec_manager.names())

  def test_uids(self):
    self.assertEqual(set([54, 55]), self._spec_manager.uids())

  def test_uid_to_spec(self):
    spec = self._spec_manager.uid_to_spec(54)
    self.assertEqual([2], spec.shape)

  def test_pack(self):
    packed = self._spec_manager.pack({'fuzz': [1.0, 2.0], 'foo': [3, 4, 5]})
    expected = {
        54: tensor_utils.pack_tensor([1.0, 2.0], dtype=np.float32),
        55: tensor_utils.pack_tensor([3, 4, 5], dtype=np.int32),
    }
    self.assertDictEqual(expected, packed)

  def test_partial_pack(self):
    packed = self._spec_manager.pack({
        'fuzz': [1.0, 2.0],
    })
    expected = {
        54: tensor_utils.pack_tensor([1.0, 2.0], dtype=np.float32),
    }
    self.assertDictEqual(expected, packed)

  def test_pack_unknown_key_raises_error(self):
    with self.assertRaisesRegex(KeyError, 'buzz'):
      self._spec_manager.pack({'buzz': 'hello'})

  def test_pack_wrong_shape_raises_error(self):
    with self.assertRaisesRegex(ValueError, 'shape'):
      self._spec_manager.pack({'foo': [1, 2]})

  def test_pack_wrong_dtype_raises_error(self):
    with self.assertRaisesRegex(TypeError, 'int32'):
      self._spec_manager.pack({'foo': 'hello'})

  def test_pack_cast_float_to_int_raises_error(self):
    with self.assertRaisesRegex(TypeError, 'int32'):
      self._spec_manager.pack({'foo': [0.5, 1.0, 1]})

  def test_pack_cast_int_to_float_is_ok(self):
    packed = self._spec_manager.pack({'fuzz': [1, 2]})
    self.assertEqual([1.0, 2.0], packed[54].floats.array)

  def test_unpack(self):
    unpacked = self._spec_manager.unpack({
        54: tensor_utils.pack_tensor([1.0, 2.0], dtype=np.float32),
        55: tensor_utils.pack_tensor([3, 4, 5], dtype=np.int32),
    })
    self.assertLen(unpacked, 2)
    np.testing.assert_array_equal(np.asarray([1.0, 2.0]), unpacked['fuzz'])
    np.testing.assert_array_equal(np.asarray([3, 4, 5]), unpacked['foo'])

  def test_partial_unpack(self):
    unpacked = self._spec_manager.unpack({
        54: tensor_utils.pack_tensor([1.0, 2.0], dtype=np.float32),
    })
    self.assertLen(unpacked, 1)
    np.testing.assert_array_equal(np.asarray([1.0, 2.0]), unpacked['fuzz'])

  def test_unpack_unknown_uid_raises_error(self):
    with self.assertRaisesRegex(KeyError, '53'):
      self._spec_manager.unpack({53: tensor_utils.pack_tensor('foo')})

  def test_unpack_wrong_shape_raises_error(self):
    with self.assertRaisesRegex(ValueError, 'shape'):
      self._spec_manager.unpack({55: tensor_utils.pack_tensor([1, 2])})

  def test_unpack_wrong_type_raises_error(self):
    with self.assertRaisesRegex(ValueError, 'dtype'):
      self._spec_manager.unpack(
          {55: tensor_utils.pack_tensor([1, 2, 3], dtype=np.float32)})


class SpecManagerConstructorTests(absltest.TestCase):

  def test_duplicate_names_raise_error(self):
    specs = {
        54:
            dm_env_rpc_pb2.TensorSpec(
                name='fuzz', shape=[3], dtype=dm_env_rpc_pb2.DataType.FLOAT),
        55:
            dm_env_rpc_pb2.TensorSpec(
                name='fuzz', shape=[2], dtype=dm_env_rpc_pb2.DataType.FLOAT),
    }
    with self.assertRaisesRegex(ValueError, 'duplicate name'):
      spec_manager.SpecManager(specs)


if __name__ == '__main__':
  absltest.main()
