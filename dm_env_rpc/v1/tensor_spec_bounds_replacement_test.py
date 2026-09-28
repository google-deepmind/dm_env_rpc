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
"""Regression tests for replacing existing TensorSpec bounds."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from dm_env_rpc.v1 import dm_env_rpc_pb2
from dm_env_rpc.v1 import tensor_spec_utils
from dm_env_rpc.v1 import tensor_utils

_DTYPES = (np.float32, np.float64, np.int8, np.int32, np.int64, np.uint8,
           np.uint32, np.uint64)


class BoundsReplacementTest(parameterized.TestCase):

  def make_spec(self, dtype, shape):
    return dm_env_rpc_pb2.TensorSpec(
        name='observation',
        shape=shape,
        dtype=tensor_utils.np_type_to_data_type(dtype))

  def assert_replacement(self, spec, minimum, maximum):
    expected = self.make_spec(tensor_utils.data_type_to_np_type(spec.dtype),
                              tuple(spec.shape))
    tensor_spec_utils.set_bounds(expected, minimum, maximum)
    self.assertIsNone(tensor_spec_utils.set_bounds(spec, minimum, maximum))
    self.assertEqual(spec, expected)
    self.assertEqual(spec.SerializeToString(), expected.SerializeToString())
    actual_bounds = tensor_spec_utils.bounds(spec)
    expected_bounds = tensor_spec_utils.bounds(expected)
    np.testing.assert_array_equal(actual_bounds.min, expected_bounds.min)
    np.testing.assert_array_equal(actual_bounds.max, expected_bounds.max)

  @parameterized.product(dtype=_DTYPES, shape=((), (2,)))
  def test_repeated_updates_replace_values(self, dtype, shape):
    spec = self.make_spec(dtype, shape)
    for lower, upper in ((1, 9), (2, 8), (2, 8), (0, 7)):
      minimum = np.full(shape, lower, dtype=dtype)
      maximum = np.full(shape, upper, dtype=dtype)
      self.assert_replacement(spec, minimum, maximum)
      np.testing.assert_array_equal(minimum, np.full(shape, lower, dtype=dtype))
      np.testing.assert_array_equal(maximum, np.full(shape, upper, dtype=dtype))

  @parameterized.product(dtype=(np.int32, np.float64),
                         vector_first=(False, True))
  def test_scalar_and_vector_bounds_can_replace_each_other(
      self, dtype, vector_first):
    spec = self.make_spec(dtype, (2,))
    shapes = [(2,), ()] if vector_first else [(), (2,)]
    for shape in shapes:
      self.assert_replacement(spec, np.full(shape, 1, dtype=dtype),
                              np.full(shape, 6, dtype=dtype))

  @parameterized.product(dtype=(np.int8, np.int32),
                         present=((False, False), (False, True), (True, False)))
  def test_clearing_either_bound_preserves_the_other_replacement(
      self, dtype, present):
    spec = self.make_spec(dtype, ())
    tensor_spec_utils.set_bounds(spec, dtype(1), dtype(5))
    minimum = dtype(2) if present[0] else None
    maximum = dtype(4) if present[1] else None
    self.assert_replacement(spec, minimum, maximum)
    self.assertEqual(spec.HasField('min'), present[0])
    self.assertEqual(spec.HasField('max'), present[1])

  @parameterized.parameters(([1, 2, 3], 5), (1, [4, 5, 6]), ([5, 1], [2, 3]),
                            (-(2**40), 5), (1, 2**40))
  def test_failed_validation_does_not_clear_existing_bounds(
      self, minimum, maximum):
    spec = self.make_spec(np.int32, (2,))
    tensor_spec_utils.set_bounds(spec, [1, 2], [5, 6])
    original = spec.SerializeToString()
    with self.assertRaises(ValueError):
      tensor_spec_utils.set_bounds(spec, minimum, maximum)
    self.assertEqual(spec.SerializeToString(), original)

  def test_variable_length_spec_retains_scalar_bounds(self):
    spec = self.make_spec(np.int32, (-1, 2))
    for minimum, maximum in ((1, 8), (2, 5)):
      self.assert_replacement(spec, minimum, maximum)
      self.assertEqual(len(spec.min.int32s.array), 1)
      self.assertEqual(len(spec.max.int32s.array), 1)


if __name__ == '__main__':
  absltest.main()
