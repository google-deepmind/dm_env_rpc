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
"""Tests for environment_stream.proto.

These aren't for testing functionality (it's assumed protobufs work) but for
testing/demonstrating how the protobufs would have to be used in code.
"""

from absl.testing import absltest
from dm_env_rpc.v1 import dm_env_rpc_pb2


class TensorTests(absltest.TestCase):

  def test_setting_tensor_data(self):
    tensor = dm_env_rpc_pb2.Tensor()
    tensor.floats.array[:] = [1, 2]

  def test_setting_tensor_data_with_wrong_type(self):
    tensor = dm_env_rpc_pb2.Tensor()
    with self.assertRaises(TypeError):
      tensor.floats.array[:] = ['hello!']  # pytype: disable=unsupported-operands

  def test_which_is_set(self):
    tensor = dm_env_rpc_pb2.Tensor()
    tensor.floats.array[:] = [1, 2]
    self.assertEqual('floats', tensor.WhichOneof('payload'))


class TensorSpec(absltest.TestCase):

  def test_setting_spec(self):
    tensor_spec = dm_env_rpc_pb2.TensorSpec()
    tensor_spec.name = 'Foo'
    tensor_spec.min.floats.array[:] = [0.0]
    tensor_spec.max.floats.array[:] = [0.0]
    tensor_spec.shape[:] = [2, 2]
    tensor_spec.dtype = dm_env_rpc_pb2.DataType.FLOAT


class JoinWorldResponse(absltest.TestCase):

  def test_setting_spec(self):
    response = dm_env_rpc_pb2.JoinWorldResponse()
    tensor_spec = response.specs.actions[1]
    tensor_spec.shape[:] = [1]
    tensor_spec.dtype = dm_env_rpc_pb2.DataType.FLOAT


if __name__ == '__main__':
  absltest.main()
