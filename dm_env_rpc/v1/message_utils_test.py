# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for dm_env_rpc/message_utils."""

from absl.testing import absltest

from google.rpc import status_pb2
from dm_env_rpc.v1 import dm_env_rpc_pb2
from dm_env_rpc.v1 import error
from dm_env_rpc.v1 import message_utils
from dm_env_rpc.v1 import tensor_utils


_CREATE_WORLD_REQUEST = dm_env_rpc_pb2.CreateWorldRequest(
    settings={'foo': tensor_utils.pack_tensor('bar')})
_CREATE_WORLD_RESPONSE = dm_env_rpc_pb2.CreateWorldResponse(world_name='qux')
_CREATE_WORLD_ENVIRONMENT_RESPONSE = dm_env_rpc_pb2.EnvironmentResponse(
    create_world=_CREATE_WORLD_RESPONSE)


class MessageUtilsTests(absltest.TestCase):

  def test_pack_create_world_request(self):
    environment_request, field_name = message_utils.pack_environment_request(
        _CREATE_WORLD_REQUEST)
    self.assertEqual(field_name, 'create_world')
    self.assertEqual(environment_request.WhichOneof('payload'),
                     'create_world')
    self.assertEqual(environment_request.create_world,
                     _CREATE_WORLD_REQUEST)

  def test_unpack_create_world_response(self):
    response = message_utils.unpack_environment_response(
        _CREATE_WORLD_ENVIRONMENT_RESPONSE, 'create_world')
    self.assertEqual(response, _CREATE_WORLD_RESPONSE)

  def test_unpack_error_response(self):
    with self.assertRaisesRegex(error.DmEnvRpcError, 'A test error.'):
      message_utils.unpack_environment_response(
          dm_env_rpc_pb2.EnvironmentResponse(
              error=status_pb2.Status(message='A test error.')),
          'create_world')

  def test_unpack_incorrect_response(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'Unexpected response message! expected: create_world, actual: '
        'leave_world'):
      message_utils.unpack_environment_response(
          dm_env_rpc_pb2.EnvironmentResponse(
              leave_world=dm_env_rpc_pb2.LeaveWorldResponse()),
          'create_world')


if __name__ == '__main__':
  absltest.main()
