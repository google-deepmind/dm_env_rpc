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
from absl.testing import parameterized

from google.rpc import status_pb2
from dm_env_rpc.v1 import dm_env_rpc_pb2
from dm_env_rpc.v1 import error
from dm_env_rpc.v1 import message_utils
from dm_env_rpc.v1 import tensor_utils
from google.protobuf import any_pb2


_CREATE_WORLD_REQUEST = dm_env_rpc_pb2.CreateWorldRequest(
    settings={'foo': tensor_utils.pack_tensor('bar')})
_CREATE_WORLD_RESPONSE = dm_env_rpc_pb2.CreateWorldResponse(world_name='qux')
_CREATE_WORLD_ENVIRONMENT_RESPONSE = dm_env_rpc_pb2.EnvironmentResponse(
    create_world=_CREATE_WORLD_RESPONSE)

# Anything that's not a "native" rpc message is an extension.
_EXTENSION_TYPE = error.status_pb2.Status
_EXTENSION_MULTI_TYPE = (dm_env_rpc_pb2.EnvironmentResponse, _EXTENSION_TYPE)
_EXTENSION_MESSAGE = _EXTENSION_TYPE(code=42)
_PACKED_EXTENSION_MESSAGE = any_pb2.Any()
_PACKED_EXTENSION_MESSAGE.Pack(_EXTENSION_MESSAGE)


class MessageUtilsTests(parameterized.TestCase):

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

  @parameterized.named_parameters(
      dict(
          testcase_name='create_world_request_passes_through',
          message=_CREATE_WORLD_REQUEST,
          expected=_CREATE_WORLD_REQUEST,
      ),
      dict(
          testcase_name='packed_extension_message_passes_through',
          message=_PACKED_EXTENSION_MESSAGE,
          expected=_PACKED_EXTENSION_MESSAGE,
      ),
      dict(
          testcase_name='extension_message_is_packed',
          message=_EXTENSION_MESSAGE,
          expected=_PACKED_EXTENSION_MESSAGE,
      ),
  )
  def test_pack_request(self, message, expected):
    self.assertEqual(message_utils.pack_rpc_request(message), expected)

  @parameterized.named_parameters(
      dict(
          testcase_name='create_world_response',
          message=_CREATE_WORLD_RESPONSE,
          expected=_CREATE_WORLD_RESPONSE,
      ),
      dict(
          testcase_name='extension_message',
          message=_EXTENSION_MESSAGE,
          expected=_PACKED_EXTENSION_MESSAGE,
      ),
      dict(
          testcase_name='packed_extension_message',
          message=_PACKED_EXTENSION_MESSAGE,
          expected=_PACKED_EXTENSION_MESSAGE,
      ),
  )
  def test_pack_response(self, message, expected):
    self.assertEqual(message_utils.pack_rpc_response(message), expected)

  @parameterized.named_parameters(
      dict(
          testcase_name='create_world_request_passes_through',
          message=_CREATE_WORLD_REQUEST,
          extensions=_EXTENSION_TYPE,
          expected=_CREATE_WORLD_REQUEST,
      ),
      dict(
          testcase_name='extension_message_passes_through',
          message=_EXTENSION_MESSAGE,
          extensions=_EXTENSION_TYPE,
          expected=_EXTENSION_MESSAGE,
      ),
      dict(
          testcase_name='extension_message_passes_through_with_multi',
          message=_EXTENSION_MESSAGE,
          extensions=_EXTENSION_MULTI_TYPE,
          expected=_EXTENSION_MESSAGE,
      ),
      dict(
          testcase_name='packed_extension_message_is_unpacked_with_multi',
          message=_PACKED_EXTENSION_MESSAGE,
          extensions=_EXTENSION_MULTI_TYPE,
          expected=_EXTENSION_MESSAGE,
      ),
  )
  def test_unpack_request(self, message, extensions, expected):
    self.assertEqual(
        message_utils.unpack_rpc_request(message, extension_type=extensions),
        expected)

  @parameterized.named_parameters(
      dict(
          testcase_name='create_world_response_passes_through',
          message=_CREATE_WORLD_RESPONSE,
          extensions=_EXTENSION_TYPE,
          expected=_CREATE_WORLD_RESPONSE,
      ),
      dict(
          testcase_name='extension_message_passes_through',
          message=_EXTENSION_MESSAGE,
          extensions=_EXTENSION_TYPE,
          expected=_EXTENSION_MESSAGE,
      ),
      dict(
          testcase_name='packed_extension_message_is_unpacked',
          message=_PACKED_EXTENSION_MESSAGE,
          extensions=_EXTENSION_TYPE,
          expected=_EXTENSION_MESSAGE,
      ),
      dict(
          testcase_name='extension_message_passes_through_with_multi',
          message=_EXTENSION_MESSAGE,
          extensions=_EXTENSION_MULTI_TYPE,
          expected=_EXTENSION_MESSAGE,
      ),
      dict(
          testcase_name='packed_extension_message_is_unpacked_with_multi',
          message=_PACKED_EXTENSION_MESSAGE,
          extensions=_EXTENSION_MULTI_TYPE,
          expected=_EXTENSION_MESSAGE,
      ),
  )
  def test_unpack_response(self, message, extensions, expected):
    self.assertEqual(
        message_utils.unpack_rpc_response(message, extension_type=extensions),
        expected)

if __name__ == '__main__':
  absltest.main()
