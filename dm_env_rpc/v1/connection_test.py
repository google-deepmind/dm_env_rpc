# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for Connection."""

import contextlib

from absl.testing import absltest
import mock

from google.protobuf import any_pb2
from google.protobuf import struct_pb2
from google.rpc import status_pb2
from dm_env_rpc.v1 import connection as dm_env_rpc_connection
from dm_env_rpc.v1 import dm_env_rpc_pb2
from dm_env_rpc.v1 import error
from dm_env_rpc.v1 import tensor_utils

_SUEZ_SERVICE = dm_env_rpc_pb2.DESCRIPTOR.services_by_name['Environment']

_CREATE_REQUEST = dm_env_rpc_pb2.CreateWorldRequest(
    settings={'foo': tensor_utils.pack_tensor('bar')})
_CREATE_RESPONSE = dm_env_rpc_pb2.CreateWorldResponse()

_BAD_CREATE_REQUEST = dm_env_rpc_pb2.CreateWorldRequest()
_TEST_ERROR = dm_env_rpc_pb2.EnvironmentResponse(
    error=status_pb2.Status(message='A test error.'))

_EXTENSION_REQUEST = struct_pb2.Value(string_value='extension request')
_EXTENSION_RESPONSE = struct_pb2.Value(number_value=555)


def _wrap_in_any(proto):
  any_proto = any_pb2.Any()
  any_proto.Pack(proto)
  return any_proto


_REQUEST_RESPONSE_PAIRS = {
    dm_env_rpc_pb2.EnvironmentRequest(
        create_world=_CREATE_REQUEST).SerializeToString():
        dm_env_rpc_pb2.EnvironmentResponse(create_world=_CREATE_RESPONSE),
    dm_env_rpc_pb2.EnvironmentRequest(
        create_world=_BAD_CREATE_REQUEST).SerializeToString():
        _TEST_ERROR,
    dm_env_rpc_pb2.EnvironmentRequest(
        extension=_wrap_in_any(_EXTENSION_REQUEST)).SerializeToString():
        dm_env_rpc_pb2.EnvironmentResponse(
            extension=_wrap_in_any(_EXTENSION_RESPONSE)),
}


def _process(request_iterator):
  for request in request_iterator:
    yield _REQUEST_RESPONSE_PAIRS.get(request.SerializeToString(), _TEST_ERROR)


@contextlib.contextmanager
def _create_mock_channel():
  """Mocks out gRPC and returns a channel to be passed to Connection."""
  with mock.patch.object(dm_env_rpc_connection,
                         'dm_env_rpc_pb2_grpc') as mock_grpc:
    mock_stub_class = mock.MagicMock()
    mock_stub_class.Process = _process
    mock_grpc.EnvironmentStub.return_value = mock_stub_class
    yield mock.MagicMock()


class ConnectionTests(absltest.TestCase):

  def test_create(self):
    with _create_mock_channel() as mock_channel:
      with dm_env_rpc_connection.Connection(mock_channel) as connection:
        response = connection.send(_CREATE_REQUEST)
        self.assertEqual(_CREATE_RESPONSE, response)

  def test_error(self):
    with _create_mock_channel() as mock_channel:
      with dm_env_rpc_connection.Connection(mock_channel) as connection:
        with self.assertRaisesRegex(error.DmEnvRpcError, 'test error'):
          connection.send(_BAD_CREATE_REQUEST)

  def test_extension(self):
    with _create_mock_channel() as mock_channel:
      with dm_env_rpc_connection.Connection(mock_channel) as connection:
        request = any_pb2.Any()
        request.Pack(_EXTENSION_REQUEST)
        response = connection.send(request)
        expected_response = any_pb2.Any()
        expected_response.Pack(_EXTENSION_RESPONSE)
        self.assertEqual(expected_response, response)


if __name__ == '__main__':
  absltest.main()
