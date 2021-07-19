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
import grpc
import mock

from google.protobuf import any_pb2
from google.protobuf import struct_pb2
from google.rpc import status_pb2
from dm_env_rpc.v1 import connection as dm_env_rpc_connection
from dm_env_rpc.v1 import dm_env_rpc_pb2
from dm_env_rpc.v1 import error
from dm_env_rpc.v1 import tensor_utils

_CREATE_REQUEST = dm_env_rpc_pb2.CreateWorldRequest(
    settings={'foo': tensor_utils.pack_tensor('bar')})
_CREATE_RESPONSE = dm_env_rpc_pb2.CreateWorldResponse()

_BAD_CREATE_REQUEST = dm_env_rpc_pb2.CreateWorldRequest()
_TEST_ERROR = dm_env_rpc_pb2.EnvironmentResponse(
    error=status_pb2.Status(message='A test error.'))

_INCORRECT_RESPONSE_TEST_MSG = dm_env_rpc_pb2.DestroyWorldRequest(
    world_name='foo')
_INCORRECT_RESPONSE = dm_env_rpc_pb2.EnvironmentResponse(
    leave_world=dm_env_rpc_pb2.LeaveWorldResponse())

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
    dm_env_rpc_pb2.EnvironmentRequest(
        destroy_world=_INCORRECT_RESPONSE_TEST_MSG).SerializeToString():
        _INCORRECT_RESPONSE,
}


def _process(request_iterator, metadata):
  del metadata
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

  @mock.patch.object(grpc, 'secure_channel')
  @mock.patch.object(grpc, 'channel_ready_future')
  def test_create_secure_channel_and_connect(self, mock_channel_ready,
                                             mock_secure_channel):
    mock_channel = mock.MagicMock()
    mock_secure_channel.return_value = mock_channel

    self.assertIsNotNone(
        dm_env_rpc_connection.create_secure_channel_and_connect(
            'valid_address', grpc.local_channel_credentials()))

    mock_channel_ready.assert_called_once_with(mock_channel)
    mock_secure_channel.assert_called_once()
    mock_channel.close.assert_called_once()

  @mock.patch.object(grpc, 'secure_channel')
  @mock.patch.object(grpc, 'channel_ready_future')
  def test_create_secure_channel_and_connect_context(self, mock_channel_ready,
                                                     mock_secure_channel):
    mock_channel = mock.MagicMock()
    mock_secure_channel.return_value = mock_channel

    with dm_env_rpc_connection.create_secure_channel_and_connect(
        'valid_address') as connection:
      self.assertIsNotNone(connection)

    mock_channel_ready.assert_called_once_with(mock_channel)
    mock_secure_channel.assert_called_once()
    mock_channel.close.assert_called_once()

  def test_create_secure_channel_and_connect_timeout(self):
    with self.assertRaises(grpc.FutureTimeoutError):
      dm_env_rpc_connection.create_secure_channel_and_connect(
          'invalid_address', grpc.local_channel_credentials(), timeout=1.)

  def test_incorrect_response(self):
    with _create_mock_channel() as mock_channel:
      with dm_env_rpc_connection.Connection(mock_channel) as connection:
        with self.assertRaisesRegex(ValueError, 'Unexpected response message'):
          connection.send(_INCORRECT_RESPONSE_TEST_MSG)

  def test_with_metadata(self):
    expected_metadata = (('key', 'value'),)
    with mock.patch.object(dm_env_rpc_connection,
                           'dm_env_rpc_pb2_grpc') as mock_grpc:
      mock_stub_class = mock.MagicMock()
      mock_grpc.EnvironmentStub.return_value = mock_stub_class
      _ = dm_env_rpc_connection.Connection(
          mock.MagicMock(), metadata=expected_metadata)
      mock_stub_class.Process.assert_called_with(
          mock.ANY, metadata=expected_metadata)


if __name__ == '__main__':
  absltest.main()
