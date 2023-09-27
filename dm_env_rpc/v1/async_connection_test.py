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
"""Tests for AsyncConnection."""

import asyncio
import contextlib
import queue
import unittest
from unittest import mock

from absl.testing import absltest
import grpc

from google.protobuf import any_pb2
from google.protobuf import struct_pb2
from google.rpc import status_pb2
from dm_env_rpc.v1 import async_connection
from dm_env_rpc.v1 import dm_env_rpc_pb2
from dm_env_rpc.v1 import dm_env_rpc_pb2_grpc
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


def _process() -> grpc.aio.StreamStreamCall:
  requests = queue.Queue()

  async def _write(request):
    requests.put(request)

  async def _read():
    request = requests.get()
    return _REQUEST_RESPONSE_PAIRS.get(request.SerializeToString(), _TEST_ERROR)

  mock_stream = mock.create_autospec(grpc.aio.StreamStreamCall)
  mock_stream.write = _write
  mock_stream.read = _read
  return mock_stream


@contextlib.contextmanager
def _create_mock_async_channel():
  """Mocks out gRPC and returns a channel to be passed to Connection."""
  with mock.patch.object(async_connection, 'dm_env_rpc_pb2_grpc') as mock_grpc:
    mock_stub_class = mock.create_autospec(dm_env_rpc_pb2_grpc.EnvironmentStub)
    mock_stub_class.Process = _process
    mock_grpc.EnvironmentStub.return_value = mock_stub_class
    yield mock.MagicMock()


class AsyncConnectionAsyncTests(unittest.IsolatedAsyncioTestCase):

  async def test_create(self):
    with _create_mock_async_channel() as mock_channel:
      with async_connection.AsyncConnection(mock_channel) as connection:
        response = await connection.send(_CREATE_REQUEST)
        self.assertEqual(_CREATE_RESPONSE, response)

  async def test_error(self):
    with _create_mock_async_channel() as mock_channel:
      with async_connection.AsyncConnection(mock_channel) as connection:
        with self.assertRaisesRegex(error.DmEnvRpcError, 'test error'):
          await connection.send(_BAD_CREATE_REQUEST)

  async def test_extension(self):
    with _create_mock_async_channel() as mock_channel:
      with async_connection.AsyncConnection(mock_channel) as connection:
        request = any_pb2.Any()
        request.Pack(_EXTENSION_REQUEST)
        response = await connection.send(request)
        expected_response = any_pb2.Any()
        expected_response.Pack(_EXTENSION_RESPONSE)
        self.assertEqual(expected_response, response)

  async def test_incorrect_response(self):
    with _create_mock_async_channel() as mock_channel:
      with async_connection.AsyncConnection(mock_channel) as connection:
        with self.assertRaisesRegex(ValueError, 'Unexpected response message'):
          await connection.send(_INCORRECT_RESPONSE_TEST_MSG)

  @mock.patch.object(grpc.aio, 'secure_channel')
  async def test_create_secure_channel_and_connect_context(
      self, mock_secure_channel):

    mock_async_channel = mock.MagicMock()
    mock_async_channel.channel_ready = absltest.mock.AsyncMock()
    mock_async_channel.close = absltest.mock.AsyncMock()
    mock_secure_channel.return_value = mock_async_channel

    with await async_connection.create_secure_async_channel_and_connect(
        'valid_address') as connection:
      self.assertIsNotNone(connection)

    await asyncio.gather(*asyncio.all_tasks() - {asyncio.current_task()})

    mock_async_channel.close.assert_called_once()
    mock_async_channel.channel_ready.assert_called_once()
    mock_secure_channel.assert_called_once()


class AsyncConnectionSyncTests(absltest.TestCase):

  @absltest.mock.patch.object(grpc.aio, 'secure_channel')
  def test_create_secure_channel_and_connect_context(self, mock_secure_channel):

    mock_async_channel = absltest.mock.MagicMock()
    mock_async_channel.channel_ready = absltest.mock.AsyncMock()
    mock_async_channel.close = absltest.mock.AsyncMock()
    mock_secure_channel.return_value = mock_async_channel

    asyncio.set_event_loop(asyncio.new_event_loop())
    loop = asyncio.get_event_loop()

    connection_task = asyncio.ensure_future(
        async_connection.create_secure_async_channel_and_connect(
            'valid_address'))
    connection = loop.run_until_complete(connection_task)

    loop.stop()
    asyncio.set_event_loop(None)

    connection.close()

    mock_async_channel.close.assert_called_once()
    mock_async_channel.channel_ready.assert_called_once()
    mock_secure_channel.assert_called_once()


if __name__ == '__main__':
  absltest.main()
