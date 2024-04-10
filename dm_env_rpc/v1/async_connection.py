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
"""A helper class to manage a connection to a dm_env_rpc server asynchronously.

This helper class allows sending the Request message types and receiving the
Response message as a future.  The class automatically wraps and unwraps from
EnvironmentRequest and EnvironmentResponse, respectively.  It also turns error
messages in to exceptions.

For most calls (such as create, join, etc.):

    with async_connection.AsyncConnection(grpc_channel) as async_channel:
      create_response = await async_channel.send(
          dm_env_rpc_pb2.CreateRequest(settings={
              'players': 5
          })

For the `extension` message type, you must send an Any proto and you'll get back
an Any proto.  It is up to you to wrap and unwrap these to concrete proto types
that you know how to handle.

    with async_connection.AsyncConnection(grpc_channel) as async_channel:
      request = struct_pb2.Struct()
      ...
      request_any = any_pb2.Any()
      request_any.Pack(request)
      response_any = await async_channel.send(request_any)
      response = my_type_pb2.MyType()
      response_any.Unpack(response)


Any errors encountered in the EnvironmentResponse are turned into Python
exceptions, so explicit error handling code isn't needed per call.
"""

import asyncio
from typing import Optional, Sequence, Tuple

import grpc

from dm_env_rpc.v1 import dm_env_rpc_pb2_grpc
from dm_env_rpc.v1 import message_utils


Metadata = Sequence[Tuple[str, str]]


class AsyncConnection:
  """A helper class for interacting with dm_env_rpc servers asynchronously."""

  def __init__(
      self, channel: grpc.aio.Channel, metadata: Optional[Metadata] = None
  ):
    """Manages an async connection to a dm_env_rpc server.

    Args:
      channel: An async grpc channel to connect to the dm_env_rpc server over.
      metadata: Optional sequence of 2-tuples, sent to the gRPC server as
        metadata.
    """
    self._stream = dm_env_rpc_pb2_grpc.EnvironmentStub(channel).Process(
        metadata=metadata
    )

  async def send(
      self,
      request: message_utils.DmEnvRpcRequest) -> message_utils.DmEnvRpcResponse:
    """Sends `request` to the dm_env_rpc server and returns a response future.

    The request should be an instance of one of the dm_env_rpc Request messages,
    such as CreateWorldRequest. Based on the type the correct payload for the
    EnvironmentRequest will be constructed and sent to the dm_env_rpc server.

    Returns an awaitable future to retrieve the response.

    Args:
      request: An instance of a dm_env_rpc Request type, such as
        CreateWorldRequest.

    Returns:
      An asyncio Future which can be awaited to retrieve the response from the
      dm_env_rpc server returned for the given RPC call, unwrapped from the
      EnvironmentStream message.  For instance if `request` had type
      `CreateWorldRequest` this returns a message of type `CreateWorldResponse`.

    Raises:
      DmEnvRpcError: The dm_env_rpc server responded to the request with an
        error.
      ValueError: The dm_env_rpc server responded to the request with an
        unexpected response message.
    """
    environment_request, field_name = (
        message_utils.pack_environment_request(request))
    if self._stream is None:
      raise ValueError('Cannot send request after stream is closed.')
    await self._stream.write(environment_request)
    return message_utils.unpack_environment_response(await self._stream.read(),
                                                     field_name)

  def close(self):
    """Closes the connection.  Call when the connection is no longer needed."""
    if self._stream:
      self._stream = None

  def __exit__(self, *args, **kwargs):
    self.close()

  def __enter__(self):
    return self


async def create_secure_async_channel_and_connect(
    server_address: str,
    credentials: grpc.ChannelCredentials = grpc.local_channel_credentials(),
    metadata: Optional[Metadata] = None,
) -> AsyncConnection:
  """Creates a secure async channel from address and credentials and connects.

  We allow the created channel to have un-bounded message lengths, to support
  large observations.

  Args:
    server_address: URI server address to connect to.
    credentials: gRPC credentials necessary to connect to the server.
    metadata: Optional sequence of 2-tuples, sent to the gRPC server as
        metadata.

  Returns:
    An instance of dm_env_rpc.AsyncConnection, where the async channel is closed
    upon the connection being closed.
  """
  options = [('grpc.max_send_message_length', -1),
             ('grpc.max_receive_message_length', -1)]
  channel = grpc.aio.secure_channel(server_address, credentials,
                                    options=options)
  await channel.channel_ready()

  class _ConnectionWrapper(AsyncConnection):
    """Utility to ensure channel is closed when the connection is closed."""

    def __init__(self, channel, metadata):
      super().__init__(channel=channel, metadata=metadata)
      self._channel = channel

    def __del__(self):
      self.close()

    def close(self):
      super().close()
      try:
        loop = asyncio.get_running_loop()
      except RuntimeError:
        loop = None
      if loop and loop.is_running():
        return asyncio.ensure_future(self._channel.close())
      else:
        return asyncio.run(self._channel.close())

  return _ConnectionWrapper(channel=channel, metadata=metadata)
