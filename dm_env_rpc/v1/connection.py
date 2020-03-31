# Lint as: python3
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
"""A helper class to manage a connection to a dm_env_rpc server.

This helper class allows sending the Request message types and receiving the
Response message types without wrapping in an EnvironmentRequest or unwrapping
from an EnvironmentResponse.  It also turns error messages in to exceptions.

For most calls (such as create, join, etc.):

    with connection.Connection(grpc_channel) as channel:
      create_response = channel.send(dm_env_rpc_pb2.CreateRequest(settings={
        'players': 5
      })

For the `extension` message type, you must send an Any proto and you'll get back
an Any proto.  It is up to you to wrap and unwrap these to concrete proto types
that you know how to handle.

    with connection.Connection(grpc_channel) as channel:
      request = struct_pb2.Struct()
      ...
      request_any = any_pb2.Any()
      request_any.Pack(request)
      response_any = channel.send(request_any)
      response = my_type_pb2.MyType()
      response_any.Unpack(response)


Any errors encountered in the EnvironmentResponse are turned into Python
exceptions, so explicit error handling code isn't needed per call.
"""

import queue

from dm_env_rpc.v1 import dm_env_rpc_pb2
from dm_env_rpc.v1 import dm_env_rpc_pb2_grpc
from dm_env_rpc.v1 import error


class _StreamReaderWriter(object):
  """Helper class for reading/writing gRPC streams."""

  def __init__(self, stub):
    self._requests = queue.Queue()
    self._stream = stub.Process(iter(self._requests.get, None))

  def write(self, request):
    """Asynchronously sends `request` to the stream."""
    self._requests.put(request)

  def read(self):
    """Returns the response from stream.  Blocking."""
    return next(self._stream)


class Connection(object):
  """A helper class for interacting with dm_env_rpc servers."""

  def __init__(self, channel):
    """Manages a connection to a dm_env_rpc server.

    Args:
      channel: A grpc channel to connect to the dm_env_rpc server over.
    """
    self._stream = _StreamReaderWriter(
        dm_env_rpc_pb2_grpc.EnvironmentStub(channel))
    self._type_to_field = {
        field.message_type.name: field.name
        for field in dm_env_rpc_pb2.EnvironmentRequest.DESCRIPTOR.fields
    }

  def send(self, request):
    """Sends the given request to the dm_env_rpc server and returns the response.

    The request should be an instance of one of the dm_env_rpc Request messages,
    such
    as CreateWorldRequest.  Based on the type the correct payload for the
    EnvironmentRequest will be constructed and set to the dm_env_rpc server.

    Blocks until the server sends back its response.

    Args:
      request: An instance of a dm_env_rpc Request type, such as
        CreateWorldRequest.

    Returns:
      The response the dm_env_rpc server returned for the given RPC call,
      unwrapped
      from the EnvironmentStream message.  For instance if `request` had type
      `CreateWorldRequest` this returns a message of type `CreateWorldResponse`.

    Raises:
      DmEnvRpcError: The dm_env_rpc server responded to the request with an
        error.
    """
    field_name = self._type_to_field[type(request).__name__]
    environment_request = dm_env_rpc_pb2.EnvironmentRequest()
    getattr(environment_request, field_name).CopyFrom(request)
    self._stream.write(environment_request)
    response = self._stream.read()
    if response.HasField('error'):
      raise error.DmEnvRpcError(response.error)
    return getattr(response, field_name)

  def close(self):
    """Closes the connection.  Call when the connection is no longer needed."""
    if self._stream:
      del self._stream

  def __exit__(self, *args, **kwargs):
    self.close()

  def __enter__(self):
    return self
