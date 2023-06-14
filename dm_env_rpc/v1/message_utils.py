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
"""Helper functions used to process dm_env_rpc request / response messages.
"""

import typing
from typing import Any, Iterable, NamedTuple, Type, Tuple, Union

import immutabledict

from google.protobuf import any_pb2
from dm_env_rpc.v1 import dm_env_rpc_pb2
from dm_env_rpc.v1 import error
from google.protobuf import message


_MESSAGE_TYPE_TO_FIELD = immutabledict.immutabledict({
    field.message_type.name: field.name
    for field in dm_env_rpc_pb2.EnvironmentRequest.DESCRIPTOR.fields
})

# An unpacked extension request (anything).
# As any proto message that is not a native request is accepted, this definition
# is overly broad - use with care.
DmEnvRpcExtensionMessage = message.Message

# A packed extension request.
# Wraps a DmEnvRpcExtensionMessage.
DmEnvRpcPackedExtensionMessage = any_pb2.Any

# A native request RPC.
DmEnvRpcNativeRequest = Union[
    dm_env_rpc_pb2.CreateWorldRequest,
    dm_env_rpc_pb2.JoinWorldRequest,
    dm_env_rpc_pb2.StepRequest,
    dm_env_rpc_pb2.ResetRequest,
    dm_env_rpc_pb2.ResetWorldRequest,
    dm_env_rpc_pb2.LeaveWorldRequest,
    dm_env_rpc_pb2.DestroyWorldRequest,
]

# A native response RPC.
DmEnvRpcNativeResponse = Union[
    dm_env_rpc_pb2.CreateWorldResponse,
    dm_env_rpc_pb2.JoinWorldResponse,
    dm_env_rpc_pb2.StepResponse,
    dm_env_rpc_pb2.ResetResponse,
    dm_env_rpc_pb2.ResetWorldResponse,
    dm_env_rpc_pb2.LeaveWorldResponse,
    dm_env_rpc_pb2.DestroyWorldResponse,
]

# A native request RPC, or an extension message, wrapped in Any.
DmEnvRpcRequest = Union[DmEnvRpcNativeRequest, DmEnvRpcPackedExtensionMessage]
# A native response RPC, or an extension message, wrapped in Any.
DmEnvRpcResponse = Union[DmEnvRpcNativeResponse, DmEnvRpcPackedExtensionMessage]


def pack_rpc_request(
    request: Union[DmEnvRpcRequest, DmEnvRpcExtensionMessage],
) -> DmEnvRpcRequest:
  """Returns a DmEnvRpcRequest that is suitable to send over the wire.

  Arguments:
    request: The request to pack.

  Returns:
    Native request - returned as-is.
    Packed extension (any_pb2.Any) - returned as-is.
    Everything else - assumed to be an extension; wrapped in any_pb2.Any.
  """
  # pytype: disable=bad-return-type
  return _pack_message(request, _get_union_args(DmEnvRpcNativeRequest))


def unpack_rpc_request(
    request: Union[DmEnvRpcRequest, DmEnvRpcExtensionMessage],
    *,
    extension_type: Union[
        Type[message.Message], Iterable[Type[message.Message]]
    ],
) -> Union[DmEnvRpcRequest, DmEnvRpcExtensionMessage]:
  """Returns a DmEnvRpcRequest without a wrapper around extension messages.

  Arguments:
    request: The request to unpack.
    extension_type: One or more extension protobuf classes of known extension
      types.

  Returns:
    Native request - returned as-is.
    Unpacked extension in |extension_type| - returned as-is.
    Packed extension (any_pb2.Any) in |extension_type| - returned unpacked

  Raises:
    ValueError: The message is packed (any_pb2.Any), but not in |extension_type|
        or the message type is not a native request or known extension.
  """
  return _unpack_message(
      request,
      _get_union_args(DmEnvRpcNativeRequest),
      extension_type=extension_type)


def pack_rpc_response(
    response: Union[DmEnvRpcResponse, DmEnvRpcExtensionMessage],
) -> DmEnvRpcResponse:
  """Returns a DmEnvRpcResponse that is suitable to send over the wire.

  Arguments:
    response: The response to pack.

  Returns:
    Native response - returned as-is.
    Packed extension (any_pb2.Any) - returned as-is.
    Everything else - assumed to be an extension; wrapped in any_pb2.Any.
  """
  # pytype: disable=bad-return-type
  return _pack_message(response, _get_union_args(DmEnvRpcNativeResponse))


def unpack_rpc_response(
    response: Union[DmEnvRpcResponse, DmEnvRpcExtensionMessage],
    *,
    extension_type: Union[
        Type[message.Message], Iterable[Type[message.Message]]
    ],
) -> Union[DmEnvRpcRequest, DmEnvRpcExtensionMessage]:
  """Returns a DmEnvRpcResponse without a wrapper around extension messages.

  Arguments:
    response: The response to unpack.
    extension_type: One or more extension protobuf classes of known extension
      types.

  Returns:
    Native response - returned as-is.
    Unpacked extension in |extension_type| - returned as-is.
    Packed extension (any_pb2.Any) in |extension_type| - returned unpacked

  Raises:
    ValueError: The message is packed (any_pb2.Any), but not in |extension_type|
        or the message type is not a native request or known extension.
  """
  return _unpack_message(response, _get_union_args(DmEnvRpcNativeResponse),
                         extension_type=extension_type)


class EnvironmentRequestAndFieldName(NamedTuple):
  """EnvironmentRequest and field name used when packing."""
  environment_request: dm_env_rpc_pb2.EnvironmentRequest
  field_name: str


def pack_environment_request(
    request: DmEnvRpcRequest) -> EnvironmentRequestAndFieldName:
  """Constructs an EnvironmentRequest containing a request message.

  Args:
    request: An instance of a dm_env_rpc Request type, such as
      CreateWorldRequest.

  Returns:
    A tuple of (environment_request, field_name) where:
      environment_request: dm_env_rpc.v1.EnvironmentRequest containing the input
        request message.
      field_name: Name of the environment request field holding the input
        request message.
  """
  field_name = _MESSAGE_TYPE_TO_FIELD[type(request).__name__]
  environment_request = dm_env_rpc_pb2.EnvironmentRequest()
  getattr(environment_request, field_name).CopyFrom(request)
  return EnvironmentRequestAndFieldName(environment_request, field_name)


def unpack_environment_response(
    environment_response: dm_env_rpc_pb2.EnvironmentResponse,
    expected_field_name: str) -> DmEnvRpcResponse:
  """Extracts the response message contained within an EnvironmentResponse.

  Args:
    environment_response: An instance of dm_env_rpc.v1.EnvironmentResponse.
    expected_field_name: Name of the environment response field expected to be
      holding the dm_env_rpc response message.

  Returns:
    dm_env_rpc response message wrapped in the input environment response.

  Raises:
    DmEnvRpcError: The dm_env_rpc EnvironmentResponse contains an error.
    ValueError: The dm_env_rpc response message contained in the
      EnvironmentResponse is held in a different field from the one expected.
  """
  response_field_name = environment_response.WhichOneof('payload')
  if response_field_name == 'error':
    raise error.DmEnvRpcError(environment_response.error)
  elif response_field_name == expected_field_name:
    return getattr(environment_response, expected_field_name)
  else:
    raise ValueError('Unexpected response message! expected: '
                     f'{expected_field_name}, actual: {response_field_name}')


def _pack_message(
    msg: message.Message,
    union_types: Tuple[Type[Any], ...],
) -> message.Message:
  """Packs a message based on its union types.

  Args:
    msg: The message to process.
    union_types: Determines which types are "native" and "extension".
        "native" types are passed through, unchanged.
        "extension" types are packed into an Any message and returned.

  Returns:
    A message within union_types, or an Any proto.
  """
  if isinstance(msg, union_types):
    return msg

  if isinstance(msg, any_pb2.Any):
    return msg

  # Assume the message is an extension.
  packed = any_pb2.Any()
  packed.Pack(msg)
  return packed


def _unpack_message(
    msg: message.Message,
    union_types: Tuple[Type[Any], ...],
    *,
    extension_type: Union[
        Type[message.Message], Iterable[Type[message.Message]]
    ],
) -> message.Message:
  """Packs a message based on its union types.

  Args:
    msg: The message to process.
    union_types: Determines which types are "native" and "extension". "native"
      types are passed through, unchanged. "extension" types are packed into an
      Any message and returned.
    extension_type: Type or type(s) used to match extension messages. The first
      matching type is used.

  Returns:
    An upacked message within |union_types| or an unpacked extension message
    with type within |extension_type|.

  Raises:
    TypeError: Raised if a return type could not be determined.
  """
  if isinstance(msg, union_types):
    return msg

  if isinstance(extension_type, type):
    extension_type = (extension_type,)
  else:
    extension_type = tuple(extension_type)

  if isinstance(msg, extension_type):
    return msg

  if isinstance(msg, any_pb2.Any):
    matching_type = next(
        (e for e in extension_type if msg.Is(e.DESCRIPTOR)), None)

    if not matching_type:
      raise ValueError(
          'Extension type could not be found to unpack message: '
          f'{type(msg).__name__}.\n'
          f'Known Types:\n' + '\n'.join(f'- {e}' for e in extension_type))

    unpacked = matching_type()
    msg.Unpack(unpacked)
    return unpacked

  raise ValueError(
      'Message type does not appear in union_types and is not an extension: '
      f'{type(msg).__name__}.\n'
      f'Union Types:\n' + '\n'.join(f'- {t}' for t in union_types))


def _get_union_args(union) -> Tuple[Type[Any], ...]:
  """Python version-safe function for getting the types in a union."""
  # __args__ exists < v3.8, typing.get_args() exists >= 3.8.
  return getattr(union, '__args__', None) or typing.get_args(union)
