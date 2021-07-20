# Lint as: python3
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

from typing import NamedTuple, Union

import immutabledict

from google.protobuf import any_pb2
from dm_env_rpc.v1 import dm_env_rpc_pb2
from dm_env_rpc.v1 import error


_MESSAGE_TYPE_TO_FIELD = immutabledict.immutabledict({
    field.message_type.name: field.name
    for field in dm_env_rpc_pb2.EnvironmentRequest.DESCRIPTOR.fields
})

DmEnvRpcRequest = Union[dm_env_rpc_pb2.CreateWorldRequest,
                        dm_env_rpc_pb2.JoinWorldRequest,
                        dm_env_rpc_pb2.StepRequest,
                        dm_env_rpc_pb2.ResetRequest,
                        dm_env_rpc_pb2.ResetWorldRequest,
                        dm_env_rpc_pb2.LeaveWorldRequest,
                        dm_env_rpc_pb2.DestroyWorldRequest,
                        any_pb2.Any,  # Extension message.
                       ]

DmEnvRpcResponse = Union[dm_env_rpc_pb2.CreateWorldResponse,
                         dm_env_rpc_pb2.JoinWorldResponse,
                         dm_env_rpc_pb2.StepResponse,
                         dm_env_rpc_pb2.ResetResponse,
                         dm_env_rpc_pb2.ResetWorldResponse,
                         dm_env_rpc_pb2.LeaveWorldResponse,
                         dm_env_rpc_pb2.DestroyWorldResponse,
                         any_pb2.Any,  # Extension message.
                        ]


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
