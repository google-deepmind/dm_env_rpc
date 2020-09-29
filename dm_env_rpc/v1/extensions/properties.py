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
"""A helper class for sending and receiving property requests and responses.

This helper class provides a Pythonic interface for reading, writing and listing
properties. It simplifies the packing and unpacking of property requests and
responses using the provided dm_env_rpc.v1.connection.Connection instance to
send and receive extension messages.

Example Usage:
  property_extension = PropertyExtension(connection)

  # To read a property:
  value = property_extension['my_property']

  # To write a property:
  property_extension['my_property'] = new_value

  # To find available properties:
  property_specs = property_extension.specs()

  spec = property_specs['my_property']
"""
from typing import Mapping, Sequence, Optional

from dm_env import specs as dm_env_specs
from google.protobuf import any_pb2
from dm_env_rpc.v1 import connection as dm_env_rpc_connection
from dm_env_rpc.v1 import dm_env_rpc_pb2
from dm_env_rpc.v1 import dm_env_utils
from dm_env_rpc.v1 import tensor_utils
from dm_env_rpc.v1.extensions import properties_pb2


class PropertySpec(object):
  """Class that represents a property's specification."""

  def __init__(self, property_spec_proto: properties_pb2.PropertySpec):
    """Constructs a property specification from PropertySpec proto message.

    Args:
      property_spec_proto: A properties_pb2.PropertySpec message.
    """
    self._property_spec_proto = property_spec_proto

  @property
  def key(self) -> str:
    """Return the property's key."""
    return self._property_spec_proto.spec.name

  @property
  def readable(self) -> bool:
    """Returns True if the property is readable."""
    return self._property_spec_proto.is_readable

  @property
  def writable(self) -> bool:
    """Returns True if the property is writable."""
    return self._property_spec_proto.is_writable

  @property
  def listable(self) -> bool:
    """Returns True if the property is listable."""
    return self._property_spec_proto.is_listable

  @property
  def spec(self) -> Optional[dm_env_specs.Array]:
    """Returns a dm_env spec if the property has a valid dtype.

    Returns:
      Either a dm_env spec or, if the dtype is invalid, None.
    """
    if self._property_spec_proto.spec.dtype != (
        dm_env_rpc_pb2.DataType.INVALID_DATA_TYPE):
      return dm_env_utils.tensor_spec_to_dm_env_spec(
          self._property_spec_proto.spec)
    else:
      return None

  @property
  def description(self) -> str:
    """Returns the property's description."""
    return self._property_spec_proto.description

  def __repr__(self):
    return (f'PropertySpec(key={self.key}, readable={self.readable}, '
            f'writable={self.writable}, listable={self.listable}, '
            f'spec={self.spec}, description={self.description})')


class PropertiesExtension(object):
  """Helper class for sending and receiving property requests and responses."""

  def __init__(self, connection: dm_env_rpc_connection.Connection):
    """Construct extension with provided dm_env_rpc connection to the env.

    Args:
      connection: An instance of Connection already connected to a dm_env_rpc
        server.
    """
    self._connection = connection

  def __getitem__(self, key: str):
    """Alias for PropertiesExtension read function."""
    return self.read(key)

  def __setitem__(self, key: str, value) -> None:
    """Alias for PropertiesExtension write function."""
    self.write(key, value)

  def specs(self, key: str = '') -> Mapping[str, PropertySpec]:
    """Helper to return sub-properties as a dict."""
    return {
        sub_property.key: sub_property for sub_property in self.list(key)
    }

  def read(self, key: str):
    """Reads the value of a property.

    Args:
      key: A string key that represents the property to read.

    Returns:
      The value of the property, either as a scalar (float, int, string, etc.)
      or, if the response tensor has a non-empty `shape` attribute, a NumPy
      array of the payload with the correct type and shape. See
      tensor_utils.unpack for more details.
    """
    response = properties_pb2.PropertyResponse()
    packed_request = any_pb2.Any()
    packed_request.Pack(
        properties_pb2.PropertyRequest(
            read_property=properties_pb2.ReadPropertyRequest(key=key)))
    self._connection.send(packed_request).Unpack(response)
    return tensor_utils.unpack_tensor(response.read_property.value)

  def write(self, key: str, value) -> None:
    """Writes the provided value to a property.

    Args:
      key: A string key that represents the property to write.
      value: A scalar (float, int, string, etc.), NumPy array, or nested lists.
        See tensor_utils.pack for more details.
    """
    packed_request = any_pb2.Any()
    packed_request.Pack(
        properties_pb2.PropertyRequest(
            write_property=properties_pb2.WritePropertyRequest(
                key=key, value=tensor_utils.pack_tensor(value))))
    self._connection.send(packed_request)

  def list(self, key: str = '') -> Sequence[PropertySpec]:
    """Lists properties residing under the provided key.

    Args:
      key: A string key to list properties at this location. If empty, returns
        properties registered at the root level.

    Returns:
      A sequence of PropertySpecs.
    """
    response = properties_pb2.PropertyResponse()
    packed_request = any_pb2.Any()
    packed_request.Pack(
        properties_pb2.PropertyRequest(
            list_property=properties_pb2.ListPropertyRequest(key=key)))
    self._connection.send(packed_request).Unpack(response)

    return tuple(
        PropertySpec(sub_property)
        for sub_property in response.list_property.values)
