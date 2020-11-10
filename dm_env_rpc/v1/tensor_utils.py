# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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
"""Helper Python utilities for bridging dm_env_rpc and NumPy.

Note that the Tensor proto payload type is not supported, as it doesn't play
well with NumPy.
"""
import abc
from typing import Optional, Sequence, Union
import numpy as np

from google.protobuf import any_pb2
from google.protobuf import message
from dm_env_rpc.v1 import dm_env_rpc_pb2


def _pack_any_proto(value):
  """Helper function to pack Any proto, iff it's not already packed."""
  if isinstance(value, any_pb2.Any):
    return value
  else:
    any_proto = any_pb2.Any()
    any_proto.Pack(value)
    return any_proto


class Packer(metaclass=abc.ABCMeta):
  """Converts between proto messages and NumPy arrays."""

  def __init__(self, name: str, np_type: np.dtype):
    self._np_type = np_type
    self._name = name

  @property
  def name(self):
    return self._name

  @property
  def np_type(self):
    return self._np_type

  @abc.abstractmethod
  def pack(self, proto, value: np.ndarray):
    """Flattens and stores the given `value` array in the given `proto`."""

  @abc.abstractmethod
  def unpack(self, proto):
    """Retrieves a flat NumPy array for the payload from the given `proto`."""


class _RepeatedFieldPacker(Packer):
  """Handles packing and unpacking most data types."""

  def pack(self, proto, value: np.ndarray):
    payload = getattr(proto, self._name)
    payload.array.extend(value.ravel().tolist())

  def unpack(self, proto):
    payload = getattr(proto, self._name)
    return np.fromiter(payload.array, self.np_type, len(payload.array))


class _BytesPacker(Packer):
  """Handles packing and unpacking int8 and uint8 arrays."""

  def pack(self, proto, value: np.ndarray):
    payload = getattr(proto, self.name)
    payload.array = value.tobytes()

  def unpack(self, proto):
    payload = getattr(proto, self.name)
    return np.frombuffer(payload.array, self.np_type)


class _RepeatedStringFieldPacker(Packer):
  """Handles packing and unpacking strings."""

  def pack(self, proto, value: np.ndarray):
    payload = getattr(proto, self.name)
    payload.array.extend(value.ravel().tolist())

  def unpack(self, proto):
    # String arrays with variable length strings can't be created with
    # np.fromiter, unlike other dtypes.
    payload = getattr(proto, self.name)
    return np.array(payload.array, self.np_type)


class _RepeatedProtoFieldPacker(Packer):
  """Handles packing of protos."""

  def pack(self, proto, value: np.ndarray):
    payload = getattr(proto, self.name)
    payload.array.extend(
        [_pack_any_proto(sub_value) for sub_value in value.ravel()])

  def unpack(self, proto):
    payload = getattr(proto, self._name)
    return np.array(payload.array, self.np_type)


_PACKERS = (
    _RepeatedFieldPacker('floats', np.float32),
    _RepeatedFieldPacker('doubles', np.float64),
    _BytesPacker('int8s', np.int8),
    _RepeatedFieldPacker('int32s', np.int32),
    _RepeatedFieldPacker('int64s', np.int64),
    _BytesPacker('uint8s', np.uint8),
    _RepeatedFieldPacker('uint32s', np.uint32),
    _RepeatedFieldPacker('uint64s', np.uint64),
    _RepeatedFieldPacker('bools', np.bool_),
    _RepeatedStringFieldPacker('strings', np.str_),
    _RepeatedProtoFieldPacker('protos', np.object_),
)

_NAME_TO_NP_TYPE = {
    packer.name: packer.np_type for packer in _PACKERS
}

_TYPE_TO_PACKER = {
    packer.np_type: packer for packer in _PACKERS
}

_DM_ENV_RPC_DTYPE_TO_NUMPY_DTYPE = {
    dm_env_rpc_pb2.DataType.FLOAT: np.float32,
    dm_env_rpc_pb2.DataType.DOUBLE: np.float64,
    dm_env_rpc_pb2.DataType.INT8: np.int8,
    dm_env_rpc_pb2.DataType.INT32: np.int32,
    dm_env_rpc_pb2.DataType.INT64: np.int64,
    dm_env_rpc_pb2.DataType.UINT8: np.uint8,
    dm_env_rpc_pb2.DataType.UINT32: np.uint32,
    dm_env_rpc_pb2.DataType.UINT64: np.uint64,
    dm_env_rpc_pb2.DataType.BOOL: np.bool_,
    dm_env_rpc_pb2.DataType.STRING: np.str_,
    dm_env_rpc_pb2.DataType.PROTO: np.object_,
}

_NUMPY_DTYPE_TO_DM_ENV_RPC_DTYPE = {
    value: key
    for key, value in _DM_ENV_RPC_DTYPE_TO_NUMPY_DTYPE.items()
}


def get_tensor_type(tensor_proto):
  """Returns the NumPy type for the given tensor."""
  payload = tensor_proto.WhichOneof('payload')
  np_type = _NAME_TO_NP_TYPE.get(payload)
  if not np_type:
    raise TypeError(f'Unknown NumPy type {payload}')
  return np_type


def data_type_to_np_type(dm_env_rpc_dtype):
  """Returns the NumPy type for the given dm_env_rpc DataType."""
  np_type = _DM_ENV_RPC_DTYPE_TO_NUMPY_DTYPE.get(dm_env_rpc_dtype)
  if not np_type:
    raise TypeError(f'Unknown DataType {dm_env_rpc_dtype}')
  return np_type


def np_type_to_data_type(np_type):
  """Returns the dm_env_rpc DataType for the given NumPy type."""
  if isinstance(np_type, np.dtype):
    # Flatten scalar types, since np.int32 is different from np.dtype(np.int32)
    # for dict key lookup.
    np_type = np_type.type
  data_type = _NUMPY_DTYPE_TO_DM_ENV_RPC_DTYPE.get(np_type)
  if data_type is None:
    raise TypeError(
        f'No dm_env_rpc DataType corresponds to NumPy type "{np_type}"')
  return data_type


def get_packer(np_type):
  """Retrieves the `Packer` which can handle the given NumPy Type.

  Note: The returned packer is a relatively low level mechanism to convert
  between NumPy arrays and the repeated `payload` fields in dm_env_rpc messages.
  It won't set shape or type on the proto message.  Instead of this packer,
  generally you should use `pack_tensor` and `unpack_tensor` to pack and unpack
  data to `Tensor` messages, as it will handle setting shape and type on the
  `Tensor` message as well.

  Args:
    np_type: The NumPy data type to retrieve a packer for.  eg: np.int32.

  Returns:
    An instance of Packer which will handle conversion between NumPy arrays of
    `np_type` and the corresponding payload field in the dm_env_rpc message.

  Raises:
    TypeError: If the provided NumPy type has no known packer.
  """
  packer = _TYPE_TO_PACKER.get(np_type)
  if not packer:
    raise TypeError(f'Unknown NumPy type "{np_type}" has no known packer.')
  return packer


def reshape_array(array: np.ndarray, shape):
  """Reshapes `array` to the given `shape` using dm_env_rpc's rules."""
  if shape:
    if len(array) == 1:
      array = np.full(np.maximum(shape, 1), array[0])
    else:
      array.shape = shape
    return array
  else:
    length = len(array)
    if length != 1:
      raise ValueError(
          'Scalar tensors must have exactly 1 element but had {} elements.'
          .format(length))
    return array[0]


def unpack_proto(proto: Union[dm_env_rpc_pb2.Tensor,
                              dm_env_rpc_pb2.TensorSpec.Value],
                 shape: Optional[Sequence[int]]):
  """Converts a proto with payload oneof to a scalar or NumPy array.

  Args:
    proto: A dm_env_rpc proto with payload oneof.
    shape: Optional dimensions of the payload data. If not set or empty, the
      data is assumed to be a scalar type.

  Returns:
    If `shape` is empty or None, returns a scalar (float, int, string, etc.)
    of the correct type and value. Otherwise returns a NumPy array of the
    payload with the correct type and shape.
  """
  np_type = get_tensor_type(proto)
  packer = get_packer(np_type)
  array = packer.unpack(proto)
  return reshape_array(array, shape)


def unpack_tensor(tensor_proto: dm_env_rpc_pb2.Tensor):
  """Converts a Tensor proto to a scalar or NumPy array.

  Args:
    tensor_proto: A dm_env_rpc Tensor protobuf.

  Returns:
    If the provided tensor_proto has a non-empty `shape` attribute, returns
    a NumPy array of the payload with the correct type and shape.  If the
    `shape` attribute is empty, returns a scalar (float, int, string, etc.)
    of the correct type and value.
  """
  return unpack_proto(tensor_proto, tensor_proto.shape)


def pack_tensor(value, dtype=None, try_compress=False):
  """Encodes the given value as a tensor.

  Args:
    value: A scalar (float, int, string, etc.), protobuf message, NumPy array,
      or nested lists.
    dtype: The type to pack the data to.  If set to None, will attempt to detect
      the correct type automatically.  Either a dm_env_rpc DataType enum or
      NumPy type is acceptable.
    try_compress: A bool, whether to try and encode the tensor in less space or
      not. This will increase the computational cost of the packing, but may
      reduce the on-the-wire size of the tensor.  There are no guarantees that
      any compression will actually happen.

  Raises:
    ValueError: If `value` is a jagged array, not a primitive type, nested
      iterable of primitive types or protobuf messages, or all elements can't be
      cast to the same type or the requested type.

  Returns:
    A dm_env_rpc Tensor proto containing the data.
  """
  packed = dm_env_rpc_pb2.Tensor()
  value = np.asarray(value)

  # For efficiency, only check that the first element is a protobuf message.
  if value.dtype == np.object and value.size > 0 and not isinstance(
      value.item(0), message.Message):
    raise ValueError('Could not convert value to a tensor of primitive types: '
                     f'{value}. Are the iterables jagged? Are the data types '
                     'not primitive scalar types like strings, floats, or '
                     'integers? Or are the elements not protobuf messages?')
  if dtype is not None:
    value = value.astype(
        dtype=_DM_ENV_RPC_DTYPE_TO_NUMPY_DTYPE.get(dtype, dtype),
        copy=False,
        casting='same_kind')
  packed.shape[:] = value.shape
  packer = _TYPE_TO_PACKER[value.dtype.type]
  if (try_compress and np.all(value == next(value.flat))):
    # All elements are the same.  Pack in to a single value.
    packer.pack(packed, value.ravel()[0:1])
  else:
    packer.pack(packed, value)
  return packed
