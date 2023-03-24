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
from typing import Optional, Type, Union
import numpy as np

from google.protobuf import any_pb2
from google.protobuf import message
from dm_env_rpc.v1 import dm_env_rpc_pb2


def _pack_any_proto(value):
  """Helper function to pack Any proto, iff it's not already packed."""
  if isinstance(value, any_pb2.Any):
    return value
  elif isinstance(value, message.Message):
    any_proto = any_pb2.Any()
    any_proto.Pack(value)
    return any_proto
  else:
    # If we reach this exception, it is normally because the type being packed
    # is not supported. Raise exception with some typical examples.
    raise ValueError("Trying to pack an Any proto with a type that's not "
                     f"recognized! Type: {type(value)}, value: '{value}'. "
                     'Is the value a jagged iterable? Is the data type not a '
                     'supported primitive type like strings, floats, integers '
                     'or protobuf messages? Are all elements in the array the '
                     'same type?')


TensorOrTensorSpecValue = Union[dm_env_rpc_pb2.Tensor,
                                dm_env_rpc_pb2.TensorSpec.Value]


class Packer(metaclass=abc.ABCMeta):
  """Converts between proto messages and NumPy arrays."""

  def __init__(self, name: str, np_type: np.dtype):
    self._np_type = np_type
    self._name = name

  @property
  def name(self) -> str:
    return self._name

  @property
  def np_type(self) -> np.dtype:
    return self._np_type

  @abc.abstractmethod
  def pack(self, proto: TensorOrTensorSpecValue, value: np.ndarray):
    """Flattens and stores the given `value` array in the given `proto`."""

  @abc.abstractmethod
  def unpack(self, proto: TensorOrTensorSpecValue) -> np.ndarray:
    """Retrieves a flat NumPy array for the payload from the given `proto`."""


class _RepeatedFieldPacker(Packer):
  """Handles packing and unpacking most data types."""

  def pack(self, proto: TensorOrTensorSpecValue, value: np.ndarray):
    payload = getattr(proto, self._name)
    payload.array.extend(value.ravel().tolist())

  def unpack(self, proto: TensorOrTensorSpecValue) -> np.ndarray:
    payload = getattr(proto, self._name)
    return np.fromiter(payload.array, self.np_type, len(payload.array))


class _BytesPacker(Packer):
  """Handles packing and unpacking int8 and uint8 arrays."""

  def pack(self, proto: TensorOrTensorSpecValue, value: np.ndarray):
    payload = getattr(proto, self.name)
    payload.array = value.tobytes()

  def unpack(self, proto: TensorOrTensorSpecValue) -> np.ndarray:
    payload = getattr(proto, self.name)
    return np.frombuffer(payload.array, self.np_type)


class _RepeatedStringFieldPacker(Packer):
  """Handles packing and unpacking strings."""

  def pack(self, proto: TensorOrTensorSpecValue, value: np.ndarray):
    payload = getattr(proto, self.name)
    payload.array.extend(value.ravel().tolist())

  def unpack(self, proto: TensorOrTensorSpecValue) -> np.ndarray:
    # String arrays with variable length strings can't be created with
    # np.fromiter, unlike other dtypes.
    payload = getattr(proto, self.name)
    return np.array(payload.array, self.np_type)


class _RepeatedProtoFieldPacker(Packer):
  """Handles packing of protos."""

  def pack(self, proto: TensorOrTensorSpecValue, value: np.ndarray):
    payload = getattr(proto, self.name)
    payload.array.extend(
        [_pack_any_proto(sub_value) for sub_value in value.ravel()])

  def unpack(self, proto: TensorOrTensorSpecValue) -> np.ndarray:
    payload = getattr(proto, self._name)
    return np.array(payload.array, self.np_type)


_PACKERS = (
    _RepeatedFieldPacker('floats', np.dtype(np.float32)),
    _RepeatedFieldPacker('doubles', np.dtype(np.float64)),
    _BytesPacker('int8s', np.dtype(np.int8)),
    _RepeatedFieldPacker('int32s', np.dtype(np.int32)),
    _RepeatedFieldPacker('int64s', np.dtype(np.int64)),
    _BytesPacker('uint8s', np.dtype(np.uint8)),
    _RepeatedFieldPacker('uint32s', np.dtype(np.uint32)),
    _RepeatedFieldPacker('uint64s', np.dtype(np.uint64)),
    _RepeatedFieldPacker('bools', np.dtype(bool)),
    _RepeatedStringFieldPacker('strings', np.dtype(str)),
    _RepeatedProtoFieldPacker('protos', np.dtype(object)),
)

_NAME_TO_NP_TYPE = {
    packer.name: packer.np_type for packer in _PACKERS
}

_TYPE_TO_PACKER = {packer.np_type: packer for packer in _PACKERS}

_DM_ENV_RPC_DTYPE_TO_NUMPY_DTYPE = {
    dm_env_rpc_pb2.DataType.FLOAT: np.dtype(np.float32),
    dm_env_rpc_pb2.DataType.DOUBLE: np.dtype(np.float64),
    dm_env_rpc_pb2.DataType.INT8: np.dtype(np.int8),
    dm_env_rpc_pb2.DataType.INT32: np.dtype(np.int32),
    dm_env_rpc_pb2.DataType.INT64: np.dtype(np.int64),
    dm_env_rpc_pb2.DataType.UINT8: np.dtype(np.uint8),
    dm_env_rpc_pb2.DataType.UINT32: np.dtype(np.uint32),
    dm_env_rpc_pb2.DataType.UINT64: np.dtype(np.uint64),
    dm_env_rpc_pb2.DataType.BOOL: np.dtype(bool),
    dm_env_rpc_pb2.DataType.STRING: np.dtype(str),
    dm_env_rpc_pb2.DataType.PROTO: np.dtype(object),
}

_NUMPY_DTYPE_TO_DM_ENV_RPC_DTYPE = {
    **{value: key
       for key, value in _DM_ENV_RPC_DTYPE_TO_NUMPY_DTYPE.items()},
    # Legacy support for numpy built-in types (no longer recommended as of
    # release 1.20.0 -
    # https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations)
    np.bool_: dm_env_rpc_pb2.DataType.BOOL,
    np.str_: dm_env_rpc_pb2.DataType.STRING,
    np.object_: dm_env_rpc_pb2.DataType.PROTO,
}


def get_tensor_type(tensor_proto: dm_env_rpc_pb2.Tensor) -> np.dtype:
  """Returns the NumPy type for the given tensor."""
  payload = tensor_proto.WhichOneof('payload')
  np_type = _NAME_TO_NP_TYPE.get(payload)
  if not np_type:
    raise TypeError(f'Unknown NumPy type {payload}')
  return np_type


def data_type_to_np_type(dm_env_rpc_dtype: dm_env_rpc_pb2.DataType) -> np.dtype:
  """Returns the NumPy type for the given dm_env_rpc DataType."""
  np_type = _DM_ENV_RPC_DTYPE_TO_NUMPY_DTYPE.get(dm_env_rpc_dtype)
  if not np_type:
    raise TypeError(f'Unknown DataType {dm_env_rpc_dtype}')
  return np_type


def np_type_to_data_type(
    np_type: Union[np.dtype, Type[np.generic]]
) -> dm_env_rpc_pb2.DataType:
  """Returns the dm_env_rpc DataType for the given NumPy type."""
  data_type = _NUMPY_DTYPE_TO_DM_ENV_RPC_DTYPE.get(np.dtype(np_type))
  if data_type is None:
    raise TypeError(
        f'No dm_env_rpc DataType corresponds to NumPy type "{np_type}"')
  return data_type


def get_packer(np_type: Union[np.dtype, Type[np.generic]]) -> Packer:
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
  packer = _TYPE_TO_PACKER.get(np.dtype(np_type))
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


def unpack_proto(proto: TensorOrTensorSpecValue) -> np.ndarray:
  """Converts a proto with payload oneof to a scalar or NumPy array.

  Args:
    proto: A dm_env_rpc proto with payload oneof.

  Returns:
    Returns a NumPy array of the payload with the correct type.
  """
  np_type = get_tensor_type(proto)
  packer = get_packer(np_type)
  return packer.unpack(proto)


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
  array = unpack_proto(tensor_proto)
  return reshape_array(array, tensor_proto.shape)


def pack_tensor(
    value,
    dtype: Optional[
        Union[np.dtype, Type[np.generic], 'dm_env_rpc_pb2.DataType']
    ] = None,
    try_compress=False,
) -> dm_env_rpc_pb2.Tensor:
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

  if value.dtype == object:
    # Because Numpy doesn't truly support variable-length string arrays, users
    # tend to use arrays of Numpy objects instead. Iff a user provides an array
    # of objects and a string dtype argument, automatically convert the value to
    # an array of strings.
    if np.issubdtype(
        _DM_ENV_RPC_DTYPE_TO_NUMPY_DTYPE.get(dtype, dtype), np.str_):
      for item in value.flat:
        if not isinstance(item, str):
          raise TypeError(f'Requested string dtype but not all elements are '
                          'Python string types. At least one element was '
                          f'{type(item)}.')
      value = np.array(value, dtype=np.str_)

  elif dtype is not None:
    # NumPy defaults to np.float64 dtype when calling np.asarray() on an empty
    # array. Allow unsafe casting in this particular case.
    value = value.astype(
        dtype=_DM_ENV_RPC_DTYPE_TO_NUMPY_DTYPE.get(dtype, dtype),
        copy=False,
        casting='same_kind' if value.size else 'unsafe')

  packed.shape[:] = value.shape
  packer = get_packer(value.dtype.type)
  if (try_compress and np.all(value == next(value.flat))):
    # All elements are the same.  Pack in to a single value.
    packer.pack(packed, value.ravel()[0:1])
  else:
    packer.pack(packed, value)
  return packed
