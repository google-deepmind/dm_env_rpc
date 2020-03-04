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

import struct

import numpy as np

from dm_env_rpc.v1 import dm_env_rpc_pb2


class _BytesWrapper(object):
  """Allows protobuf bytes field to be set using the [:] = ... syntax."""

  def __init__(self, array, signed):
    self._array = array
    self._signed = signed

  def __getitem__(self, index):
    return self._to_scalar(self._array.array[index])

  def __setitem__(self, index, value):
    if index == slice(None, None, None):
      self._array.array = struct.pack(
          str(len(value)) + ('b' if self._signed else 'B'), *value)
    else:
      raise ValueError('Unsupported index {}'.format(index))

  def __len__(self):
    return len(self._array.array)

  def _to_scalar(self, value):
    return np.int8(value) if self._signed else np.uint8(value)

  def as_np_array(self):
    return np.frombuffer(
        self._array.array, dtype=np.int8 if self._signed else np.uint8)


# Payload channel name, NumPy representation, and the payload array.
_RAW_ASSOCIATIONS = (
    ('floats', np.float32, lambda tensor: tensor.floats.array),
    ('doubles', np.float64, lambda tensor: tensor.doubles.array),
    ('int8s', np.int8, lambda tensor: _BytesWrapper(tensor.int8s, signed=True)),
    ('int32s', np.int32, lambda tensor: tensor.int32s.array),
    ('int64s', np.int64, lambda tensor: tensor.int64s.array),
    ('uint8s', np.uint8,
     lambda tensor: _BytesWrapper(tensor.uint8s, signed=False)),
    ('uint32s', np.uint32, lambda tensor: tensor.uint32s.array),
    ('uint64s', np.uint64, lambda tensor: tensor.uint64s.array),
    ('bools', np.bool_, lambda tensor: tensor.bools.array),
    ('strings', np.str_, lambda tensor: tensor.strings.array),
)

_NAME_TO_NP_TYPE = {
    name: np_type for name, np_type, payload in _RAW_ASSOCIATIONS
}

_TYPE_TO_PAYLOAD = {
    np_type: payload for name, np_type, payload in _RAW_ASSOCIATIONS
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
}


def get_tensor_type(tensor_proto):
  """Returns the NumPy type for the given tensor."""
  payload = tensor_proto.WhichOneof('payload')
  np_type = _NAME_TO_NP_TYPE.get(payload)
  if not np_type:
    raise TypeError('Unknown type {}'.format(payload))
  return np_type


def data_type_to_np_type(dm_env_rpc_dtype):
  """Returns the NumPy type for the given dm_env_rpc DataType."""
  np_type = _DM_ENV_RPC_DTYPE_TO_NUMPY_DTYPE.get(dm_env_rpc_dtype)
  if not np_type:
    raise TypeError('Unknown type {}'.format(dm_env_rpc_dtype))
  return np_type


def unpack_proto(proto, shape):
  """Converts a proto with payload oneof to a scalar or NumPy array.

  Args:
    proto: A dm_env_rpc proto with payload oneof.
    shape: Optional dimensions of the payload data. If not set or empty, the
      data is assumed to be a scalar type.

  Returns:
    If `shape` is empty or None,, returns a scalar (float, int, string, etc.)
    of the correct type and value. Otherwise returns a NumPy array of the
    payload with the correct type and shape.
  """
  np_type = get_tensor_type(proto)
  payload = _TYPE_TO_PAYLOAD[np_type](proto)
  if shape:
    if len(payload) == 1:
      array = np.full(np.maximum(shape, 1), payload[0])
    else:
      if isinstance(payload, _BytesWrapper):
        array = payload.as_np_array()
      else:
        array = np.array(payload, np_type)
      array.shape = shape
    return array
  else:
    length = len(payload)
    if length != 1:
      raise ValueError(
          'Scalar tensors must have exactly 1 element but had {} elements.'
          .format(length))
    return np_type(payload[0])


def unpack_tensor(tensor_proto):
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
    value: A scalar (float, int, string, etc.), NumPy array, or nested lists.
    dtype: The type to pack the data to.  If set to None, will attempt to detect
      the correct type automatically.  Either a dm_env_rpc DataType enum or
      NumPy type is acceptable.
    try_compress: A bool, whether to try and encode the tensor in less space or
      not. This will increase the computational cost of the packing, but may
      reduce the on-the-wire size of the tensor.  There are no guarantees that
      any compression will actually happen.

  Raises:
    ValueError: If `value` is a jagged array, not a primitive type or nested
      iterable of primitive types, or all elements can't be cast to the same
      type or the requested type.

  Returns:
    A dm_env_rpc Tensor proto containing the data.
  """
  packed = dm_env_rpc_pb2.Tensor()
  value = np.asarray(value)
  if value.dtype == np.object:
    raise ValueError('Could not convert to a tensor of primitive types.  Are '
                     'the iterables jagged?  Or are the data types not '
                     'primitive scalar types like strings, floats, or '
                     'integers?')
  if dtype is not None:
    value = value.astype(
        dtype=_DM_ENV_RPC_DTYPE_TO_NUMPY_DTYPE.get(dtype, dtype),
        copy=False,
        casting='same_kind')
  packed.shape[:] = value.shape
  pack_target = _TYPE_TO_PAYLOAD[value.dtype.type](packed)
  if (try_compress and np.all(value == next(value.flat))):
    # All elements are the same.  Pack in to a single value.
    pack_target[:] = [next(value.flat)]
  else:
    pack_target[:] = np.ravel(value).tolist()
  return packed
