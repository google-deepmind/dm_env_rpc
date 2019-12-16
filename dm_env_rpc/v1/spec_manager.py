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
"""Manager class to manage the dm_env_rpc UID system."""

import numpy as np

from dm_env_rpc.v1 import tensor_utils


def _assert_shapes_match(tensor, dm_env_rpc_spec):
  """Raises ValueError if shape of tensor and spec don't match."""
  if not np.array_equal(tensor.shape, dm_env_rpc_spec.shape):
    raise ValueError(
        'Received dm_env_rpc tensor {} with shape {} but spec has shape {}.'
        .format(dm_env_rpc_spec.name, tensor.shape, dm_env_rpc_spec.shape))


class SpecManager(object):
  """Manages transitions between Python dicts and dm_env_rpc UIDs.

  To make sending and receiving actions and observations easier for dm_env_rpc,
  this helps manage the transition between UID-keyed dicts mapping to dm_env_rpc
  tensors and string-keyed dicts mapping to scalars, lists, or NumPy arrays.
  """

  def __init__(self, specs):
    """Builds the SpecManager from the given dm_env_rpc specs.

    Args:
      specs: A dict mapping UIDs to dm_env_rpc TensorSpecs, similar to what is
        stored in `actions` and `observations` in ActionObservationSpecs.
    """
    self._name_to_uid = {
        spec.name: uid for uid, spec in specs.items()
    }
    self._uid_to_name = {
        uid: spec.name for uid, spec in specs.items()
    }
    if len(self._name_to_uid) != len(self._uid_to_name):
      raise ValueError('There are duplicate names in the tensor specs.')

    self._specs_by_uid = specs
    self._specs_by_name = {
        spec.name: spec for spec in specs.values()
    }

  def name_to_uid(self, name):
    """Returns the UID for the given name."""
    return self._name_to_uid[name]

  def uid_to_name(self, uid):
    """Returns the name for the given UID."""
    return self._uid_to_name[uid]

  def name_to_spec(self, name):
    """Returns the dm_env_rpc TensorSpec named `name`."""
    return self._specs_by_name[name]

  def uid_to_spec(self, uid):
    """Returns the dm_env_rpc TensorSpec for the given UID."""
    return self._specs_by_uid[uid]

  def names(self):
    """Returns the spec names in no particular order."""
    return self._name_to_uid.keys()

  def uids(self):
    """Returns the spec UIDs in no particular order."""
    return self._uid_to_name.keys()

  def unpack(self, dm_env_rpc_tensors):
    """Unpacks a dm_env_rpc uid-to-tensor map to a name-keyed Python dict.

    Args:
      dm_env_rpc_tensors: A dict mapping UIDs to dm_env_rpc tensor protos.

    Returns:
      A dict mapping names to scalars and arrays.
    """
    unpacked = {}
    for uid, tensor in dm_env_rpc_tensors.items():
      name = self._uid_to_name[uid]
      dm_env_rpc_spec = self.name_to_spec(name)
      _assert_shapes_match(tensor, dm_env_rpc_spec)
      tensor_dtype = tensor_utils.get_tensor_type(tensor)
      spec_dtype = tensor_utils.data_type_to_np_type(dm_env_rpc_spec.dtype)
      if tensor_dtype != spec_dtype:
        raise ValueError(
            'Received dm_env_rpc tensor {} with dtype {} but spec has dtype {}.'
            .format(name, tensor_dtype, spec_dtype))
      tensor_unpacked = tensor_utils.unpack_tensor(tensor)
      unpacked[name] = tensor_unpacked
    return unpacked

  def pack(self, tensors):
    """Packs a name-keyed Python dict to a dm_env_rpc uid-to-tensor map.

    Args:
      tensors: A dict mapping string names to scalars and arrays.

    Returns:
      A dict mapping UIDs to dm_env_rpc tensor protos.
    """
    packed = {}
    for name, value in tensors.items():
      dm_env_rpc_spec = self.name_to_spec(name)
      tensor = tensor_utils.pack_tensor(value, dtype=dm_env_rpc_spec.dtype)
      _assert_shapes_match(tensor, dm_env_rpc_spec)
      packed[self.name_to_uid(name)] = tensor
    return packed
